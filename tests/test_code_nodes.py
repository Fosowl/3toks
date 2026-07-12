"""Offline tests for the coding generation nodes (plan/implement/test)."""
import ast
import unittest

from threetoks.code.nodes import (IMPLEMENT_SYSTEM, ImplementNode, PlanNode,
                                 TestNode)

PERM = ()  # option-less generation nodes ignore the permutation


class PlanNodeTest(unittest.TestCase):
    def test_prefill_is_fed_to_the_parser_so_the_first_method_is_recovered(self):
        # The completion text lacks the "1. " list marker; the prefill supplies
        # it, so the first line must still parse into a method.
        node = PlanNode("count words")
        decision = node.parse("count_vowels(text): counts vowels", PERM)
        self.assertTrue(decision.valid)
        self.assertEqual([m.name for m in decision.value], ["count_vowels"])

    def test_multi_method_plan_parses_all(self):
        node = PlanNode("g")
        decision = node.parse(
            "count(text): counts\n2. tally(words): tallies", PERM)
        self.assertEqual([m.name for m in decision.value], ["count", "tally"])

    def test_invalid_when_no_method_recovered(self):
        node = PlanNode("g")
        decision = node.parse("", PERM)
        self.assertFalse(decision.valid)
        self.assertEqual(decision.value, [])


class ImplementNodeParseTest(unittest.TestCase):
    def test_valid_when_reconstructed_source_parses_as_the_named_def(self):
        node = ImplementNode("clamp", "value, low, high", "limit to range")
        decision = node.parse("return max(low, min(high, value))", PERM)
        self.assertTrue(decision.valid)
        self.assertIn("def clamp", decision.value)
        ast.parse(decision.value)

    def test_value_is_the_reconstructed_source(self):
        node = ImplementNode("f", "x", "c")
        decision = node.parse("return x", PERM)
        self.assertEqual(decision.value, "def f(x):\n    return x")

    def test_invalid_on_a_syntax_error(self):
        node = ImplementNode("f", "x", "c")
        decision = node.parse("return (((", PERM)
        self.assertFalse(decision.valid)
        self.assertIsNone(decision.value)


class ImplementNodeConstructionTest(unittest.TestCase):
    def test_temperature_is_none_by_default(self):
        self.assertIsNone(ImplementNode("f", "x", "c").temperature)

    def test_temperature_is_carried_when_set_for_repair(self):
        node = ImplementNode("f", "x", "c", temperature=0.4)
        self.assertEqual(node.temperature, 0.4)

    def test_system_is_the_implement_system_prompt(self):
        self.assertEqual(ImplementNode("f", "x", "c").system, IMPLEMENT_SYSTEM)

    def test_prefill_supplies_the_def_line_and_four_space_indent(self):
        self.assertEqual(ImplementNode("f", "x", "c").prefill,
                         "def f(x):\n    ")


class TestNodeTest(unittest.TestCase):
    def test_parse_is_always_valid(self):
        # Asserts are advisory, so the ladder must never resample on them.
        decision = TestNode("add", "a, b").parse("total garbage", PERM)
        self.assertTrue(decision.valid)

    def test_keeps_single_assert_lines_that_mention_the_function(self):
        node = TestNode("add", "a, b")
        decision = node.parse("add(1, 2) == 3\nassert add(0, 0) == 0", PERM)
        self.assertEqual(decision.value,
                         ["assert add(1, 2) == 3", "assert add(0, 0) == 0"])

    def test_drops_tautologies_that_do_not_mention_the_function(self):
        node = TestNode("add", "a, b")
        decision = node.parse("assert True\n1 == 1", PERM)
        self.assertEqual(decision.value, [])

    def test_drops_a_line_that_does_not_parse(self):
        node = TestNode("add", "a, b")
        decision = node.parse("assert add(((", PERM)
        self.assertEqual(decision.value, [])


if __name__ == "__main__":
    unittest.main()
