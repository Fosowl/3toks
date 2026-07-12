"""Offline tests for the lenient plan parser."""
import unittest

from threetoks.code.plan import MAX_METHODS, ParsedMethod, parse_plan


class ParseLineRecoveryTest(unittest.TestCase):
    def test_recovers_name_args_contract_from_a_clean_line(self):
        methods = parse_plan("count_vowels(text): counts vowels")
        self.assertEqual(len(methods), 1)
        self.assertEqual(
            (methods[0].name, methods[0].args, methods[0].contract),
            ("count_vowels", "text", "counts vowels"))

    def test_strips_dotted_list_numbering(self):
        methods = parse_plan("1. tally(words): count each word")
        self.assertEqual(methods[0].name, "tally")

    def test_strips_paren_list_numbering(self):
        methods = parse_plan("2) tokenize(text): split into words")
        self.assertEqual(methods[0].name, "tokenize")

    def test_strips_a_def_prefix(self):
        methods = parse_plan("def parse(text): parse it")
        self.assertEqual(methods[0].name, "parse")

    def test_dash_separator_without_parens_yields_empty_args(self):
        methods = parse_plan("normalize - lowercase the text")
        self.assertEqual(methods[0].name, "normalize")
        self.assertEqual(methods[0].args, "")
        self.assertEqual(methods[0].contract, "lowercase the text")

    def test_equals_separator_is_accepted(self):
        methods = parse_plan("tokenize(text) = split into words")
        self.assertEqual(methods[0].args, "text")
        self.assertEqual(methods[0].contract, "split into words")

    def test_missing_contract_gets_a_default(self):
        methods = parse_plan("helper():")
        self.assertEqual(methods[0].contract, "implement helper")


class ProseSkipTest(unittest.TestCase):
    def test_prose_line_with_no_structural_signal_is_skipped(self):
        # No parentheses and no :/-/= separator after the first word.
        methods = parse_plan("just some prose with no identifier signal")
        self.assertEqual(methods, [])

    def test_line_not_starting_with_an_identifier_is_skipped(self):
        self.assertEqual(parse_plan("!!! nonsense line"), [])


class DedupeAndCapTest(unittest.TestCase):
    def test_duplicate_names_are_dropped_keeping_the_first(self):
        methods = parse_plan(
            "count(text): first\n1. count(text): duplicate dropped")
        self.assertEqual([m.name for m in methods], ["count"])

    def test_caps_at_max_methods(self):
        plan = "\n".join(f"m{i}(): purpose {i}" for i in range(MAX_METHODS + 5))
        methods = parse_plan(plan)
        self.assertEqual(len(methods), MAX_METHODS)


class ExampleClauseTest(unittest.TestCase):
    def test_eq_comparison_example_becomes_advisory_assert(self):
        methods = parse_plan("count(text): counts; e.g. count('hi')==1")
        self.assertEqual(methods[0].advisory_assert, "assert count('hi')==1")

    def test_non_comparison_example_yields_no_assert(self):
        methods = parse_plan("build(x): construct; e.g. build(1)")
        self.assertIsNone(methods[0].advisory_assert)

    def test_example_clause_is_stripped_from_the_contract(self):
        methods = parse_plan("count(text): counts vowels; e.g. count('hi')==1")
        self.assertNotIn("e.g.", methods[0].contract)
        self.assertIn("counts vowels", methods[0].contract)


class ParsedMethodShapeTest(unittest.TestCase):
    def test_parsed_methods_are_the_dataclass(self):
        methods = parse_plan("f(x): do")
        self.assertIsInstance(methods[0], ParsedMethod)
        self.assertIsNone(methods[0].advisory_assert)


if __name__ == "__main__":
    unittest.main()
