"""Unit tests for decision-node rendering and parsing."""
import unittest

from threetoks.nodes import (ESCAPE, MenuNode, PickManyNode, ShortTextNode,
                            confirm_node)


class MenuNodeTest(unittest.TestCase):
    def setUp(self):
        self.menu = MenuNode("Pick.", ["alpha", "beta", "gamma"])
        self.perm = (2, 0, 1)

    def test_render_uses_permuted_order(self):
        rendered = self.menu.render(self.perm)
        self.assertIn("1 = gamma", rendered)
        self.assertIn("3 = beta", rendered)
        self.assertIn("4 = none of these fit", rendered)
        self.assertNotIn("0 =", rendered)

    def test_parse_maps_digit_through_permutation(self):
        self.assertEqual(self.menu.parse(" 1", self.perm).value, "gamma")
        self.assertEqual(self.menu.parse("2.", self.perm).value, "alpha")

    def test_parse_last_numbered_option_is_escape(self):
        decision = self.menu.parse("4", self.perm)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.value, ESCAPE)

    def test_parse_rejects_zero_garbage_and_out_of_range(self):
        self.assertFalse(self.menu.parse("0", self.perm).valid)
        self.assertFalse(self.menu.parse("sure!", self.perm).valid)
        self.assertFalse(self.menu.parse("7", self.perm).valid)

    def test_option_count_bounds(self):
        with self.assertRaises(ValueError):
            MenuNode("q", [])
        with self.assertRaises(ValueError):
            MenuNode("q", [str(i) for i in range(9)])  # 9 + escape > 9
        MenuNode("q", [str(i) for i in range(9)], escape=False)  # ok

    def test_confirm_has_no_escape(self):
        confirm = confirm_node("Sure?")
        self.assertFalse(confirm.parse("0", (0, 1)).valid)
        self.assertFalse(confirm.parse("3", (0, 1)).valid)
        self.assertEqual(confirm.parse("2", (0, 1)).value, "no")


class PickManyNodeTest(unittest.TestCase):
    def test_parse_dedupes_and_bounds(self):
        node = PickManyNode("Which?", 10)
        self.assertEqual(node.parse(" 3, 7, 3, 12", ()).value, [3, 7])

    def test_parse_caps_over_picking(self):
        node = PickManyNode("Which?", 10, max_picks=2)
        self.assertEqual(node.parse("1,2,3,4,5", ()).value, [1, 2])

    def test_has_newline_stop_for_clean_termination(self):
        self.assertIn("\n", PickManyNode("Which?", 5).stop)

    def test_parse_only_first_line(self):
        node = PickManyNode("Which?", 10)
        self.assertEqual(node.parse("2,5\n9", ()).value, [2, 5])

    def test_empty_is_invalid(self):
        self.assertFalse(PickManyNode("Which?", 5).parse("none", ()).valid)


class ShortTextNodeTest(unittest.TestCase):
    def test_parse_takes_first_line_stripped(self):
        node = ShortTextNode("Query?", "QUERY:")
        self.assertEqual(node.parse(" france population \nx", ()).value,
                         "france population")

    def test_empty_is_invalid(self):
        self.assertFalse(ShortTextNode("Query?", "QUERY:").parse("  ", ()).valid)

    def test_template_markers_are_stripped(self):
        node = ShortTextNode("Query?", "QUERY:")
        self.assertEqual(node.parse("1<|im_end|>", ()).value, "1")
        self.assertEqual(node.parse(" query<|endoftext|>\nx", ()).value,
                         "query")
        self.assertFalse(node.parse("<|im_end|>", ()).valid)  # marker only


if __name__ == "__main__":
    unittest.main()
