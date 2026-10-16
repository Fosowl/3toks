"""Offline tests for the symbol index and the paged nav menus.

No model, no network.
"""
import tempfile
import unittest
from pathlib import Path

from threetoks.code.edit import nav
from threetoks.code.edit.index import SymbolIndex


def _corpus(tree: dict[str, str]) -> tuple[tempfile.TemporaryDirectory, Path]:
    holder = tempfile.TemporaryDirectory()
    root = Path(holder.name)
    for rel, source in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return holder, root


class SymbolIndexTest(unittest.TestCase):
    def setUp(self):
        self._holder, self.root = _corpus({
            "a/one.py": ('def foo(x):\n    """Adds via helper."""\n'
                         "    return helper(x)\n\n\n"
                         "def helper(x):\n    return x + 1\n\n\n"
                         "class Thing:\n    def bar(self):\n        return 1\n"),
            "a/two.py": "def foo(y):\n    return y\n",
        })
        self.addCleanup(self._holder.cleanup)
        self.index = SymbolIndex(self.root)

    def test_collects_functions_classes_and_methods(self):
        names = {(s.qualname, s.kind) for s in self.index.symbols}
        self.assertEqual(names, {("foo", "function"), ("helper", "function"),
                                 ("Thing", "class"), ("Thing.bar", "method"),
                                 ("foo", "function")})

    def test_unique_name_resolves_free_and_ambiguous_lists_all(self):
        self.assertEqual(len(self.index.find_by_name("helper")), 1)
        self.assertEqual(len(self.index.find_by_name("foo")), 2)

    def test_docstring_first_line_is_captured(self):
        foo = self.index.symbol_at("a/one.py", "foo")
        self.assertEqual(foo.doc, "Adds via helper.")

    def test_call_sites_key_on_bare_callee_name(self):
        self.assertEqual(self.index.call_sites["helper"], [("a/one.py", 3)])

    def test_unparseable_file_is_skipped_without_crashing(self):
        (self.root / "a" / "broken.py").write_text("def broken(:\n")
        rebuilt = SymbolIndex(self.root)
        self.assertNotIn("broken", {s.name for s in rebuilt.symbols})


class NavTest(unittest.TestCase):
    def test_zero_or_one_item_needs_no_menu(self):
        self.assertIsNone(nav.level_menu("q", [], 0))
        self.assertIsNone(nav.level_menu("q", ["only"], 0))
        self.assertEqual(nav.LevelChoice("q", ["only"]).free_take(), "only")

    def test_overflow_pages_instead_of_widening(self):
        many = [f"item{i}" for i in range(12)]
        first = nav.level_menu("q", many, 0)
        self.assertIn(nav.MORE_LABEL, first.options)
        self.assertEqual(len(first.options), nav.PAGE_SIZE)
        self.assertNotIn(nav.MORE_LABEL, nav.level_menu("q", many, 1).options)

    def test_more_advances_page_and_escape_returns_none(self):
        many = [f"item{i}" for i in range(12)]
        choice = nav.LevelChoice("q", many)
        node = choice.node()
        perm = tuple(range(len(node.options)))
        more_digit = node.options.index(nav.MORE_LABEL) + 1
        self.assertEqual(choice.apply(node.parse(f" {more_digit}", perm)),
                         nav.MORE_LABEL)
        self.assertEqual(choice.page, 1)
        pair = nav.LevelChoice("q", ["x", "y"])
        n = pair.node()
        p = tuple(range(len(n.options)))
        self.assertIsNone(pair.apply(n.parse(f" {len(n.options) + 1}", p)))


if __name__ == "__main__":
    unittest.main()
