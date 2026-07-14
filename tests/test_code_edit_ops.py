"""Offline tests for the edit vertical's deterministic machinery.

Index, navigation paging, splice primitives, the free gates around span
generation (placeholder / oscillation / whole-file parse), and the
trusted-oracle runner. No model, no network.
"""
import ast
import tempfile
import unittest
from pathlib import Path

from threetoks.code.edit import nav, ops, oracle
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


class SpliceTest(unittest.TestCase):
    SRC = "def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y\n"

    def test_replace_and_insert_keep_the_file_parseable(self):
        replaced = ops.replace_span(self.SRC, 1, 2, "def f(x):\n    return 9")
        inserted = ops.insert_after_span(self.SRC, 2, "def h(z):\n    return z")
        ast.parse(replaced)
        ast.parse(inserted)
        self.assertIn("return 9", replaced)
        self.assertIn("def h(z):", inserted)

    def test_delete_is_a_raw_primitive_the_caller_must_gate(self):
        # Deleting a function's only statement legitimately breaks parsing;
        # the vertical pre-filters such spans out of the delete menu.
        deleted = ops.delete_span(self.SRC, 2, 2)
        self.assertNotIn("return x + 1", deleted)
        with self.assertRaises(SyntaxError):
            ast.parse(deleted)

    def test_span_text_is_verbatim(self):
        self.assertEqual(ops.span_text(self.SRC, 2, 2), "    return x + 1")

    def test_statement_spans_work_for_methods_too(self):
        classy = ("class Box:\n    def get(self):\n        a = 1\n"
                  "        return a\n")
        self.assertEqual(ops.statement_spans(classy, "Box.get"),
                         [(3, 3), (4, 4)])
        self.assertEqual(ops.def_args(classy, "Box.get"), "self")

    def test_find_def_resolves_qualnames(self):
        classy = "class A:\n    def m(self):\n        return 0\n"
        self.assertEqual(ops.find_def(classy, "A.m").name, "m")
        self.assertIsNone(ops.find_def(classy, "A.zz"))
        self.assertIsNone(ops.find_def("def broken(:\n", "broken"))


class SpliceGateTest(unittest.TestCase):
    SRC = "def f(x):\n    return x + 1\n"

    def test_accepts_a_clean_replacement(self):
        fn = ops.replace_splice_fn(self.SRC, 1, 2, "f", "x")
        candidate, new_source = fn("return x * 3")
        self.assertIn("return x * 3", candidate)
        self.assertTrue(new_source.startswith("def f(x):"))

    def test_rejects_syntax_break_and_placeholder_for_free(self):
        fn = ops.replace_splice_fn(self.SRC, 1, 2, "f", "x")
        self.assertIsNone(fn("return ((("))
        self.assertIsNone(fn("raise NotImplementedError"))

    def test_oscillation_guard_rejects_an_already_seen_body(self):
        seen = {hash("def f(x):\n    return x + 1")}   # the original bug
        fn = ops.replace_splice_fn(self.SRC, 1, 2, "f", "x", seen)
        self.assertIsNone(fn("return x + 1"))          # regenerated the bug
        self.assertIsNotNone(fn("return x - 1"))

    def test_method_replacement_is_reindented_into_the_class(self):
        classy = ("class Box:\n    def get(self):\n        return self.v + 1\n")
        fn = ops.replace_splice_fn(classy, 2, 3, "get", "self", indent="    ")
        candidate, _ = fn("return self.v - 1")
        self.assertIn("    def get(self):", candidate)
        ast.parse(candidate)

    def test_missing_helper_is_found_with_name_and_arity(self):
        src = ("def average(nums):\n    total = 0\n"
               "    for n in nums:\n        total = _safe_add(total, n)\n"
               "    return total\n")
        self.assertEqual(ops.find_missing_helper(src, "average", {"average"}),
                         ("_safe_add", "a, b"))
        clean = "def f(x):\n    return x\n"
        self.assertIsNone(ops.find_missing_helper(clean, "f", {"f"}))


class OracleTest(unittest.TestCase):
    def test_fails_before_passes_after_and_writes_no_bytecode(self):
        holder, root = _corpus({"mod.py": "def add(a, b):\n    return a - b\n"})
        self.addCleanup(holder.cleanup)
        test = "from mod import add\nassert add(2, 2) == 4\n"
        ok, err = oracle.run_test(root, test)
        self.assertFalse(ok)
        self.assertIn("AssertionError", err)
        (root / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        self.assertEqual(oracle.run_test(root, test), (True, ""))
        self.assertEqual(list(root.rglob("__pycache__")), [])


if __name__ == "__main__":
    unittest.main()
