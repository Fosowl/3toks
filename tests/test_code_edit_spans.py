"""Offline tests for the span helpers of the edit vertical.

Splice primitives over 1-based inclusive line spans, def lookup that
tells same-named defs apart by line, and verbatim def headers. No model,
no network.
"""
import ast
import unittest

from threetoks.code.edit import spans

# Same-named defs: property getter (line 3) and setter (line 7).
SETTER_SRC = ("class Box:\n    @property\n    def width(self):\n"
              "        return self._w\n\n    @width.setter\n"
              "    def width(self, value):\n        value = int(value)\n"
              "        self._w = value\n")


class SpliceTest(unittest.TestCase):
    SRC = "def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y\n"

    def test_replace_and_insert_keep_the_file_parseable(self):
        replaced = spans.replace_span(self.SRC, 1, 2,
                                      "def f(x):\n    return 9")
        inserted = spans.insert_after_span(self.SRC, 2,
                                           "def h(z):\n    return z")
        ast.parse(replaced)
        ast.parse(inserted)
        self.assertIn("return 9", replaced)
        self.assertIn("def h(z):", inserted)

    def test_trailing_newline_of_the_file_is_kept_as_it_was(self):
        bare = self.SRC.rstrip("\n")
        self.assertTrue(spans.replace_span(self.SRC, 6, 6, "    return 0")
                        .endswith("return 0\n"))
        self.assertTrue(spans.replace_span(bare, 6, 6, "    return 0")
                        .endswith("return 0"))

    def test_delete_is_a_raw_primitive_the_caller_must_gate(self):
        # Deleting a function's only statement legitimately breaks parsing;
        # the vertical pre-filters such spans out of the delete menu.
        deleted = spans.delete_span(self.SRC, 2, 2)
        self.assertNotIn("return x + 1", deleted)
        with self.assertRaises(SyntaxError):
            ast.parse(deleted)

    def test_span_text_is_verbatim(self):
        self.assertEqual(spans.span_text(self.SRC, 2, 2), "    return x + 1")

    def test_statement_spans_work_for_methods_too(self):
        classy = ("class Box:\n    def get(self):\n        a = 1\n"
                  "        return a\n")
        self.assertEqual(spans.statement_spans(classy, "Box.get", 2),
                         [(3, 3), (4, 4)])

    def test_statement_spans_never_offer_the_docstring(self):
        src = 'def f(x):\n    """Doc."""\n    y = x\n    return y\n'
        self.assertEqual(spans.statement_spans(src, "f", 1), [(3, 3), (4, 4)])

    def test_statement_spans_pick_the_same_named_def_by_line(self):
        self.assertEqual(spans.statement_spans(SETTER_SRC, "Box.width", 3),
                         [(4, 4)])
        self.assertEqual(spans.statement_spans(SETTER_SRC, "Box.width", 7),
                         [(8, 8), (9, 9)])
        self.assertEqual(spans.statement_spans(SETTER_SRC, "Box.width", 5), [])

    def test_form_feed_in_file_does_not_shift_spans(self):
        src = "import os\n\x0c\ndef f(x):\n    y = x\n    return y\n"
        found = spans.statement_spans(src, "f", 3)
        self.assertEqual([spans.span_text(src, *s) for s in found],
                         ["    y = x", "    return y"])
        deleted = spans.delete_span(src, *found[0])
        self.assertIn("\x0c", deleted)
        self.assertNotIn("y = x", deleted)

    def test_find_def_resolves_qualnames(self):
        classy = "class A:\n    def m(self):\n        return 0\n"
        self.assertEqual(spans.find_def(classy, "A.m").name, "m")
        self.assertIsNone(spans.find_def(classy, "A.zz"))
        self.assertIsNone(spans.find_def("def broken(:\n", "broken"))

    def test_find_def_selects_a_same_named_def_by_line(self):
        getter = spans.find_def(SETTER_SRC, "Box.width")
        self.assertEqual(len(getter.args.args), 1)
        setter = spans.find_def(SETTER_SRC, "Box.width", 7)
        self.assertEqual([a.arg for a in setter.args.args], ["self", "value"])
        self.assertIsNone(spans.find_def(SETTER_SRC, "Box.width", 6))


class DefHeaderTest(unittest.TestCase):
    def test_async_defaults_kwonly_and_kwargs_are_verbatim(self):
        src = ("async def fetch(url, retries=3, *, timeout: float = 1.0,"
               " **opts) -> dict:\n    return {}\n")
        self.assertEqual(spans.def_header(src, "fetch", 1),
                         "async def fetch(url, retries=3, *, "
                         "timeout: float = 1.0, **opts) -> dict:")

    def test_multi_line_method_header_is_kept_and_dedented(self):
        classy = ("class Box:\n    @property\n    def scale(self,\n"
                  "              factor=2):\n        return factor\n")
        self.assertEqual(spans.def_header(classy, "Box.scale", 3),
                         "def scale(self,\n          factor=2):")

    def test_one_liner_header_stops_at_the_colon(self):
        self.assertEqual(spans.def_header("def f(x): return x\n", "f", 1),
                         "def f(x):")

    def test_colons_inside_brackets_do_not_end_the_header(self):
        src = ("def f(key=lambda v: v, d: dict[str, int] = {}) -> int:\n"
               "    return 0\n")
        self.assertEqual(spans.def_header(src, "f", 1), src.split("\n")[0])

    def test_setter_header_is_selected_by_line(self):
        self.assertEqual(spans.def_header(SETTER_SRC, "Box.width", 3),
                         "def width(self):")
        self.assertEqual(spans.def_header(SETTER_SRC, "Box.width", 7),
                         "def width(self, value):")

    def test_unknown_target_has_no_header(self):
        self.assertIsNone(spans.def_header("def f():\n    return 1\n", "g", 1))
        self.assertIsNone(spans.def_header("def f():\n    return 1\n", "f", 2))


if __name__ == "__main__":
    unittest.main()
