"""Offline tests for the splice functions and gates of the edit vertical.

The free gates around span generation (placeholder / undefined name /
already tried / whole-file compile / unchanged), canonical keys, and
headers that survive a replace. No model, no network.
"""
import ast
import unittest

from threetoks.code.edit import spans, splice


class CanonicalKeyTest(unittest.TestCase):
    def test_whitespace_comments_and_docstring_do_not_change_the_key(self):
        a = 'def f(x):\n    """Doc."""\n    return x+1\n'
        b = "def f(x):\n    # a comment\n    return (x + 1)\n"
        self.assertEqual(splice.canonical_key(a), splice.canonical_key(b))
        self.assertEqual(splice.canonical_key("    " + b.replace("\n    ",
                                                               "\n        ")),
                         splice.canonical_key(b))

    def test_different_code_has_a_different_key(self):
        self.assertNotEqual(splice.canonical_key("def f(x):\n    return x\n"),
                            splice.canonical_key("def f(x):\n    return -x\n"))

    def test_unparseable_text_has_an_empty_key(self):
        self.assertEqual(splice.canonical_key("def f(:\n"), "")


class SpliceGateTest(unittest.TestCase):
    SRC = "def f(x):\n    return x + 1\n"
    SETTER_SRC = ("class Box:\n    @property\n    def width(self):\n"
                  "        return self._w\n\n    @width.setter\n"
                  "    def width(self, value):\n        value = int(value)\n"
                  "        self._w = value\n")

    def _replace(self, source, qualname, known=frozenset(), tried=None,
                 lineno=None):
        node = spans.find_def(source, qualname, lineno)
        indent = spans.leading_indent(source.split("\n")[node.lineno - 1])
        return splice.replace_splice_fn(
            source, node.lineno, node.end_lineno,
            spans.def_header(source, qualname, node.lineno), indent,
            set(known), set() if tried is None else tried)

    def test_accepts_a_clean_replacement(self):
        candidate, new_source = self._replace(self.SRC, "f")("return x * 3")
        self.assertEqual(candidate, "def f(x):\n    return x * 3\n")
        self.assertEqual(new_source, "def f(x):\n    return x * 3")

    def test_async_header_and_decorator_survive_a_replace(self):
        src = ("import functools\n\n\n@functools.cache\n"
               "async def fetch(url, retries=3, *, timeout: float = 1.0,"
               " **opts) -> dict:\n    return {}\n")
        candidate, _ = self._replace(src, "fetch", {"functools"})(
            "return {'url': url, 'n': retries, 't': timeout, **opts}")
        tree = ast.parse(candidate)
        fetch = tree.body[1]
        self.assertIsInstance(fetch, ast.AsyncFunctionDef)
        self.assertEqual(len(fetch.decorator_list), 1)
        self.assertEqual([a.arg for a in fetch.args.kwonlyargs], ["timeout"])
        self.assertEqual(fetch.args.kwarg.arg, "opts")
        self.assertEqual(len(fetch.args.defaults), 1)
        self.assertIn(src.split("\n")[4], candidate)

    def test_multi_line_method_header_survives_a_replace(self):
        classy = ("class Box:\n    def scale(self,\n"
                  "              factor=2):\n        return self.v\n")
        candidate, _ = self._replace(classy, "Box.scale")(
            "return self.v * factor")
        self.assertEqual(candidate, (
            "class Box:\n    def scale(self,\n              factor=2):\n"
            "        return self.v * factor\n"))

    def test_one_liner_is_replaced_by_a_full_def(self):
        src = "def f(x): return x\n\n\ndef g():\n    return f(1)\n"
        candidate, _ = self._replace(src, "f")("return x * 2")
        self.assertTrue(candidate.startswith(
            "def f(x):\n    return x * 2\n\n\ndef g():"))

    def test_docstring_is_kept_when_the_model_drops_it(self):
        src = 'def f(x):\n    """Add one.\n\n    More text.\n    """\n' \
              "    return x + 1\n"
        candidate, new_source = self._replace(src, "f")("return x + 2")
        self.assertEqual(ast.get_docstring(ast.parse(candidate).body[0]),
                         "Add one.\n\nMore text.")
        self.assertIn("return x + 2", new_source)

    def test_model_returning_the_original_is_unchanged(self):
        src = 'def f(x):\n    """Doc."""\n    return x + 1\n'
        fn = self._replace(src, "f")
        self.assertEqual(fn("return x+1"), splice.UNCHANGED)
        self.assertEqual(fn('"""Other doc."""\n    return (x + 1)  # same'),
                         splice.UNCHANGED)
        node = splice.GenerateSpanNode("q", "p", fn)
        decision = node.parse("return x+1", ())
        self.assertTrue(decision.valid)
        self.assertEqual(decision.value, splice.UNCHANGED)

    def test_rejects_syntax_break_and_placeholder_for_free(self):
        fn = self._replace(self.SRC, "f")
        self.assertIsNone(fn("return ((("))
        self.assertIsNone(fn("raise NotImplementedError"))
        self.assertFalse(splice.GenerateSpanNode("q", "p", fn)
                         .parse("return (((", ()).valid)

    def test_undefined_name_in_body_is_rejected(self):
        self.assertIsNone(self._replace(self.SRC, "f")("return clamp(x)"))
        self.assertIsNotNone(
            self._replace(self.SRC, "f", {"clamp"})("return clamp(x)"))

    def test_already_tried_candidate_is_rejected(self):
        tried = {splice.canonical_key("def f(x):\n    return x - 1")}
        fn = self._replace(self.SRC, "f", tried=tried)
        self.assertIsNone(fn("return x - 1"))
        self.assertIsNone(fn("return (x-1)  # whitespace differs"))
        self.assertIsNotNone(fn("return x * 2"))

    def test_method_replacement_is_reindented_into_the_class(self):
        classy = "class Box:\n    def get(self):\n        return self.v + 1\n"
        candidate, method_src = self._replace(classy, "Box.get")(
            "return self.v - 1")
        self.assertIn("    def get(self):\n        return self.v - 1",
                      candidate)
        self.assertTrue(method_src.startswith("def get(self):"))
        ast.parse(candidate)

    def test_insert_after_adds_a_gated_helper(self):
        src = "def f(x):\n    return _twice(x)\n"
        fn = splice.insert_after_splice_fn(src, 2, "_twice", "arg1", set(),
                                        set())
        candidate, helper = fn("return arg1 * 2")
        self.assertEqual(candidate, src + "\ndef _twice(arg1):\n"
                                          "    return arg1 * 2\n")
        self.assertEqual(helper, "def _twice(arg1):\n    return arg1 * 2")
        self.assertIsNone(fn("return lookup[arg1]"))
        self.assertIsNone(fn("pass"))

    def test_setter_replace_keeps_the_setter_header(self):
        candidate, new_source = self._replace(self.SETTER_SRC, "Box.width",
                                              lineno=7)(
            "self._w = max(0, value)")
        self.assertEqual(new_source, "def width(self, value):\n"
                                     "    self._w = max(0, value)")
        self.assertIn("    @width.setter\n    def width(self, value):\n"
                      "        self._w = max(0, value)\n", candidate)
        self.assertIn("    def width(self):\n        return self._w\n",
                      candidate)

    def test_overload_implementation_keeps_its_own_header(self):
        src = ("from typing import overload\n\n\n@overload\n"
               "def scale(x: int) -> int: ...\n\n\n"
               "def scale(x, factor=2):\n    return x\n")
        candidate, new_source = self._replace(src, "scale", lineno=8)(
            "return x * factor")
        self.assertEqual(new_source,
                         "def scale(x, factor=2):\n    return x * factor")
        self.assertIn("@overload\ndef scale(x: int) -> int: ...\n", candidate)

    def test_star_import_file_accepts_names_it_may_provide(self):
        src = "from math import *\n\n\ndef hyp(a, b):\n    return a * b\n"
        candidate, _ = self._replace(src, "hyp")("return sqrt(a * a + b * b)")
        self.assertIn("return sqrt(a * a + b * b)", candidate)
        self.assertIsNone(self._replace(src, "hyp")("pass"))


if __name__ == "__main__":
    unittest.main()
