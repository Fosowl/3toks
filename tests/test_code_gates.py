"""Offline tests for the deterministic gates.

run_asserts / import_ok shell out to a real python3 subprocess (that is the
gate's design); the modules under test are tiny so these stay fast.
"""
import ast
import unittest

from threetoks.code import gates


class ReconstructTest(unittest.TestCase):
    def test_strips_stray_leading_space_so_body_lands_at_four_spaces(self):
        # Regression: " return x" would otherwise indent the first body line
        # to five spaces, and a later inserted docstring at four -> IndentError.
        source = gates.reconstruct("f", "x", " return x")
        self.assertEqual(source, "def f(x):\n    return x")
        ast.parse(source)

    def test_leaves_a_clean_body_untouched(self):
        source = gates.reconstruct("f", "x", "return x")
        self.assertEqual(source, "def f(x):\n    return x")

    def test_reconstructed_then_documented_source_still_parses(self):
        source = gates.reconstruct("f", "x", " return x")  # stray space
        documented = gates.ensure_docstring(source, "doc")
        ast.parse(documented)  # the IndentationError regression guard


class FunctionDefTest(unittest.TestCase):
    def test_finds_a_sync_def_by_name(self):
        node = gates.function_def("def f(x):\n    return x\n", "f")
        self.assertIsInstance(node, ast.FunctionDef)

    def test_finds_an_async_def_by_name(self):
        node = gates.function_def("async def f(x):\n    return x\n", "f")
        self.assertIsInstance(node, ast.AsyncFunctionDef)

    def test_none_on_syntax_error(self):
        self.assertIsNone(gates.function_def("def f(:\n    bad", "f"))

    def test_none_when_name_does_not_match(self):
        self.assertIsNone(gates.function_def("def g(x):\n    return x\n", "f"))


class ImportAwareUndefinedNamesTest(unittest.TestCase):
    """Import statements bind names (live gemma failure: every legitimate
    `import requests` weather body was falsely rejected and stubbed)."""

    def test_plain_import_binds_the_module_name(self):
        body = ("def main():\n    import requests\n"
                "    return requests.get('http://x').text\n")
        self.assertEqual(gates.undefined_names(body, set()), [])

    def test_from_import_and_aliases_bind_their_names(self):
        body = ("def f():\n"
                "    from json import loads as parse\n"
                "    import os.path as p\n"
                "    return parse('1'), p.sep\n")
        self.assertEqual(gates.undefined_names(body, set()), [])

    def test_genuinely_undefined_names_are_still_caught(self):
        body = "def f(x):\n    import re\n    return helper(x)\n"
        self.assertEqual(gates.undefined_names(body, set()), ["helper"])


class RenormalizeIndentFallbackTest(unittest.TestCase):
    """function_source's second chance for whole-body indent drift (E8)."""

    def test_recovers_a_body_drifted_to_five_spaces(self):
        # Live qwen2.5:1.5b-instruct drift: docstring + every body line at 5.
        drifted = '"""doc"""\n     start = (p - 1) * s\n     return start'
        source = gates.function_source("f", "p, s", drifted)
        self.assertIsNotNone(source)
        ast.parse(source)
        self.assertNotIn("\n     ", source)

    def test_never_shifts_a_valid_block_opening_body(self):
        # A body whose continuation lines are ALL legitimately deeper (the
        # first line opens a for-block) must not be dedented: it is valid
        # as-is, so the fallback must never run on it.
        loop = "for n in nums:\n        total += n\n        count += 1"
        source = gates.function_source("f", "nums", loop)
        self.assertIsNotNone(source)
        self.assertIn("\n        total += n", source)

    def test_preserves_relative_nesting_when_it_does_shift(self):
        drifted = "if x:\n         return 1\n     return 0"   # 9 / 5 spaces
        source = gates.function_source("f", "x", drifted)
        self.assertIsNotNone(source)
        self.assertIn("\n        return 1", source)   # inner stays one deeper
        self.assertIn("\n    return 0", source)

    def test_single_line_completion_is_untouched(self):
        self.assertEqual(gates.renormalize_indent("return x"), "return x")

    def test_unrecoverable_completion_still_returns_none(self):
        self.assertIsNone(gates.function_source("f", "x", "return ((("))


class IsPlaceholderTest(unittest.TestCase):
    CHEATS = ("pass", "...", "raise NotImplementedError",
              "raise NotImplementedError('todo')",
              "return NotImplementedError",
              "return NotImplementedError('todo')",
              "return", "return None", "return ...",
              "'only a docstring'", "'doc'\n    pass")

    REAL = ("return x * 2", "return 0", "return ''", "return []",
            "print(x)", "'doc'\n    return x",
            "print(x)\n    return None")   # multi-statement: not a dodge

    def test_every_cheat_form_is_caught(self):
        for body in self.CHEATS:
            source = f"def f(x):\n    {body}"
            self.assertTrue(gates.is_placeholder(source, "f"), body)

    def test_real_bodies_are_never_flagged(self):
        for body in self.REAL:
            source = f"def f(x):\n    {body}"
            self.assertFalse(gates.is_placeholder(source, "f"), body)


class ExecuteTest(unittest.TestCase):
    def test_failure_returns_the_full_traceback(self):
        ok, stderr = gates.execute("def f():\n    return 1 // 0\nf()\n")
        self.assertFalse(ok)
        self.assertIn('File "<stdin>", line', stderr)   # blame needs frames
        self.assertIn("ZeroDivisionError", stderr)

    def test_success_returns_ok(self):
        self.assertTrue(gates.execute("x = 1\n")[0])


class SignatureMatchesTest(unittest.TestCase):
    _SRC = "def f(a, b):\n    return a + b\n"

    def test_true_on_exact_arg_name_match(self):
        self.assertTrue(gates.signature_matches(self._SRC, "f", "a, b"))

    def test_false_on_differing_args(self):
        self.assertFalse(gates.signature_matches(self._SRC, "f", "a"))

    def test_ignores_type_annotations_in_plan_args(self):
        src = "def f(a, b):\n    return a\n"
        self.assertTrue(gates.signature_matches(src, "f", "a: int, b: str"))

    def test_ignores_defaults_in_plan_args(self):
        src = "def f(a, b):\n    return a\n"
        self.assertTrue(gates.signature_matches(src, "f", "a, b=3"))

    def test_false_when_def_is_missing(self):
        self.assertFalse(gates.signature_matches("x = 1\n", "f", "a"))


class UndefinedNamesTest(unittest.TestCase):
    def test_reports_a_name_defined_nowhere(self):
        src = "def f(x):\n    return helper(x)\n"
        self.assertEqual(gates.undefined_names(src, set()), ["helper"])

    def test_allowed_name_is_not_reported(self):
        src = "def f(x):\n    return helper(x)\n"
        self.assertEqual(gates.undefined_names(src, {"helper"}), [])

    def test_builtin_is_not_reported(self):
        src = "def f(x):\n    return len(x)\n"
        self.assertEqual(gates.undefined_names(src, set()), [])

    def test_no_false_positive_on_comprehension_bound_name(self):
        # A correct comprehension must never be rejected: `w` is bound.
        src = "def f(xs):\n    return [w for w in xs if w]\n"
        self.assertEqual(gates.undefined_names(src, set()), [])

    def test_no_false_positive_on_for_loop_bound_name(self):
        src = ("def f(xs):\n"
               "    total = 0\n"
               "    for item in xs:\n"
               "        total = total + item\n"
               "    return total\n")
        self.assertEqual(gates.undefined_names(src, set()), [])

    def test_empty_list_on_syntax_error(self):
        self.assertEqual(gates.undefined_names("def f(:\n bad", set()), [])


class EnsureDocstringTest(unittest.TestCase):
    def test_inserts_a_docstring_when_missing(self):
        out = gates.ensure_docstring("def f(x):\n    return x\n", "return x")
        self.assertEqual(ast.get_docstring(ast.parse(out).body[0]), "return x")

    def test_keeps_an_existing_docstring_unchanged(self):
        src = 'def f(x):\n    """already here"""\n    return x\n'
        self.assertEqual(gates.ensure_docstring(src, "new contract"), src)

    def test_null_byte_never_crashes_the_ast_gates(self):
        # A stray null byte makes ast.parse raise ValueError, not
        # SyntaxError; the gates must swallow it so the episode resamples.
        assert gates.function_source("f", "x", "return x\x00") is None
        assert gates.function_def("return x\x00", "f") is None
        assert gates.undefined_names("def f():\n return g()\x00", set()) == []
        assert gates.ensure_docstring("def f():\x00", "c") == "def f():\x00"

    def test_returns_source_unchanged_on_syntax_error(self):
        bad = "def f(:\n bad"
        self.assertEqual(gates.ensure_docstring(bad, "c"), bad)


class RunAssertsTest(unittest.TestCase):
    _MODULE = "def add(a, b):\n    return a + b\n"

    def test_passing_asserts_return_true_and_empty_error(self):
        self.assertEqual(
            gates.run_asserts(self._MODULE, ["assert add(1, 2) == 3"]),
            (True, ""))

    def test_failing_assert_returns_false_and_error_line(self):
        ok, err = gates.run_asserts(self._MODULE, ["assert add(1, 2) == 4"])
        self.assertFalse(ok)
        self.assertIn("AssertionError", err)

    def test_no_asserts_trivially_passes(self):
        self.assertEqual(gates.run_asserts(self._MODULE, []), (True, ""))


class ImportOkTest(unittest.TestCase):
    def test_clean_module_imports(self):
        self.assertTrue(gates.import_ok("def f():\n    return 1\n")[0])

    def test_bad_import_fails(self):
        ok, err = gates.import_ok("import nonexistent_pkg_xyz\n")
        self.assertFalse(ok)
        self.assertTrue(err)


if __name__ == "__main__":
    unittest.main()
