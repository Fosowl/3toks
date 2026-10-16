"""Offline tests for the module runner (subprocess only, no model).

Covers check collection (examples for every bodied method, a smoke call
ONLY for the script entry, stubs excluded), the echo entry call,
traceback-to-method blame mapping (deepest frame wins, assert failures
default to the checked method), the run working directory, and the
inconclusive-skip rule for interactive ``input()`` code.
"""
import tempfile
import unittest
from pathlib import Path

from threetoks.code.runner import (Check, KIND_CALL, KIND_EXAMPLE, blame,
                                  collect_checks, entry_call, failures,
                                  is_inconclusive, method_spans, run_checks)
from threetoks.code.store import MethodStore

MODULE = ('"demo"\n'
          "\n\n"
          "def boom(n):\n"
          "    return n // 0\n"
          "\n\n"
          "def entry():\n"
          "    return boom(4)\n"
          "\n\n"
          "def add(a, b):\n"
          "    return a + b\n")


def _store(*methods):
    """A MethodStore with (name, args, body, example) records."""
    store = MethodStore("g")
    for name, args, body, example in methods:
        record = store.add(name, args, "c", example)
        record.body = body
    return store


class CollectChecksTest(unittest.TestCase):
    def _store(self):
        return _store(("a", "", "def a(): return 1", "a() == 1"),
                      ("b", "", "def b(): return 2", None),
                      ("c", "x", "def c(x): return x", None),
                      ("stub", "", None, None))

    def test_only_the_entry_is_smoke_called(self):
        checks = collect_checks(self._store().methods, "b")
        self.assertEqual([(c.name, c.kind) for c in checks],
                         [("a", KIND_EXAMPLE), ("b", KIND_CALL)])
        self.assertEqual(checks[0].statement, "assert a() == 1")
        self.assertEqual(checks[1].statement, "b()")

    def test_no_entry_means_no_smoke_call_at_all(self):
        checks = collect_checks(self._store().methods, None)
        self.assertEqual([c.kind for c in checks], [KIND_EXAMPLE])

    def test_stub_entry_is_never_called(self):
        checks = collect_checks(self._store().methods, "stub")
        self.assertNotIn(KIND_CALL, [c.kind for c in checks])


class EntryCallTest(unittest.TestCase):
    def test_echo_prints_a_returned_value(self):
        checks = collect_checks(
            _store(("answer", "", "def answer():\n    return 42", None)
                   ).methods, "answer", echo=True)
        [result] = run_checks("def answer():\n    return 42\n", checks)
        self.assertTrue(result.passed)
        self.assertEqual(result.stdout, "42\n")

    def test_echo_stays_silent_for_none(self):
        [result] = run_checks("def quiet():\n    return None\n",
                              [Check("quiet", entry_call("quiet", True),
                                     KIND_CALL)])
        self.assertEqual(result.stdout, "")

    def test_without_echo_a_return_value_is_not_printed(self):
        [result] = run_checks("def answer():\n    return 42\n",
                              [Check("answer", entry_call("answer", False),
                                     KIND_CALL)])
        self.assertEqual(result.stdout, "")

    def test_indented_call_parses_as_a_guard_body(self):
        guard = 'if __name__ == "__main__":\n' + entry_call("f", True, "    ")
        compile("def f():\n    return 1\n" + guard, "<t>", "exec")


class RunCwdTest(unittest.TestCase):
    def test_checks_run_in_the_given_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "data.txt").write_text("from disk")
            module = "def main():\n    print(open('data.txt').read())\n"
            [result] = run_checks(module, [Check("main", "main()",
                                                 KIND_CALL)], Path(folder))
        self.assertTrue(result.passed, result.error)
        self.assertEqual(result.stdout, "from disk\n")


class SpansAndBlameTest(unittest.TestCase):
    def test_spans_cover_every_top_level_def(self):
        spans = method_spans(MODULE)
        self.assertEqual(set(spans), {"boom", "entry", "add"})
        for start, end in spans.values():
            self.assertLessEqual(start, end)

    def test_unparsable_module_yields_no_spans(self):
        self.assertEqual(method_spans("def broken(:"), {})

    def test_no_stdin_frames_blames_the_default(self):
        self.assertEqual(blame("no traceback here", {"a": (1, 2)}, "d"), "d")


class RunChecksTest(unittest.TestCase):
    def test_crash_inside_a_callee_blames_the_callee(self):
        [result] = run_checks(MODULE, [Check("entry", "entry()", KIND_CALL)])
        self.assertFalse(result.passed)
        self.assertEqual(result.blamed, "boom")
        self.assertIn("ZeroDivisionError", result.error)

    def test_runtime_stub_call_blames_the_stub(self):
        module = (MODULE + "\n\ndef ghost():\n    raise NotImplementedError\n"
                  "\n\ndef caller():\n    return ghost()\n")
        [result] = run_checks(module,
                              [Check("caller", "caller()", KIND_CALL)])
        self.assertEqual(result.blamed, "ghost")

    def test_plain_assert_failure_blames_the_checked_method(self):
        [result] = run_checks("def add(a, b):\n    return a - b\n",
                              [Check("add", "assert add(1, 2) == 3",
                                     KIND_EXAMPLE)])
        self.assertFalse(result.passed)
        self.assertEqual(result.blamed, "add")

    def test_passing_check_reports_clean(self):
        [result] = run_checks(MODULE, [Check("add", "assert add(1, 2) == 3",
                                             KIND_EXAMPLE)])
        self.assertTrue(result.passed)
        self.assertEqual(result.error, "")

    def test_input_eof_is_skipped_not_failed(self):
        module = 'def ask():\n    return input("? ")\n'
        [result] = run_checks(module, [Check("ask", "ask()", KIND_CALL)])
        self.assertTrue(result.skipped)
        self.assertFalse(result.passed)

    def test_eoferror_in_an_error_message_is_still_a_real_failure(self):
        # The skip rule must key on the raised exception TYPE, not on the
        # substring appearing anywhere in stderr — otherwise this crashing
        # module would ship with a clean run report.
        module = ("def main():\n"
                  "    raise ValueError('parser hit EOFError token')\n")
        [result] = run_checks(module, [Check("main", "main()", KIND_CALL)])
        self.assertFalse(result.skipped)
        self.assertEqual(result.blamed, "main")
        self.assertIn("ValueError", result.error)

    def test_is_inconclusive_reads_only_the_final_exception_line(self):
        self.assertTrue(is_inconclusive("Traceback ...\nEOFError\n"))
        self.assertTrue(is_inconclusive("Traceback ...\nEOFError: EOF hit\n"))
        self.assertTrue(is_inconclusive("KeyboardInterrupt\n"))
        self.assertFalse(is_inconclusive(
            "Traceback ...\nValueError: mentions EOFError\n"))
        self.assertFalse(is_inconclusive(""))

    def test_failures_excludes_passes_and_skips(self):
        checks = [Check("entry", "entry()", KIND_CALL),
                  Check("add", "assert add(1, 2) == 3", KIND_EXAMPLE)]
        results = run_checks(MODULE, checks)
        self.assertEqual([r.check.name for r in failures(results)],
                         ["entry"])


if __name__ == "__main__":
    unittest.main()
