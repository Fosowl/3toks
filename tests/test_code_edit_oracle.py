"""Offline tests for the repo-test oracle of the edit vertical.

Real subprocesses on temp corpora; no model, no network. The unittest
runner path is forced with ``oracle.PREFER_PYTEST = False`` so the tests
behave the same whether or not the interpreter has pytest.
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from threetoks.code.edit import oracle
from threetoks.code.edit.index import SymbolIndex

MOD_BUGGY = "def add(a, b):\n    return a - b\n"
MOD_FIXED = "def add(a, b):\n    return a + b\n"
TEST_ADD = ("import unittest\nfrom mod import add\n\n"
            "class AddTest(unittest.TestCase):\n"
            "    def test_two(self):\n        self.assertEqual(add(2, 2), 4)\n"
            "    def test_zero(self):\n        self.assertEqual(add(0, 0), 0)\n")
TWO_ID = "tests.test_add.AddTest.test_two"
ZERO_ID = "tests.test_add.AddTest.test_zero"


def _corpus(tree: dict[str, str]) -> tuple[tempfile.TemporaryDirectory, Path]:
    holder = tempfile.TemporaryDirectory()
    root = Path(holder.name)
    for rel, source in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return holder, root


class ImportCheckTest(unittest.TestCase):
    def setUp(self):
        self._holder, self.root = _corpus({
            "mod.py": MOD_FIXED,
            "bad.py": "raise ValueError('boom at import')\n",
            "pkg/__init__.py": "VALUE = 1\n",
            "my-mod.py": MOD_FIXED,
        })
        self.addCleanup(self._holder.cleanup)

    def test_importable_module_passes(self):
        self.assertEqual(oracle.import_check(self.root, "mod.py"), (True, ""))

    def test_package_init_imports_as_the_package(self):
        self.assertEqual(oracle.import_check(self.root, "pkg/__init__.py"),
                         (True, ""))

    def test_module_raising_at_import_fails_with_its_error(self):
        ok, err = oracle.import_check(self.root, "bad.py")
        self.assertFalse(ok)
        self.assertIn("boom at import", err)

    def test_hyphenated_path_is_not_importable(self):
        self.assertEqual(oracle.import_check(self.root, "my-mod.py"),
                         (False, "not importable as a module: my-mod.py"))


class FindTestsTest(unittest.TestCase):
    def setUp(self):
        self._holder, self.root = _corpus({
            "mod.py": MOD_FIXED + "\ndef adder():\n    return 1\n",
            "app.py": "from mod import add\n\ndef main():\n    return add(1, 2)\n",
            "tests/test_add.py": TEST_ADD,
            "tests/test_other.py": ("from mod import adder\n\n"
                                    "def test_adder():\n    assert adder() == 1\n"),
            "tests/helpers.py": ("from mod import add\n\n"
                                 "def make():\n    return add\n"),
            "pkg/add_test.py": ("import mod\n\n"
                                "def test_ref():\n    assert mod.add\n"),
        })
        self.addCleanup(self._holder.cleanup)
        self.index = SymbolIndex(self.root)
        self.target = self.index.find_by_name("add")[0]

    def test_keeps_test_files_referencing_the_name_by_call_or_text(self):
        self.assertEqual(oracle.find_tests(self.index, self.target),
                         ["pkg/add_test.py", "tests/helpers.py",
                          "tests/test_add.py"])

    def test_ignores_unrelated_tests_and_non_test_callers(self):
        found = oracle.find_tests(self.index, self.target)
        self.assertNotIn("tests/test_other.py", found)
        self.assertNotIn("app.py", found)

    def test_caps_the_number_of_files(self):
        for i in range(oracle.MAX_TEST_FILES + 2):
            (self.root / "tests" / f"test_more{i}.py").write_text(
                f"from mod import add\n\ndef test_{i}():\n    assert add(1, 1)\n")
        found = oracle.find_tests(SymbolIndex(self.root), self.target)
        self.assertEqual(len(found), oracle.MAX_TEST_FILES)
        self.assertEqual(found, sorted(found))


class RunTestsUnittestTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(oracle, "PREFER_PYTEST", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._holder, self.root = _corpus({"mod.py": MOD_BUGGY,
                                           "tests/test_add.py": TEST_ADD})
        self.addCleanup(self._holder.cleanup)
        self.files = ["tests/test_add.py"]

    def test_reports_each_test_id_with_its_outcome(self):
        run = oracle.run_tests(self.root, self.files)
        self.assertEqual(run, oracle.TestRun({TWO_ID: False, ZERO_ID: True}, ""))

    def test_fix_is_reported_as_fixed(self):
        before = oracle.run_tests(self.root, self.files)
        (self.root / "mod.py").write_text(MOD_FIXED)
        verdict = oracle.compare(before, oracle.run_tests(self.root, self.files))
        self.assertEqual(verdict, oracle.Verdict(True, [TWO_ID], []))

    def test_regression_is_reported_as_broken(self):
        (self.root / "mod.py").write_text(MOD_FIXED)
        before = oracle.run_tests(self.root, self.files)
        (self.root / "mod.py").write_text("def add(a, b):\n    return 1\n")
        verdict = oracle.compare(before, oracle.run_tests(self.root, self.files))
        self.assertEqual(verdict, oracle.Verdict(False, [], [TWO_ID, ZERO_ID]))

    def test_edit_breaking_the_import_counts_as_broken_not_ok(self):
        (self.root / "mod.py").write_text(MOD_FIXED)
        before = oracle.run_tests(self.root, self.files)
        (self.root / "mod.py").write_text("def add(a, b:\n")
        after = oracle.run_tests(self.root, self.files)
        self.assertEqual(after.results, {})
        self.assertNotEqual(after.error, "")
        verdict = oracle.compare(before, after)
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.broken, [TWO_ID, ZERO_ID])

    def test_no_bytecode_is_written(self):
        oracle.run_tests(self.root, self.files)
        self.assertEqual(list(self.root.rglob("__pycache__")), [])


class CollectionErrorTest(unittest.TestCase):
    def setUp(self):
        self._holder, self.root = _corpus({
            "mod.py": MOD_FIXED,
            "tests/test_broken.py": "from mod import missing_name\n",
        })
        self.addCleanup(self._holder.cleanup)

    def _assert_error_without_results(self):
        run = oracle.run_tests(self.root, ["tests/test_broken.py"])
        self.assertEqual(run.results, {})
        self.assertIn("missing_name", run.error)

    def test_unittest_runner_reports_the_import_error(self):
        with mock.patch.object(oracle, "PREFER_PYTEST", False):
            self._assert_error_without_results()

    def test_default_runner_reports_the_collection_error(self):
        self._assert_error_without_results()

    def test_no_files_is_an_error_not_a_crash(self):
        self.assertEqual(oracle.run_tests(self.root, []),
                         oracle.TestRun({}, "no test files"))


class CompareTest(unittest.TestCase):
    def test_error_on_either_side_is_not_ok(self):
        good = oracle.TestRun({"t": True}, "")
        failed = oracle.TestRun({}, "collection error")
        self.assertFalse(oracle.compare(failed, good).ok)
        self.assertFalse(oracle.compare(good, failed).ok)

    def test_same_results_are_ok_with_nothing_fixed(self):
        run = oracle.TestRun({"t": True, "u": False}, "")
        self.assertEqual(oracle.compare(run, run), oracle.Verdict(True, [], []))


class RunTestTest(unittest.TestCase):
    def test_fails_before_passes_after_and_writes_no_bytecode(self):
        holder, root = _corpus({"mod.py": MOD_BUGGY})
        self.addCleanup(holder.cleanup)
        test = "from mod import add\nassert add(2, 2) == 4\n"
        ok, err = oracle.run_test(root, test)
        self.assertFalse(ok)
        self.assertIn("AssertionError", err)
        (root / "mod.py").write_text(MOD_FIXED)
        self.assertEqual(oracle.run_test(root, test), (True, ""))
        self.assertEqual(list(root.rglob("__pycache__")), [])


if __name__ == "__main__":
    unittest.main()
