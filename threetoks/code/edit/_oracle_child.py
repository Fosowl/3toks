"""The child-side test runner of the repo-test oracle.

oracle.run_tests does not import this module. It reads this file as text
and sends it to a sandboxed subprocess of the project interpreter, after
a prelude that sets ``_CONFIG`` (a dict built by oracle._tests_script).
The child runs the test files with pytest (when asked and installed) or
with unittest, then prints one line: the report marker plus a JSON report
``{"runner", "results", "errors"}``. Run directly, it uses SMOKE_CONFIG.
"""
import importlib
import importlib.util
import json
import sys
import unittest

PYTEST_RUNNER = "pytest"
UNITTEST_RUNNER = "unittest"
PYTEST_FLAGS = ["-q", "-p", "no:cacheprovider", "--junitxml"]
SMOKE_CONFIG = {"pytest": False, "files": [], "modules": [], "xml": "",
                "marker": "@@oracle-child-smoke@@"}


class Collector(unittest.TestResult):
    """A unittest result that keeps test id -> passed for every test."""

    def __init__(self):
        """Start with no outcomes; unittest.TestSuite.run fills them."""
        super().__init__()
        self.passed = {}

    def addSuccess(self, test):
        """Record a passed test (unittest calls it)."""
        self.passed[test.id()] = True

    def addFailure(self, test, err):
        """Record a failed assertion (unittest calls it)."""
        self.passed[test.id()] = False

    def addError(self, test, err):
        """Record a test that raised (unittest calls it)."""
        self.passed[test.id()] = False

    def addSubTest(self, test, subtest, err):
        """Mark the parent test failed when one subtest fails."""
        if err is not None:
            self.passed[test.id()] = False

    def addExpectedFailure(self, test, err):
        """An expected failure counts as passed."""
        self.passed[test.id()] = True

    def addUnexpectedSuccess(self, test):
        """An unexpected success counts as failed."""
        self.passed[test.id()] = False


def import_file(name, path):
    """Import one test file by its path under a dotted name.

    load_suite uses it, because an installed ``tests`` package can hide a
    corpus ``tests/`` folder from a normal import.

    Args:
        name: str, the dotted module name (test ids start with it).
        path: str, the file path relative to the cwd (corpus root).

    Returns:
        module, the executed module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_suite(modules, errors):
    """Load the unittest tests of every test file.

    Args:
        modules: list of [name, path] pairs from the config.
        errors: list[str], gets one line per file that failed to import.

    Returns:
        unittest.TestSuite, the tests of every importable file.
    """
    suite, loader = unittest.TestSuite(), unittest.TestLoader()
    for name, path in modules:
        try:
            suite.addTests(loader.loadTestsFromModule(import_file(name, path)))
        except (Exception, SystemExit) as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return suite


def run_unittest(modules):
    """Run the test files with unittest.

    Args:
        modules: list of [name, path] pairs from the config.

    Returns:
        dict, the report with test id -> passed and import errors.
    """
    errors, collector = [], Collector()
    load_suite(modules, errors).run(collector)
    return {"runner": UNITTEST_RUNNER, "results": collector.passed,
            "errors": errors}


def run_pytest(files, xml_path):
    """Run the test files with pytest; the parent reads the junit XML.

    Args:
        files: list[str], test file paths relative to the cwd.
        xml_path: str, where pytest writes its junit report.

    Returns:
        dict, a report that names the runner and holds no results.
    """
    import pytest
    pytest.main(files + PYTEST_FLAGS + [xml_path])
    return {"runner": PYTEST_RUNNER, "results": {}, "errors": []}


def main(config):
    """Run the tests and print the report line on the real stdout.

    Test output goes to stderr, so the report line stays easy to find.

    Args:
        config: dict, from oracle._tests_script (or SMOKE_CONFIG).

    Returns:
        dict, the printed report.
    """
    stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        if config["pytest"] and importlib.util.find_spec(PYTEST_RUNNER):
            report = run_pytest(config["files"], config["xml"])
        else:
            report = run_unittest(config["modules"])
    finally:
        sys.stdout = stdout
    print("\n" + config["marker"] + json.dumps(report))
    return report


if __name__ == "__main__":
    main(globals().get("_CONFIG", SMOKE_CONFIG))
