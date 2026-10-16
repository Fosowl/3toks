"""The trusted oracles of the edit vertical: tests run before and after.

Zero model calls. Every run is a fresh sandboxed subprocess of the
project's own interpreter, with the corpus root first on ``sys.path``, so
it always reflects what is currently on disk. Three oracles exist:

- ``run_test``: one caller-supplied test script (must fail before, pass after);
- ``import_check``: the edited module still imports;
- ``run_tests`` + ``compare``: the repo's own tests that reference the
  target, run before and after an edit — a free signal of fixed/broken tests.
"""
import json
import keyword
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from threetoks.code import gates, sandbox
from threetoks.code.edit import rank
from threetoks.code.edit._oracle_child import PYTEST_RUNNER

MAX_TEST_FILES = 5
TEST_TIMEOUT_S = 60
# Tests set this False to force the unittest path.
PREFER_PYTEST = True
TEST_DIR_NAMES = ("tests", "test")
TEST_FILE_PREFIX = "test_"
TEST_FILE_SUFFIX = "_test.py"
NOT_TEST_FILES = ("conftest.py", "__init__.py")
REPORT_MARKER = "@@threetoks-oracle-report@@"
JUNIT_FILE = "junit.xml"
COLLECTION_FAILURE = "collection failure"
IDENT_KEEP = "_"
CHILD_SCRIPT = Path(__file__).with_name("_oracle_child.py")

# The oracle rewrites the same file before and after an edit, often within
# one wall-clock second; a stale __pycache__ .pyc keyed on mtime+size can
# then silently re-run pre-edit bytecode. The prelude therefore disables
# bytecode writing in every oracle subprocess. Pre-existing .pyc files in a
# user corpus remain a (documented) residual risk.
_PRELUDE = "import sys\nsys.dont_write_bytecode = True\n"


@dataclass(frozen=True)
class TestRun:
    """The outcome of one run of the repo tests."""
    __test__ = False   # not a test class for pytest
    results: dict[str, bool]   # test id -> passed
    error: str                 # "" when results exist; else why none


@dataclass(frozen=True)
class Verdict:
    """The before/after comparison of two TestRun values."""
    ok: bool            # both runs had results, nothing broke
    fixed: list[str]    # failed before, pass after
    broken: list[str]   # passed before, fail after


def _prelude(root: Path) -> str:
    """Give the source that starts every oracle subprocess.

    The oracle functions use it so the child can import corpus modules
    and writes no bytecode. It returns the no-bytecode prelude plus a line
    that puts the corpus root first on sys.path.

    Args:
        root: Path, the corpus root from the caller.

    Returns:
        str, Python source.
    """
    return _PRELUDE + f"sys.path.insert(0, {str(root)!r})\n"


def _run(root: Path, source: str, timeout_s: float) -> sandbox.RunResult:
    """Run source in the sandbox with the project interpreter.

    Every oracle uses it, so all runs share the same cwd, interpreter and
    sys.path rules. It adds the prelude and calls sandbox.run_python.

    Args:
        root: Path, the corpus root (cwd and sys.path entry).
        source: str, trusted harness or caller Python source.
        timeout_s: float, the wall-clock limit in seconds.

    Returns:
        sandbox.RunResult, the result of the child.
    """
    return sandbox.run_python(_prelude(root) + source, cwd=root,
                              timeout_s=timeout_s,
                              interpreter=sandbox.project_python(root))


def _failure_line(result: sandbox.RunResult) -> str:
    """Give one short line that tells why a child run failed.

    run_test and import_check use it for their error text. It returns
    gates.TIMEOUT_MESSAGE on a timeout, else the first error line of stderr.

    Args:
        result: sandbox.RunResult, a failed child run.

    Returns:
        str, the error line.
    """
    if result.timed_out:
        return gates.TIMEOUT_MESSAGE
    return gates.first_error(result.stderr)


def run_test(root: Path, test_code: str) -> tuple[bool, str]:
    """Run one caller test script against the corpus on disk.

    The edit vertical calls it before an edit (the test must fail) and
    after each candidate (the test must pass). test_code is trusted harness
    or caller source, never model output.

    Args:
        root: Path, the corpus root.
        test_code: str, Python source that raises when the test fails.

    Returns:
        tuple[bool, str], (passed, first error line or "").
    """
    result = _run(root, test_code, sandbox.DEFAULT_TIMEOUT_S)
    return result.ok, ("" if result.ok else _failure_line(result))


def _module_name(rel_file: str) -> str | None:
    """Give the dotted module name of a corpus-relative .py path.

    import_check and run_tests use it to import files as modules. A
    package ``__init__.py`` maps to its package name.

    Args:
        rel_file: str, a posix path relative to the corpus root.

    Returns:
        str | None, the dotted name, or None when a part is not an identifier.
    """
    path = PurePosixPath(rel_file)
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or not all(p.isidentifier() and not keyword.iskeyword(p)
                            for p in parts):
        return None
    return ".".join(parts)


def import_check(root: Path, rel_file: str) -> tuple[bool, str]:
    """Check that one corpus file still imports as a module.

    The edit vertical calls it after a candidate edit when no caller test
    exists. A path that is not a dotted identifier path fails without a
    subprocess.

    Args:
        root: Path, the corpus root.
        rel_file: str, the edited file, relative to root.

    Returns:
        tuple[bool, str], (imports, first error line or "").
    """
    module = _module_name(rel_file)
    if module is None:
        return False, f"not importable as a module: {rel_file}"
    source = f"import importlib\nimportlib.import_module({module!r})\n"
    result = _run(root, source, sandbox.DEFAULT_TIMEOUT_S)
    return result.ok, ("" if result.ok else _failure_line(result))


def _is_test_file(rel_file: str) -> bool:
    """Tell if a corpus path is a test file by its name or folder.

    find_tests uses it to filter the indexed files.

    Args:
        rel_file: str, a posix path relative to the corpus root.

    Returns:
        bool, True for test_*.py, *_test.py, or a .py under tests/ or test/.
    """
    path = PurePosixPath(rel_file)
    if path.suffix != ".py" or path.name in NOT_TEST_FILES:
        return False
    if path.name.startswith(TEST_FILE_PREFIX) or path.name.endswith(TEST_FILE_SUFFIX):
        return True
    return any(part in TEST_DIR_NAMES for part in path.parts[:-1])


def _mentions(text: str, name: str) -> bool:
    """Tell if text contains name as a whole identifier.

    find_tests uses it for references that are not calls (imports,
    attribute reads, callbacks).

    Args:
        text: str, the source of a test file.
        name: str, the bare target name.

    Returns:
        bool, True when name is one of the identifier words of text.
    """
    return name in rank.words(text, keep=IDENT_KEEP)


def find_tests(index, target) -> list[str]:
    """Find the repo test files that reference the edit target.

    The edit vertical calls it to choose repo tests to run before and after
    an edit. Candidate files are every indexed file (``index.files``).

    Args:
        index: SymbolIndex, the corpus index.
        target: Symbol, the located target (its bare name is matched).

    Returns:
        list[str], at most MAX_TEST_FILES rel paths, sorted.
    """
    callers = {rel for rel, _ in index.call_sites.get(target.name, [])}
    found = [rel for rel in index.files if _is_test_file(rel)
             and (rel in callers or _mentions(index.source_of(rel), target.name))]
    return found[:MAX_TEST_FILES]


def _tests_script(test_files: list[str], xml_path: Path) -> str:
    """Build the child source that runs the test files.

    run_tests uses it. The source is CHILD_SCRIPT (``_oracle_child.py``)
    after one line that sets its ``_CONFIG``. The child runs pytest when
    PREFER_PYTEST is set and pytest imports; else it runs unittest.

    Args:
        test_files: list[str], rel paths from find_tests.
        xml_path: Path, where pytest writes its junit report.

    Returns:
        str, Python source (without the prelude).
    """
    modules = [[_module_name(rel) or rel, rel] for rel in test_files]
    config = {"pytest": PREFER_PYTEST, "files": test_files, "modules": modules,
              "xml": str(xml_path), "marker": REPORT_MARKER}
    prefix = f"import json\n_CONFIG = json.loads({json.dumps(config)!r})\n"
    return prefix + CHILD_SCRIPT.read_text(encoding="utf-8")


def _report(stdout: str) -> dict | None:
    """Read the JSON report line that the child printed last.

    Args:
        stdout: str, the child stdout.

    Returns:
        dict | None, the report, or None when it is absent or bad.
    """
    start = stdout.rfind(REPORT_MARKER)
    if start == -1:
        return None
    line = stdout[start + len(REPORT_MARKER):].split("\n", 1)[0]
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _case_outcome(case: ET.Element) -> tuple[str, bool | None]:
    """Classify one junit testcase element.

    Args:
        case: ET.Element, a <testcase> from the pytest junit report.

    Returns:
        tuple[str, bool | None], (collection error text or "", passed or
        None when skipped or not collected).
    """
    error = case.find("error")
    if error is not None and COLLECTION_FAILURE in error.get("message", ""):
        return f"{case.get('name')}: {gates.first_error(error.text or '')}", None
    if case.find("skipped") is not None:
        return "", None
    return "", error is None and case.find("failure") is None


def _junit(xml_path: Path) -> tuple[dict[str, bool], list[str]]:
    """Parse the pytest junit report into results and collection errors.

    Test ids use unittest style: "<classname>.<name>".

    Args:
        xml_path: Path, the report that pytest wrote.

    Returns:
        tuple[dict[str, bool], list[str]], (test id -> passed, errors).
    """
    try:
        tree = ET.parse(xml_path)
    except (OSError, ET.ParseError):
        return {}, ["pytest wrote no report"]
    results, errors = {}, []
    for case in tree.iter("testcase"):
        error, passed = _case_outcome(case)
        if error:
            errors.append(error)
        if passed is not None:
            name, classname = case.get("name", ""), case.get("classname", "")
            results[f"{classname}.{name}" if classname else name] = passed
    return results, errors


def _test_run(result: sandbox.RunResult, xml_path: Path) -> TestRun:
    """Turn the child result of the test script into a TestRun.

    Args:
        result: sandbox.RunResult, the test child run.
        xml_path: Path, the junit report path (pytest runner only).

    Returns:
        TestRun, results, or no results and the reason.
    """
    if result.timed_out:
        return TestRun({}, f"tests timed out after {TEST_TIMEOUT_S} s")
    report = _report(result.stdout)
    if report is None:
        return TestRun({}, "test runner failed: " + gates.first_error(result.stderr))
    if report["runner"] == PYTEST_RUNNER:
        results, errors = _junit(xml_path)
    else:
        results, errors = report["results"], report["errors"]
    if results:
        return TestRun(results, "")
    return TestRun({}, "; ".join(errors) or "no tests ran")


def run_tests(root: Path, test_files: list[str]) -> TestRun:
    """Run repo test files against the corpus on disk.

    The edit vertical calls it once before an edit and once after a
    candidate edit, then gives both runs to compare.

    Args:
        root: Path, the corpus root.
        test_files: list[str], rel paths from find_tests.

    Returns:
        TestRun, test id -> passed, or no results and an error.
    """
    if not test_files:
        return TestRun({}, "no test files")
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / JUNIT_FILE
        result = _run(root, _tests_script(test_files, xml_path), TEST_TIMEOUT_S)
        return _test_run(result, xml_path)


def compare(before: TestRun, after: TestRun) -> Verdict:
    """Compare repo test runs from before and after an edit.

    The edit vertical uses it to mark an edit verified (a test is fixed)
    or rejected (a test broke). A test that passed before and is missing
    after counts as broken.

    Args:
        before: TestRun, the run before the edit.
        after: TestRun, the run with the candidate on disk.

    Returns:
        Verdict, ok is False when a run has an error or no results.
    """
    fixed = sorted(t for t, passed in after.results.items()
                   if passed and before.results.get(t) is False)
    broken = sorted(t for t, passed in before.results.items()
                    if passed and not after.results.get(t, False))
    usable = bool(before.results and after.results
                  and not before.error and not after.error)
    return Verdict(usable and not broken, fixed, broken)


if __name__ == "__main__":
    from threetoks.code.edit.index import SymbolIndex
    from threetoks.code.edit._oracle_child import SMOKE_CONFIG

    alone = sandbox.run_python(CHILD_SCRIPT.read_text(encoding="utf-8"))
    assert alone.ok, alone.stderr
    assert SMOKE_CONFIG["marker"] + '{"runner": "unittest"' in alone.stdout, alone

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "mod.py").write_text("def add(a, b):\n    return a - b\n")
        test = "from mod import add\nassert add(2, 2) == 4, add(2, 2)\n"
        before_ok, before_err = run_test(root, test)
        assert not before_ok and "AssertionError" in before_err
        assert import_check(root, "mod.py") == (True, "")
        assert not import_check(root, "my-mod.py")[0]

        (root / "tests").mkdir()
        (root / "tests" / "test_mod.py").write_text(
            "import unittest\nfrom mod import add\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 2), 4)\n")
        target = SymbolIndex(root).find_by_name("add")[0]
        files = find_tests(SymbolIndex(root), target)
        assert files == ["tests/test_mod.py"], files
        before = run_tests(root, files)

        (root / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        assert run_test(root, test) == (True, "")
        verdict = compare(before, run_tests(root, files))
        assert verdict.ok and verdict.fixed == ["tests.test_mod.T.test_add"], verdict
        assert not list(root.rglob("__pycache__")), "bytecode must not be written"
    print("smoke OK")
