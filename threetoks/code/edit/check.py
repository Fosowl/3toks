"""Check one edit candidate without ever leaving it on disk.

The edit vertical delegates every check to one CandidateChecker. The
checker writes the candidate, runs the best available oracle, and
ALWAYS restores the original bytes (KeyboardInterrupt included). Oracle
order: the caller's test, else the repo tests that reference the target
(baseline before the edit, compare after), else "the module imports".
Zero model calls.
"""
from dataclasses import dataclass, field
from pathlib import Path

from threetoks.code.edit import oracle
from threetoks.code.edit.encoding import encode_like

ERROR_CLIP = 80


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one candidate check."""
    passed: bool
    verified: bool        # a test proved the fix
    error: str            # "" when passed
    tests: dict = field(default_factory=dict)   # repo-test summary or {}


class CandidateChecker:
    """Runs the free checks of one edit episode.

    Public use: ``caller_test_passes()`` before any model call,
    ``prepare(target)`` once per accepted target, ``check(target,
    candidate)`` per candidate, ``success_reason(result)`` on success.
    """

    def __init__(self, index, test_code: str | None, log):
        """Keep the corpus, the caller test, and the log line writer.

        The edit vertical creates one checker per episode.

        Args:
            index: SymbolIndex, the corpus index (gives root and files).
            test_code: str | None, trusted caller test source, or None.
            log: callable(str), writes one line to the menu session log.
        """
        self.root = Path(index.root)
        self.index = index
        self.test_code = test_code
        self._log = log
        self._baselines: dict = {}      # Symbol -> (files, TestRun) | None
        self._importable: set[str] = set()

    def caller_test_passes(self) -> bool:
        """Tell if the caller's test already passes before any edit.

        The edit vertical calls this once, before any model call: a
        passing test means there is nothing to fix.

        Returns:
            bool: True only when a caller test exists and passes.
        """
        if self.test_code is None:
            return False
        passed, _ = oracle.run_test(self.root, self.test_code)
        return passed

    def prepare(self, target) -> str:
        """Prepare the check of one target before any generation.

        The edit vertical calls this when it accepts a target. Repo tests
        are found and run once (baseline); without them, the module must
        import before the edit.

        Args:
            target: Symbol, the accepted edit target.
        Returns:
            str: "" when a check is available, else why the episode ends.
        """
        if self.test_code is not None:
            return ""
        if target not in self._baselines:
            self._baselines[target] = self._repo_baseline(target)
        if self._baselines[target] is not None:
            return ""
        return self._import_precheck(target.file)

    def _repo_baseline(self, target):
        """Find repo tests for the target and run them before the edit.

        Only prepare calls this.

        Args:
            target: Symbol, the accepted edit target.
        Returns:
            tuple[list[str], oracle.TestRun] | None: the files and their
            baseline run, or None when no usable test exists.
        """
        files = oracle.find_tests(self.index, target)
        if not files:
            return None
        before = oracle.run_tests(self.root, files)
        if before.error or not before.results:
            self._log(f"> repo tests unusable: {before.error[:ERROR_CLIP]}")
            return None
        self._log(f"> repo tests: {', '.join(files)}")
        return files, before

    def _import_precheck(self, rel: str) -> str:
        """Check once per file that the target module imports.

        Only prepare calls this.

        Args:
            rel: str, the target file path relative to the root.
        Returns:
            str: "" when it imports, else the finish reason.
        """
        if rel in self._importable:
            return ""
        ok, err = oracle.import_check(self.root, rel)
        if not ok:
            return f"{rel} does not import before any edit: {err[:ERROR_CLIP]}"
        self._importable.add(rel)
        return ""

    def check(self, target, candidate: str) -> CheckResult:
        """Write the candidate, run the check, ALWAYS restore the bytes.

        The edit vertical calls this for every candidate. The finally
        block restores the original bytes on any exit, KeyboardInterrupt
        included. A file that cannot be read, encoded, or written gives a
        failed check with the reason, not an exception.

        Args:
            target: Symbol, the accepted edit target.
            candidate: str, the full new file text (LF).
        Returns:
            CheckResult: the verdict; the file is unchanged afterwards.
        """
        path = self.root / target.file
        try:
            original = path.read_bytes()
            encoded = encode_like(original, candidate)
        except (OSError, UnicodeEncodeError) as error:
            return CheckResult(False, False, _io_reason(error))
        try:
            path.write_bytes(encoded)
        except OSError as error:
            _restore_if_changed(path, original)
            return CheckResult(False, False, _io_reason(error))
        try:
            return self._run_checks(target)
        finally:
            path.write_bytes(original)

    def _run_checks(self, target) -> CheckResult:
        """Run the best available check against the files on disk.

        Only check calls this, while the candidate is on disk.

        Args:
            target: Symbol, the accepted edit target.
        Returns:
            CheckResult: the verdict.
        """
        if self.test_code is not None:
            passed, err = oracle.run_test(self.root, self.test_code)
            return CheckResult(passed, passed, err)
        baseline = self._baselines.get(target)
        if baseline is not None:
            return self._compare_repo_tests(*baseline)
        passed, err = oracle.import_check(self.root, target.file)
        return CheckResult(passed, False, err)

    def _compare_repo_tests(self, files: list[str], before) -> CheckResult:
        """Run the repo tests again and compare with the baseline.

        Only _run_checks calls this.

        Args:
            files: list[str], test file paths from oracle.find_tests.
            before: oracle.TestRun, the baseline run.
        Returns:
            CheckResult: passed when nothing broke; verified when a test
            was fixed.
        """
        after = oracle.run_tests(self.root, files)
        verdict = oracle.compare(before, after)
        tests = {"files": list(files), "fixed": list(verdict.fixed),
                 "broken": list(verdict.broken)}
        if verdict.ok:
            return CheckResult(True, bool(verdict.fixed), "", tests)
        error = after.error or "breaks " + ", ".join(verdict.broken)
        return CheckResult(False, False, error, tests)

    def success_reason(self, result: CheckResult) -> str:
        """Say in one line which check the candidate passed.

        The edit vertical calls this with a passed result.

        Args:
            result: CheckResult, from check.
        Returns:
            str: the finish reason.
        """
        if self.test_code is not None:
            return "test passes after edit"
        if not result.tests:
            return "module imports after the edit (no test given)"
        if result.verified:
            count = len(result.tests["fixed"])
            return f"repo tests pass; the edit fixed {count} failing test(s)"
        return "repo tests still pass; none fixed by the edit"


def _io_reason(error: Exception) -> str:
    """Say in one line why a candidate could not be put on disk.

    Only CandidateChecker.check calls this.

    Args:
        error: OSError or UnicodeEncodeError from the read, encode, or write.
    Returns:
        str: the reason for the repair log and the result.
    """
    if isinstance(error, UnicodeEncodeError):
        return (f"the new code cannot be saved in the file's encoding "
                f"({error.encoding})")
    return f"cannot read or write the target file: {error.strerror or error}"


def _restore_if_changed(path: Path, original: bytes) -> None:
    """Put the original bytes back after a failed write, if needed.

    Only CandidateChecker.check calls this. A write that failed on open
    (read-only file) changed nothing; a write that failed midway did.

    Args:
        path: Path, the target file.
        original: bytes, the file content before the write.
    """
    try:
        if path.read_bytes() == original:
            return
    except OSError:
        pass
    path.write_bytes(original)


if __name__ == "__main__":
    import tempfile

    from threetoks.code.edit.index import SymbolIndex

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = b"def add(a, b):\r\n    return a - b\r\n"
        (root / "mod.py").write_bytes(original)
        index = SymbolIndex(root)
        target = index.symbol_at("mod.py", "add")
        lines = []
        checker = CandidateChecker(
            index, "from mod import add\nassert add(2, 3) == 5\n",
            lines.append)
        assert not checker.caller_test_passes()
        assert checker.prepare(target) == ""
        good = checker.check(target, "def add(a, b):\n    return a + b\n")
        assert good.passed and good.verified, good
        assert checker.success_reason(good) == "test passes after edit"
        bad = checker.check(target, "def add(a, b):\n    return a * b\n")
        assert not bad.passed and "AssertionError" in bad.error, bad
        assert (root / "mod.py").read_bytes() == original

        plain = CandidateChecker(index, None, lines.append)
        assert plain.prepare(target) == ""          # module imports
        imported = plain.check(target, "def add(a, b):\n    return a + b\n")
        assert imported.passed and not imported.verified
        assert (root / "mod.py").read_bytes() == original
    print("smoke OK")
