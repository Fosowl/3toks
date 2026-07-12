"""The trusted-oracle gate: a caller-supplied test, run before and after.

Zero model calls, same trust model as gates.execute — a fresh python3
subprocess per run, so it always reflects what is currently on disk. The
vertical runs the test once *before* touching the target file (it must
FAIL — that is what "there is a bug" means) and once after every candidate
edit (it must PASS — the only signal an edit is trusted as done).
"""
from pathlib import Path

from threetoks.code import gates

# The oracle rewrites the same file before and after an edit, often within
# one wall-clock second; a stale __pycache__ .pyc keyed on mtime+size can
# then silently re-run pre-edit bytecode. The prelude therefore disables
# bytecode writing in every oracle subprocess. Pre-existing .pyc files in a
# user corpus remain a (documented) residual risk.
_PRELUDE = "import sys\nsys.dont_write_bytecode = True\n"


def run_test(root: Path, test_code: str) -> tuple[bool, str]:
    """Run test_code in a subprocess with ``root`` on sys.path.

    Returns (passed, first_error_line). test_code is trusted harness or
    caller source (never model output).
    """
    prelude = _PRELUDE + f"sys.path.insert(0, {str(root)!r})\n"
    ok, stderr = gates.execute(prelude + test_code)
    return ok, ("" if ok else gates.first_error(stderr))


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "mod.py").write_text("def add(a, b):\n    return a - b\n")
        test = "from mod import add\nassert add(2, 2) == 4, add(2, 2)\n"
        before_ok, before_err = run_test(root, test)
        assert not before_ok and "AssertionError" in before_err

        (root / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        after_ok, after_err = run_test(root, test)
        assert after_ok and after_err == ""
        assert not list(root.rglob("__pycache__")), "bytecode must not be written"
    print("smoke OK")
