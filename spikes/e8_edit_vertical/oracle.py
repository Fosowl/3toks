"""The trusted-oracle gate: a scenario's own test, run before and after.

Zero model calls, same trust model as threetoks/code/gates.execute -- a
fresh python3 subprocess per run, so it always reflects whatever is
currently on disk. The harness runs this test once *before* touching the
target file (it must FAIL -- that is what "there is a bug" means) and once
*after* a candidate edit is written (it must PASS -- that is the only
signal the vertical trusts to call an edit done).
"""
import os
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_DIR.parent.parent
for _p in (str(SPIKE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from threetoks.code import gates  # noqa: E402

# The oracle rewrites the SAME file path before and after an edit, often
# within the same wall-clock second; a stale __pycache__ .pyc keyed on
# mtime would then make the "after" run silently re-execute the pre-edit
# bytecode. Bytecode caching buys nothing for one-shot subprocess checks,
# so it is disabled for every oracle subprocess.
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def run_test(root: Path, test_code: str) -> tuple[bool, str]:
    """Run test_code as a subprocess with `root` on sys.path.

    Returns (passed, first_error_line). test_code is trusted harness/
    scenario source (never model output), so it is fine to prepend a raw
    sys.path.insert and exec it as-is.
    """
    prelude = f"import sys\nsys.path.insert(0, {str(root)!r})\n"
    ok, stderr = gates.execute(prelude + test_code)
    return ok, ("" if ok else gates.first_error(stderr))


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "mod.py").write_text("def add(a, b):\n    return a - b\n")
        test = ("from mod import add\n"
               "assert add(2, 2) == 4, add(2, 2)\n")
        before_ok, before_err = run_test(root, test)
        assert not before_ok and "AssertionError" in before_err, before_err

        (root / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        after_ok, after_err = run_test(root, test)
        assert after_ok and after_err == ""
    print("smoke OK")
