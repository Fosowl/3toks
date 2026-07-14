"""Execution judging: the subprocess IS the judge, never the model.

Reuses ``threetoks/code/gates.py`` verbatim for the actual subprocess
execution (task instructions: "reuse or adapt threetoks/code/gates.py") —
its ``execute_asserts`` already runs ``python3 -`` with the module source
piped over stdin, a 5s timeout, and no interactive stdin available (the
pipe is closed after writing, so an accidental ``input()`` call gets an
immediate EOFError rather than hanging).

The only thing added here is wall-clock timing per candidate, since
gates.py has no reason to care about that for the greenfield vertical.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from threetoks.code.gates import run_asserts  # noqa: E402


def judge(snippet: str, anchor_assert: str) -> tuple:
    """Run ``anchor_assert`` against ``snippet`` in a fresh subprocess.

    Returns (passed: bool, error: str, elapsed_seconds: float). ``error``
    is the harness's first-error-line extraction (empty string on pass).
    """
    start = time.perf_counter()
    ok, err = run_asserts(snippet, [anchor_assert])
    elapsed = time.perf_counter() - start
    return ok, err, elapsed


if __name__ == "__main__":
    good = "def add(a, b):\n    return a + b\n"
    ok, err, elapsed = judge(good, "assert add(2, 3) == 5")
    assert ok and err == "" and elapsed >= 0, (ok, err, elapsed)

    wrong = "def add(a, b):\n    return a - b\n"
    ok2, err2, _ = judge(wrong, "assert add(2, 3) == 5")
    assert not ok2 and "AssertionError" in err2, (ok2, err2)

    hang_free = "def f():\n    return input()\n"
    ok3, err3, elapsed3 = judge(hang_free, "assert f() == 'x'")
    assert not ok3 and elapsed3 < 5.0, (ok3, err3, elapsed3)  # EOF, not a hang
    print("smoke OK")
