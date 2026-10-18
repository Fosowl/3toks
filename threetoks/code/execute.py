"""Run a delivered script or a shell command for real, and report it.

The code agent uses this after it writes a script (author mode) and in
shell mode. Both run in the project folder (``services.files_root``):
the script under the project's python, the command under the user's
shell with the user's full environment. This is NOT a sandbox: the
script and the command can change any file the user can change.
"""
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from threetoks.code import sandbox

SCRIPT_TIMEOUT_S = 60       # a delivered script; plots and downloads take time
COMMAND_TIMEOUT_S = 120     # a shell command; pip installs take time
OUTPUT_TAIL_LINES = 30      # lines of each stream kept in the report
TIMEOUT_NOTE = "(stopped: it ran longer than {seconds}s)"


@dataclass(frozen=True)
class Execution:
    """The outcome of one real run, ready to report."""
    label: str              # what ran, e.g. "$ pip install numpy"
    cwd: Path               # the folder it ran in
    ok: bool                # exit code 0 and not timed out
    exit_code: int | None   # None when it did not start or timed out
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False


def run_script(source: str, root: Path) -> Execution:
    """Run a delivered Python script once in the project folder.

    The author mode uses this so that the code it writes really runs.
    The code agent calls it with the full script text and the files root.
    It runs the script on stdin under the project's python
    (``sandbox.project_python``), with SCRIPT_TIMEOUT_S as the limit.

    Args:
        source (str): the script text with its ``__main__`` guard.
        root (Path): the project folder, from services.files_root.

    Returns:
        Execution: the run outcome; the label names the interpreter.
    """
    interpreter = sandbox.project_python(Path(root))
    started = time.monotonic()
    run = sandbox.run_python(source, cwd=Path(root),
                             timeout_s=SCRIPT_TIMEOUT_S,
                             interpreter=interpreter)
    exit_code = None if run.timed_out else run.exit_code
    return Execution(f"python script ({interpreter})", Path(root), run.ok,
                     exit_code, run.stdout, run.stderr,
                     time.monotonic() - started, run.timed_out)


def run_command(command: str, root: Path) -> Execution:
    """Run one shell command in the project folder.

    The shell mode uses this after the user said yes to the command.
    The code agent calls it with the command and the files root.
    It runs the command with the user's shell and environment, with no
    input, and stops it after COMMAND_TIMEOUT_S.

    Args:
        command (str): the shell command line, written by the model.
        root (Path): the project folder, from services.files_root.

    Returns:
        Execution: the run outcome; the label is "$ command".
    """
    started = time.monotonic()
    try:
        done = subprocess.run(command, shell=True, cwd=Path(root),
                              stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, errors="replace",
                              timeout=COMMAND_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as stopped:
        return Execution(f"$ {command}", Path(root), False, None,
                         _text(stopped.stdout), _text(stopped.stderr),
                         time.monotonic() - started, timed_out=True)
    except OSError as error:
        return Execution(f"$ {command}", Path(root), False, None, "",
                         str(error), time.monotonic() - started)
    return Execution(f"$ {command}", Path(root), done.returncode == 0,
                     done.returncode, done.stdout, done.stderr,
                     time.monotonic() - started)


def _text(stream) -> str:
    """Give the partial output of a stopped command as text."""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream or ""


def status(execution: Execution) -> str:
    """Name the outcome of a run in a few words.

    The code agent puts this on the end row of the run branch in the
    progress tree. It receives one Execution.

    Args:
        execution (Execution): the finished run.

    Returns:
        str: "timed out", "exit N" or "could not start".
    """
    if execution.timed_out:
        return "timed out"
    if execution.exit_code is None:
        return "could not start"
    return f"exit {execution.exit_code}"


def report(execution: Execution) -> str:
    """Write the run report that goes under the answer.

    The code agent adds this to the author and shell answers, so the user
    sees what ran, where, how it ended, and what it printed.

    Args:
        execution (Execution): the finished run.

    Returns:
        str: a header line, the stdout tail, then the stderr tail.
    """
    header = (f"▶ {execution.label} in {execution.cwd} — "
              f"{status(execution)} · {execution.seconds:.1f}s")
    lines = [header]
    lines += _tail("output", execution.stdout)
    lines += _tail("errors", execution.stderr)
    if execution.timed_out:
        lines.append(TIMEOUT_NOTE.format(seconds=round(execution.seconds)))
    if len(lines) == 1:
        lines.append("(it printed nothing)")
    return "\n".join(lines)


def _tail(title: str, text: str) -> list[str]:
    """Give the last OUTPUT_TAIL_LINES lines of one stream under a title.

    Args:
        title (str): "output" or "errors".
        text (str): the captured stream.

    Returns:
        list[str]: [] for an empty stream, else the title and the lines.
    """
    rows = text.rstrip().splitlines()
    if not rows:
        return []
    cut = len(rows) - OUTPUT_TAIL_LINES
    head = [f"{title} (last {OUTPUT_TAIL_LINES} of {len(rows)} lines):"
            if cut > 0 else f"{title}:"]
    return head + rows[max(cut, 0):]


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        script = run_script('open("made.txt", "w").write("hi")\nprint("ok")',
                            Path(folder))
        assert script.ok and script.stdout == "ok\n", script
        assert (Path(folder) / "made.txt").read_text() == "hi"
        shell = run_command("ls", Path(folder))
        assert shell.ok and "made.txt" in shell.stdout, shell
        failed = run_command("exit 3", Path(folder))
        assert status(failed) == "exit 3" and not failed.ok
        print(report(shell))
    print("smoke OK")
