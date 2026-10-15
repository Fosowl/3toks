"""Run Python source in a child process with limited reach.

The code agent runs model-written code and the user's own code. This
module gives every such run the same isolation:

- isolated interpreter flags (``-I -B -X utf8``): no user site, no
  ``PYTHON*`` variables, no cwd on ``sys.path``, no ``.pyc`` files;
- a fresh temporary working directory unless the caller gives one;
- an environment that keeps only an allowlist (never API keys);
- output written to temporary files and read back with a size cap;
- on POSIX, a new process group that is killed as a whole after the
  run, plus best-effort CPU, file-size and memory limits.

This is NOT a security sandbox: the child can still use the network and
read or write any file the user can. On Windows there are no resource
limits and only the direct child is killed.
"""
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 5
MAX_OUTPUT_CHARS = 200_000          # per stream, after decode
ISOLATED_FLAGS = ("-I", "-B", "-X", "utf8", "-")
ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR",
                 "SYSTEMROOT", "TERM", "PYENV_VERSION", "VIRTUAL_ENV")
CPU_GRACE_S = 1
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_MEMORY_BYTES = 2 * 1024 * 1024 * 1024
MAX_BYTES_PER_CHAR = 4               # utf-8 worst case
TEMP_PREFIX = "threetoks-run-"
VENV_PYTHONS = (".venv/bin/python", ".venv/Scripts/python.exe")
_POSIX = os.name == "posix"
_LINUX = sys.platform.startswith("linux")


@dataclass(frozen=True)
class RunResult:
    """The outcome of one child run."""
    ok: bool          # exit code 0 and not timed out
    stdout: str       # capped, utf-8 decoded with errors="replace"
    stderr: str       # capped, same decoding
    timed_out: bool


def run_python(source: str, cwd: Path | None = None,
               timeout_s: float = DEFAULT_TIMEOUT_S,
               interpreter: str | None = None) -> RunResult:
    """Run Python source in an isolated child process.

    The gates, the runner and the test oracle call this to execute
    generated code or the user's code. The source goes to the child on
    stdin. The function never raises for a failure of the child.

    Args:
        source: Python program text, from the caller.
        cwd: working directory for the child; None gives a fresh
            temporary directory that is removed afterwards.
        timeout_s: wall-clock limit in seconds.
        interpreter: path of the Python to run; None uses sys.executable.

    Returns:
        RunResult: success flag, capped stdout and stderr, timeout flag.
    """
    argv = [interpreter or sys.executable, *ISOLATED_FLAGS]
    if cwd is not None:
        return _run_in(argv, source, Path(cwd), timeout_s)
    with tempfile.TemporaryDirectory(prefix=TEMP_PREFIX,
                                     ignore_cleanup_errors=True) as scratch:
        return _run_in(argv, source, Path(scratch), timeout_s)


def project_python(root: Path) -> str:
    """The interpreter that runs the user's project code.

    The oracle and the compute mode call this so that the user's code
    runs with the project's own packages. A project virtualenv wins;
    else the first python on PATH; else the harness interpreter.

    Args:
        root: project root folder, from the agent services.

    Returns:
        str: path or command name of the interpreter.
    """
    for relative in VENV_PYTHONS:
        candidate = Path(root) / relative
        if candidate.exists():
            return str(candidate)
    return shutil.which("python3") or shutil.which("python") or sys.executable


def _run_in(argv: list[str], source: str, cwd: Path,
            timeout_s: float) -> RunResult:
    """Start the child in cwd, wait for it, and collect its output.

    run_python calls this after it selects the working directory.

    Args:
        argv: interpreter command line, from run_python.
        source: program text, from run_python.
        cwd: existing working directory, from run_python.
        timeout_s: wall-clock limit in seconds, from run_python.

    Returns:
        RunResult: the outcome; a start failure gives ok False.
    """
    if not cwd.is_dir():
        return RunResult(False, "", f"working directory not found: {cwd}",
                         False)
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as out, \
            tempfile.TemporaryFile() as err:
        stdin.write(source.encode("utf-8", errors="replace"))
        stdin.seek(0)
        started = _spawn(argv, cwd, timeout_s, (stdin, out, err))
        if isinstance(started, RunResult):
            return started
        timed_out = _wait(started, timeout_s)
        ok = started.returncode == 0 and not timed_out
        return RunResult(ok, _read_capped(out), _read_capped(err), timed_out)


def _spawn(argv: list[str], cwd: Path, timeout_s: float,
           streams: tuple) -> "subprocess.Popen | RunResult":
    """Start the child process, or report why it could not start.

    _run_in calls this with the three open temporary files.

    Args:
        argv: interpreter command line, from run_python.
        cwd: working directory, from _run_in.
        timeout_s: wall-clock limit, used for the CPU limit.
        streams: (stdin, stdout, stderr) binary files, from _run_in.

    Returns:
        subprocess.Popen | RunResult: the running child, or a failed
        RunResult when the interpreter cannot start.
    """
    stdin, out, err = streams
    try:
        return subprocess.Popen(
            argv, stdin=stdin, stdout=out, stderr=err, cwd=cwd,
            env=_child_env(), start_new_session=_POSIX,
            preexec_fn=_limiter(timeout_s) if _POSIX else None)
    except FileNotFoundError:
        return RunResult(False, "", f"interpreter not found: {argv[0]}", False)
    except OSError as error:
        return RunResult(False, "", f"cannot start interpreter: {error}",
                         False)


def _child_env() -> dict[str, str]:
    """The environment of the child: allowlisted variables only.

    _spawn calls this so that secrets such as API keys never reach
    generated code.

    Returns:
        dict[str, str]: the allowlisted variables that are set now.
    """
    return {name: os.environ[name] for name in ENV_ALLOWLIST
            if name in os.environ}


def _limiter(timeout_s: float):
    """A function that sets resource limits inside the child (POSIX).

    _spawn gives the result to Popen as preexec_fn. Each limit is best
    effort: a platform that refuses a limit keeps running without it.

    Args:
        timeout_s: wall-clock limit; the CPU limit is this plus a grace.

    Returns:
        callable: a zero-argument function for preexec_fn.
    """
    import resource

    limits = [(resource.RLIMIT_CPU, math.ceil(timeout_s) + CPU_GRACE_S),
              (resource.RLIMIT_FSIZE, MAX_FILE_BYTES)]
    if _LINUX:
        limits.append((resource.RLIMIT_AS, MAX_MEMORY_BYTES))

    def apply_limits() -> None:
        for kind, value in limits:
            _set_limit(resource, kind, value)

    return apply_limits


def _set_limit(resource, kind: int, value: int) -> None:
    """Lower one resource limit; ignore a refusal.

    The preexec function from _limiter calls this in the child.

    Args:
        resource: the resource module, from _limiter.
        kind: the RLIMIT_* constant.
        value: the wanted soft and hard limit.
    """
    try:
        _, hard = resource.getrlimit(kind)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(kind, (value, value))
    except (ValueError, OSError):
        return


def _wait(process: subprocess.Popen, timeout_s: float) -> bool:
    """Wait for the child, then kill what is left of its process tree.

    _run_in calls this. The group is killed and reaped in ``finally``:
    after a normal exit, after a timeout, and also when the wait is
    interrupted (KeyboardInterrupt or any other exception), so a child
    or background grandchild never survives the run.

    Args:
        process: the running child, from _spawn.
        timeout_s: wall-clock limit in seconds.

    Returns:
        bool: True when the child did not finish in time.
    """
    try:
        process.wait(timeout=timeout_s)
        return False
    except subprocess.TimeoutExpired:
        return True
    finally:
        _kill_tree(process)
        process.wait()


def _kill_tree(process: subprocess.Popen) -> None:
    """Kill the child's process group (POSIX) or the child (Windows).

    _wait calls this when the wait ends, for any reason.

    Args:
        process: the child, from _spawn.
    """
    if not _POSIX:
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        return                    # group already gone


def _read_capped(stream) -> str:
    """Read a temporary output file back as capped text.

    _run_in calls this for stdout and for stderr.

    Args:
        stream: binary temporary file the child wrote to.

    Returns:
        str: at most MAX_OUTPUT_CHARS characters, utf-8 with replacement.
    """
    stream.seek(0)
    raw = stream.read(MAX_OUTPUT_CHARS * MAX_BYTES_PER_CHAR)
    return raw.decode("utf-8", errors="replace")[:MAX_OUTPUT_CHARS]


if __name__ == "__main__":
    os.environ["FAKE_API_KEY"] = "secret"
    leaked = run_python("import os\nprint(os.environ.get('FAKE_API_KEY'))\n")
    assert leaked.ok and leaked.stdout == "None\n", leaked
    where = run_python("import os\nprint(os.getcwd())\n")
    assert where.stdout.strip() != os.getcwd(), where
    slow = run_python("import time\ntime.sleep(10)\n", timeout_s=0.5)
    assert slow.timed_out and not slow.ok, slow
    missing = run_python("print(1)\n", interpreter="/no/such/python")
    assert not missing.ok and "interpreter not found" in missing.stderr
    noisy = run_python(f"print('x' * {MAX_OUTPUT_CHARS * 2})\n")
    assert len(noisy.stdout) == MAX_OUTPUT_CHARS, len(noisy.stdout)
    binary = run_python("import sys\nsys.stdout.buffer.write(b'\\xff')\n")
    assert binary.stdout == "\ufffd", binary
    assert project_python(Path("/nonexistent"))
    print("smoke OK")
