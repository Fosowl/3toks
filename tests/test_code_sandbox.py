"""Offline tests for the child-process runner (threetoks/code/sandbox.py).

Covers what a user relies on: secrets in the environment never reach
generated code (while interpreter selectors do), the run does not happen
in the user's cwd unless asked, a timeout or an interrupt kills the whole
process tree, output is capped and decoded safely, and a bad interpreter
is reported instead of raised.
"""
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from threetoks.code import sandbox

SHORT_TIMEOUT_S = 1
GRANDCHILD_SLEEP_S = 30
GRACE_S = 0.5


class EnvironmentTest(unittest.TestCase):
    def test_api_key_is_not_visible_to_the_child(self):
        os.environ["FAKE_API_KEY"] = "sk-secret"
        try:
            run = sandbox.run_python(
                "import os\nprint(os.environ.get('FAKE_API_KEY'))\n")
        finally:
            del os.environ["FAKE_API_KEY"]
        self.assertTrue(run.ok, run.stderr)
        self.assertEqual(run.stdout, "None\n")

    def test_python_path_variables_are_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "json.py").write_text("raise SystemExit(3)\n")
            os.environ["PYTHONPATH"] = folder
            try:
                run = sandbox.run_python("import json\nprint('real')\n")
            finally:
                del os.environ["PYTHONPATH"]
        self.assertEqual(run.stdout, "real\n")

    def test_interpreter_selectors_are_kept(self):
        with mock.patch.dict(os.environ, {"PYENV_VERSION": "3.12.1",
                                          "VIRTUAL_ENV": "/proj/.venv"}):
            run = sandbox.run_python(
                "import os\nprint(os.environ.get('PYENV_VERSION'),"
                " os.environ.get('VIRTUAL_ENV'))\n")
        self.assertEqual(run.stdout, "3.12.1 /proj/.venv\n")


class WorkingDirectoryTest(unittest.TestCase):
    def test_default_cwd_is_a_fresh_directory(self):
        run = sandbox.run_python("import os\nprint(os.getcwd())\n")
        self.assertNotEqual(Path(run.stdout.strip()).resolve(),
                            Path.cwd().resolve())

    def test_given_cwd_is_used(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "data.txt").write_text("from disk")
            run = sandbox.run_python("print(open('data.txt').read())\n",
                                     cwd=Path(folder))
        self.assertEqual(run.stdout, "from disk\n")

    def test_cwd_is_not_on_sys_path(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "random.py").write_text("raise SystemExit(4)\n")
            run = sandbox.run_python("import random\nprint('ok')\n",
                                     cwd=Path(folder))
        self.assertTrue(run.ok, run.stderr)

    def test_missing_cwd_fails_without_raising(self):
        run = sandbox.run_python("print(1)\n", cwd=Path("/no/such/dir"))
        self.assertFalse(run.ok)
        self.assertIn("working directory not found", run.stderr)


class TimeoutTest(unittest.TestCase):
    def test_timeout_is_reported(self):
        run = sandbox.run_python("import time\ntime.sleep(10)\n",
                                 timeout_s=SHORT_TIMEOUT_S)
        self.assertTrue(run.timed_out)
        self.assertFalse(run.ok)

    @unittest.skipUnless(os.name == "posix", "process groups are POSIX")
    def test_timeout_kills_a_grandchild(self):
        with tempfile.TemporaryDirectory() as folder:
            pid_file = Path(folder, "pid")
            source = ("import subprocess, sys, time\n"
                      "child = subprocess.Popen([sys.executable, '-c', "
                      f"'import time; time.sleep({GRANDCHILD_SLEEP_S})'])\n"
                      f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
                      "time.sleep(10)\n")
            run = sandbox.run_python(source, timeout_s=SHORT_TIMEOUT_S)
            grandchild = int(pid_file.read_text())
        self.assertTrue(run.timed_out)
        time.sleep(GRACE_S)
        self.assertFalse(_alive(grandchild))

    @unittest.skipUnless(os.name == "posix", "process groups are POSIX")
    def test_interrupt_during_wait_kills_the_child(self):
        real_wait = subprocess.Popen.wait
        started = []

        def interrupted_wait(process, timeout=None):
            if timeout is None:
                return real_wait(process)
            started.append(process.pid)
            raise KeyboardInterrupt

        with mock.patch.object(subprocess.Popen, "wait", autospec=True,
                               side_effect=interrupted_wait):
            with self.assertRaises(KeyboardInterrupt):
                sandbox.run_python("import time\ntime.sleep(10)\n")
        [child] = started
        self.assertFalse(_alive(child))


class OutputTest(unittest.TestCase):
    def test_output_is_capped(self):
        run = sandbox.run_python(
            f"print('x' * {sandbox.MAX_OUTPUT_CHARS * 2})\n")
        self.assertTrue(run.ok)
        self.assertEqual(len(run.stdout), sandbox.MAX_OUTPUT_CHARS)

    def test_non_utf8_output_is_replaced_not_raised(self):
        run = sandbox.run_python(
            "import sys\nsys.stdout.buffer.write(b'a\\xffb')\n"
            "sys.stderr.buffer.write(b'\\xfe')\n")
        self.assertEqual(run.stdout, "a\ufffdb")
        self.assertEqual(run.stderr, "\ufffd")

    def test_failure_keeps_stderr_and_partial_stdout(self):
        run = sandbox.run_python("print('before')\n1 // 0\n")
        self.assertFalse(run.ok)
        self.assertFalse(run.timed_out)
        self.assertEqual(run.stdout, "before\n")
        self.assertIn("ZeroDivisionError", run.stderr)

    def test_input_hits_eof(self):
        run = sandbox.run_python("input()\n")
        self.assertIn("EOFError", run.stderr)


class InterpreterTest(unittest.TestCase):
    def test_bad_interpreter_fails_without_raising(self):
        run = sandbox.run_python("print(1)\n",
                                 interpreter="/no/such/python")
        self.assertFalse(run.ok)
        self.assertIn("interpreter not found", run.stderr)

    def test_project_venv_python_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            venv_python = Path(folder, ".venv", "bin", "python")
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("")
            self.assertEqual(sandbox.project_python(Path(folder)),
                             str(venv_python))

    def test_without_venv_some_python_is_found(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(sandbox.project_python(Path(folder)))


def _alive(pid: int) -> bool:
    """True when a process with this pid still runs (not a zombie)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    unittest.main()
