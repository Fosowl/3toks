"""Offline tests for real runs: delivered scripts, shell commands, reports."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code import execute, shell
from threetoks.policy import Policy, PolicyConfig


def folder(test) -> Path:
    """A temp project folder removed after the test."""
    holder = tempfile.TemporaryDirectory()
    test.addCleanup(holder.cleanup)
    return Path(holder.name)


class RunScriptTest(unittest.TestCase):
    def test_script_runs_in_the_folder_and_reports_its_exit_code(self):
        root = folder(self)
        run = execute.run_script("import sys\nopen('x.txt','w').write('1')\n"
                                 "print('done')\nsys.exit(2)", root)
        self.assertEqual((run.ok, run.exit_code, run.stdout),
                         (False, 2, "done\n"))
        self.assertTrue((root / "x.txt").exists())

    def test_a_timeout_is_named_in_the_status_and_the_report(self):
        with mock.patch.object(execute, "SCRIPT_TIMEOUT_S", 0.5):
            run = execute.run_script("import time\ntime.sleep(5)",
                                     folder(self))
        self.assertEqual(execute.status(run), "timed out")
        self.assertIn("stopped: it ran longer", execute.report(run))


class RunCommandTest(unittest.TestCase):
    def test_command_sees_the_folder_and_its_exit_code(self):
        root = folder(self)
        (root / "a.txt").write_text("hi")
        run = execute.run_command("cat a.txt; exit 4", root)
        self.assertEqual((run.stdout, run.exit_code), ("hi", 4))
        self.assertEqual(execute.status(run), "exit 4")

    def test_a_long_command_is_stopped(self):
        with mock.patch.object(execute, "COMMAND_TIMEOUT_S", 0.5):
            run = execute.run_command("sleep 5", folder(self))
        self.assertTrue(run.timed_out)
        self.assertIsNone(run.exit_code)


class ReportTest(unittest.TestCase):
    def _run(self, stdout="", stderr=""):
        return execute.Execution("$ x", Path("/p"), True, 0, stdout, stderr,
                                 0.25)

    def test_long_output_keeps_only_the_tail_with_a_count(self):
        lines = [f"line {n}" for n in range(100)]
        text = execute.report(self._run(stdout="\n".join(lines)))
        self.assertIn(f"last {execute.OUTPUT_TAIL_LINES} of 100 lines", text)
        self.assertIn("line 99", text)
        self.assertNotIn("line 69\n", text)

    def test_errors_follow_the_output(self):
        text = execute.report(self._run("ok", "warn"))
        self.assertLess(text.index("ok"), text.index("errors:"))

    def test_silent_run_says_so(self):
        self.assertIn("(it printed nothing)", execute.report(self._run()))


class ProposeCommandTest(unittest.TestCase):
    def _propose(self, reply, root):
        prompts: list[str] = []

        class _Backend:
            def complete(self, model, raw_prompt, opts):
                prompts.append(raw_prompt)
                return GenResult(reply, 5, 2, 0.0, "stop")

        policy = Policy(_Backend(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        return shell.propose_command("list it", root, policy), prompts

    def test_backticks_and_prompt_sign_are_removed(self):
        command, _ = self._propose(" `$ ls -la`", folder(self))
        self.assertEqual(command, "ls -la")

    def test_the_model_sees_the_folder_files_and_python(self):
        root = folder(self)
        (root / "main.py").write_text("")
        (root / "data").mkdir()
        _, prompts = self._propose(" ls", root)
        self.assertIn("FILES: data/, main.py", prompts[0])
        self.assertIn("PYTHON: ", prompts[0])
        self.assertIn(f"FOLDER: {root.resolve()}", prompts[0])

    def test_python_and_pip_words_become_the_project_python(self):
        root = folder(self)
        venv = root / ".venv" / "bin"
        venv.mkdir(parents=True)
        (venv / "python").write_text("")
        python = str(venv / "python")
        cases = {"PYTHON -m pytest": f"{python} -m pytest",
                 "python3 main.py": f"{python} main.py",
                 "pip install numpy": f"{python} -m pip install numpy",
                 "git status": "git status",
                 "echo python": "echo python"}
        for line, expected in cases.items():
            self.assertEqual(shell.with_project_python(line, root), expected)


if __name__ == "__main__":
    unittest.main()
