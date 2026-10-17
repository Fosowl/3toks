"""Tests for the ``threetoks`` command (flags, config tools, doctor)."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from threetoks import launch
from threetoks.config import CONFIG_ENV_VAR, load_config
from threetoks.line_input import ARG_CHOICES, candidates, wrap_prompt
from threetoks.probes import Probes


class CliCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = os.path.join(folder.name, "c.ini")
        patcher = mock.patch.dict(os.environ, {CONFIG_ENV_VAR: self.path})
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = launch.main(list(argv))
        return code, out.getvalue(), err.getvalue()


class ConfigCommandTest(CliCase):
    def test_set_then_get_round_trips_and_writes_the_file(self):
        code, out, _ = self.run_cli("config", "set", "llm.model", "mine")
        self.assertEqual(code, 0)
        self.assertIn("(live)", out)
        self.assertEqual(self.run_cli("config", "get", "llm.model")[1],
                         "mine\n")
        self.assertEqual(load_config(self.path).llm.model, "mine")

    def test_bad_value_exits_2_and_writes_nothing(self):
        code, _, err = self.run_cli("config", "set", "research.max_rounds",
                                    "0")
        self.assertEqual(code, launch.EXIT_BAD_INPUT)
        self.assertIn("at least 1", err)
        self.assertFalse(os.path.exists(self.path))

    def test_unknown_key_exits_2(self):
        for action in ("get", "set"):
            code, _, err = self.run_cli("config", action, "llm.nope", "x")
            self.assertEqual(code, launch.EXIT_BAD_INPUT)
            self.assertIn("unknown setting", err)

    def test_bool_is_stored_as_true_false(self):
        self.run_cli("config", "set", "browser.visible", "on")
        self.assertEqual(self.run_cli("config", "get", "browser.visible")[1],
                         "true\n")

    def test_path_reports_a_missing_file(self):
        _, out, _ = self.run_cli("config", "path")
        self.assertIn(self.path, out)
        self.assertIn("not created yet", out)

    def test_show_lists_every_section_without_a_terminal(self):
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            _, out, _ = self.run_cli("config")
        for title in ("HELM", "SONAR", "BEACON"):
            self.assertIn(title, out)
        self.assertIn("llm.model", out)


class FlagTest(CliCase):
    def test_config_flag_sets_the_env_var(self):
        self.run_cli("-c", "/tmp/other.ini", "config", "path")
        self.assertEqual(os.environ[CONFIG_ENV_VAR], "/tmp/other.ini")

    def test_model_flag_reaches_the_chat(self):
        with mock.patch("threetoks.tui.main") as chat:
            self.assertEqual(launch.main(["-m", "tiny"]), 0)
        chat.assert_called_once_with("tiny")

    def test_no_arguments_start_the_chat_with_no_model(self):
        with mock.patch("threetoks.tui.main") as chat:
            launch.main([])
        chat.assert_called_once_with(None)

    def test_version_flag_exits_cleanly(self):
        with self.assertRaises(SystemExit) as stop, \
                contextlib.redirect_stdout(io.StringIO()) as out:
            launch.main(["--version"])
        self.assertEqual(stop.exception.code, 0)
        self.assertIn("threetoks 0.", out.getvalue())


class DoctorTest(CliCase):
    def doctor(self, probes):
        with mock.patch("threetoks.probes.sweep", return_value=probes), \
                mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            return self.run_cli("doctor")

    def test_missing_model_makes_doctor_fail_with_the_fix(self):
        code, out, _ = self.doctor(Probes(ollama_up=True, models=("x",),
                                          searxng_up=True))
        self.assertEqual(code, 1)
        self.assertIn("ollama pull qwen3.5:2b", out)

    def test_healthy_setup_exits_0(self):
        code, out, _ = self.doctor(Probes(ollama_up=True,
                                          models=("qwen3.5:2b",),
                                          searxng_up=True))
        self.assertEqual(code, 0)
        self.assertIn("pulled", out)

    def test_a_broken_ini_file_fails_doctor(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("not a comment\n[llm]\nmodel = x\n")
        code, out, _ = self.doctor(Probes(ollama_up=True, models=("x",),
                                          searxng_up=True))
        self.assertEqual(code, 1)
        self.assertIn("was ignored", out)


class LineInputTest(unittest.TestCase):
    COMMANDS = ("/help", "/hello", "/model", "/browser", "/debug")

    def test_command_names_complete_by_prefix(self):
        self.assertEqual(candidates("/he", self.COMMANDS),
                         ["/hello", "/help"])

    def test_plain_chat_text_never_completes(self):
        self.assertEqual(candidates("he", self.COMMANDS), [])

    def test_arguments_complete_for_known_commands_only(self):
        self.assertEqual(candidates("/debug f", self.COMMANDS), ["full"])
        self.assertEqual(candidates("/model q", self.COMMANDS), [])
        self.assertIn("/browser", ARG_CHOICES)

    def test_colour_codes_are_wrapped_as_zero_width(self):
        prompt = "\x1b[38;5;51m❯\x1b[0m "
        self.assertEqual(wrap_prompt(prompt),
                         "\x01\x1b[38;5;51m\x02❯\x01\x1b[0m\x02 ")
        self.assertEqual(wrap_prompt("no colour"), "no colour")


if __name__ == "__main__":
    unittest.main()
