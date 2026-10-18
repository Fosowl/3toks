"""Tests for the config screen: key handling, rendering, session."""
import io
import unittest

from threetoks.config import ThreetoksConfig
from threetoks.config_ui import Session
from threetoks.probes import Probes
from threetoks.screen_state import (ACTION_PROBE, ACTION_QUIT, ACTION_SAVE,
                                    Screen, current_field, handle_key,
                                    unsaved)
from threetoks.screen_view import render


def make_screen(probes=None, config=None):
    base = config or ThreetoksConfig()
    return Screen(base, base, probes or Probes(), "x.ini")


def press(screen, *keys):
    return [handle_key(screen, key) for key in keys]


class BrowseTest(unittest.TestCase):
    def test_cursor_wraps_and_tab_changes_section(self):
        screen = make_screen()
        press(screen, "up")
        self.assertEqual(current_field(screen).key, "disable_think")
        press(screen, "tab")
        self.assertEqual((screen.section, screen.row), (1, 0))
        press(screen, "btab", "btab")
        self.assertEqual(screen.section, 9)

    def test_digit_jumps_to_a_section(self):
        screen = make_screen()
        press(screen, "3")
        self.assertEqual(current_field(screen).dotted, "search.searxng_url")
        press(screen, "0")
        self.assertEqual(current_field(screen).dotted, "relay.pin")

    def test_non_ascii_digit_keys_are_ignored_not_fatal(self):
        screen = make_screen()   # "²" is one key on an AZERTY keyboard
        self.assertIsNone(handle_key(screen, "²"))
        self.assertEqual(screen.section, 0)

    def test_arrows_cycle_choice_and_flip_bool(self):
        screen = make_screen()
        press(screen, "right")
        self.assertEqual(screen.config.llm.provider, "anthropic")
        press(screen, "6", "enter")   # section 6 = LOGBOOK, enabled flag
        self.assertFalse(screen.config.memory.enabled)

    def test_reset_restores_the_default(self):
        screen = make_screen()
        press(screen, "4", "right", "r")
        self.assertEqual(screen.config.research.max_rounds, 3)


class EditBoxTest(unittest.TestCase):
    def test_typed_text_is_stored_on_enter(self):
        screen = make_screen()
        press(screen, "down", "enter", "ctrl-u", "m", "1", "enter")
        self.assertEqual(screen.config.llm.model, "m1")
        self.assertIsNone(screen.edit)

    def test_bad_number_keeps_the_box_open_with_an_error(self):
        screen = make_screen()
        press(screen, "4", "enter", "ctrl-u", "0", "enter")
        self.assertEqual(screen.config.research.max_rounds, 3)
        self.assertIsNotNone(screen.edit)
        self.assertIn("at least 1", screen.message)

    def test_esc_in_the_box_drops_the_text_only(self):
        screen = make_screen()
        press(screen, "down", "enter", "x", "esc")
        self.assertIsNone(screen.edit)
        self.assertEqual(screen.config.llm.model, "qwen3.5:2b")

    def test_q_and_digits_are_text_while_typing(self):
        screen = make_screen()
        press(screen, "down", "enter", "ctrl-u", "q", "2", "enter")
        self.assertEqual(screen.config.llm.model, "q2")


class LeaveTest(unittest.TestCase):
    def test_clean_screen_leaves_at_once(self):
        self.assertEqual(handle_key(make_screen(), "esc"), ACTION_QUIT)

    def test_unsaved_changes_need_a_second_esc(self):
        screen = make_screen()
        press(screen, "right")
        self.assertIsNone(handle_key(screen, "esc"))
        self.assertEqual(handle_key(screen, "esc"), ACTION_QUIT)

    def test_any_other_key_cancels_the_warning(self):
        screen = make_screen()
        press(screen, "right", "esc", "down")
        self.assertIsNone(handle_key(screen, "esc"))

    def test_save_key_returns_the_save_action(self):
        self.assertEqual(handle_key(make_screen(), "s"), ACTION_SAVE)


class ProbeAndSuggestTest(unittest.TestCase):
    def test_model_arrows_walk_the_pulled_models(self):
        screen = make_screen(Probes(ollama_up=True, models=("a", "b")))
        press(screen, "down", "right")
        self.assertEqual(screen.config.llm.model, "a")
        press(screen, "right", "right")
        self.assertEqual(screen.config.llm.model, "a")  # wrapped

    def test_model_arrow_without_models_says_to_type(self):
        screen = make_screen()
        press(screen, "down", "right")
        self.assertEqual(screen.config.llm.model, "qwen3.5:2b")
        self.assertIn("press enter", screen.message)

    def test_changing_the_provider_asks_for_a_new_probe(self):
        self.assertEqual(handle_key(make_screen(), "right"), ACTION_PROBE)


class SessionTest(unittest.TestCase):
    class Keys:
        def __init__(self, keys):
            self.keys = list(keys)

        def read(self):
            return self.keys.pop(0)

    def run_session(self, keys, **kwargs):
        saved = []
        screen = make_screen()
        session = Session(screen, self.Keys(keys), io.StringIO(),
                          enabled=False, save=lambda c: saved.append(c) or "p",
                          probe=lambda c: Probes(ollama_up=True), **kwargs)
        return session, session.run(), saved

    def test_save_then_leave_returns_the_saved_config(self):
        _, result, saved = self.run_session(["tab", "right", "s", "esc"])
        self.assertEqual(result.browser.mode, "plain")
        self.assertEqual(len(saved), 1)

    def test_leaving_without_saving_returns_none(self):
        _, result, saved = self.run_session(["esc"])
        self.assertIsNone(result)
        self.assertEqual(saved, [])

    def test_ctrl_c_leaves_but_keeps_an_earlier_save(self):
        class Boom(self.Keys):
            def read(self):
                if not self.keys:
                    raise KeyboardInterrupt
                return super().read()
        screen = make_screen()
        session = Session(screen, Boom(["tab", "right", "s"]), io.StringIO(),
                          enabled=False, save=lambda c: "p")
        self.assertEqual(session.run().browser.mode, "plain")

    def test_a_failed_save_is_shown_not_raised(self):
        def refuse(_config):
            raise PermissionError("read-only folder")
        screen = make_screen()
        class Then(self.Keys):
            def read(self):
                if not self.keys:
                    raise KeyboardInterrupt   # the user gives up
                return super().read()
        session = Session(screen, Then(["right", "s"]), io.StringIO(),
                          enabled=False, save=refuse,
                          probe=lambda c: Probes())
        self.assertIsNone(session.run())
        self.assertIn("cannot save", screen.message)
        self.assertEqual(len(unsaved(screen)), 1)

    def test_probe_action_refreshes_the_probes(self):
        session, _, _ = self.run_session(["right", "s", "esc"])
        self.assertTrue(session.screen.probes.ollama_up)

    def test_screen_uses_the_alternate_buffer_and_restores_it(self):
        session, _, _ = self.run_session(["esc"])
        out = session.out.getvalue()
        self.assertTrue(out.startswith("\x1b[?1049h"))
        self.assertIn("\x1b[?1049l", out)


class RenderTest(unittest.TestCase):
    def test_height_never_changes_between_sections(self):
        screen = make_screen()
        heights = set()
        for key in ["tab"] * 10:
            press(screen, key)
            heights.add(len(render(screen, False, 70)))
        self.assertEqual(len(heights), 1)

    def test_unsaved_marker_and_counter(self):
        screen = make_screen()
        press(screen, "tab", "right")
        text = "\n".join(render(screen, False))
        self.assertIn("NETS•", text)
        self.assertIn("1 unsaved", text)
        self.assertEqual(len(unsaved(screen)), 1)

    def test_plain_output_has_no_escape_codes(self):
        self.assertNotIn("\x1b", "\n".join(render(make_screen(), False)))

    def test_narrow_terminal_still_draws_the_active_tab(self):
        screen = make_screen()
        press(screen, "tab")
        text = "\n".join(render(screen, False, 40))
        self.assertIn("2/10 NETS", text)


class ConfigCommandGuardTest(unittest.TestCase):
    """/config must never end the chat session."""

    def run_with(self, effect):
        from unittest import mock
        from threetoks import config_ui

        class State:
            enabled = False
            config_stamp = None  # a hand-built state: no live reload
        with mock.patch.object(config_ui, "run_screen", side_effect=effect):
            return config_ui.config_command(State())

    def test_ctrl_c_during_the_probe_sweep_is_a_quiet_close(self):
        self.assertIn("nothing changed", self.run_with(KeyboardInterrupt))

    def test_any_failure_becomes_a_one_line_message(self):
        reply = self.run_with(RuntimeError("boom"))
        self.assertIn("config screen failed: boom", reply)


if __name__ == "__main__":
    unittest.main()
