"""Offline tests for the TUI: no LLM, no network.

Covers ticker-line formatting from fake events, NO_COLOR stripping, slash
command dispatch with injected specs and a scripted input function, and
result-panel rendering with missing optional keys.
"""
import random
import re
import unittest
from dataclasses import dataclass
from unittest import mock

from threetoks import tui


@dataclass
class FakeSpec:
    """Stand-in AgentSpec: only name/description matter for the TUI."""
    name: str
    description: str


class _RecordingFetcher:
    """Browser-fetcher double: exposes ``fetch_page`` and counts ``close``."""

    def __init__(self):
        self.closed = 0

    def fetch_page(self, url):
        return url

    def close(self):
        self.closed += 1


class FakeServices:
    """Minimal Services double the slash commands mutate."""

    def __init__(self):
        self.max_research_rounds = 3
        self.fetch_page = None


def make_state(**overrides):
    """A ReplState wired with fake specs and a recording policy factory."""
    built = []

    def factory(name):
        policy = object()
        built.append(name)
        return policy

    state = tui.ReplState(
        services=FakeServices(),
        specs=[FakeSpec("casual", "small talk"),
               FakeSpec("web", "internet research")],
        policy=object(),
        model="qwen2.5:1.5b-instruct",
        policy_factory=factory,
        enabled=overrides.get("enabled", True),
    )
    state._built = built  # expose for assertions
    return state


class TickerFormatTest(unittest.TestCase):
    def test_menu_event_shows_kind_value_and_stats(self):
        event = {"node": "menu", "value": "open result 2",
                 "valid": True, "out_tokens": 2, "wall_s": 0.31}
        line = tui.build_ticker_line(event, enabled=False)
        self.assertIn("menu", line)
        self.assertIn("open result 2", line)
        self.assertIn("2t·0.31s", line)

    def test_stats_column_aligns_across_events(self):
        short = {"node": "menu", "value": "go", "valid": True,
                 "out_tokens": 2, "wall_s": 0.1}
        longer = {"node": "menu", "value": "open result 2: pricing",
                  "valid": True, "out_tokens": 3, "wall_s": 0.2}
        col = tui.build_ticker_line(short, enabled=False).index("2t·")
        self.assertEqual(col,
                         tui.build_ticker_line(longer,
                                               enabled=False).index("3t·"))

    def test_token_bar_grows_with_cost_and_saturates(self):
        self.assertEqual(tui._token_bar(0), "·")
        self.assertEqual(tui._token_bar(1), tui.TOKEN_BARS[0])
        self.assertEqual(tui._token_bar(3), tui.TOKEN_BARS[2])
        self.assertEqual(tui._token_bar(99), tui.TOKEN_BARS[-1])

    def test_pace_color_maps_speed_to_green_amber_pink(self):
        self.assertEqual(tui._pace_color(0.1), tui.GREEN)
        self.assertEqual(tui._pace_color(0.8), tui.AMBER)
        self.assertEqual(tui._pace_color(2.5), tui.PINK)

    def test_malformed_numeric_fields_do_not_crash(self):
        event = {"node": "menu", "value": "x", "valid": True,
                 "out_tokens": "abc", "wall_s": None}
        self.assertIn("0t·0.00s", tui.build_ticker_line(event, enabled=False))

    def test_pick_many_shows_indices(self):
        event = {"node": "pick_many", "value": [3, 7], "valid": True,
                 "out_tokens": 5, "wall_s": 0.4}
        self.assertIn("3,7", tui.build_ticker_line(event, enabled=False))

    def test_short_text_clips_to_forty_chars(self):
        long_value = "a" * 80
        event = {"node": "short_text", "value": long_value, "valid": True}
        line = tui.build_ticker_line(event, enabled=False)
        self.assertIn("…", line)
        self.assertNotIn("a" * 60, line)

    def test_invalid_event_renders_dash_without_crashing(self):
        line = tui.build_ticker_line({"node": "menu", "valid": False},
                                     enabled=False)
        self.assertIn("menu", line)
        self.assertIn("—", line)


class NoColorTest(unittest.TestCase):
    def test_ticker_has_no_escape_when_disabled(self):
        event = {"node": "menu", "value": "x", "valid": True,
                 "out_tokens": 1, "wall_s": 0.1}
        self.assertNotIn("\x1b", tui.build_ticker_line(event, enabled=False))

    def test_panel_and_banner_have_no_escape_when_disabled(self):
        panel = tui.render_result({"agent": "web", "answer": "hi"},
                                  enabled=False)
        banner = tui.build_banner("m", 2, enabled=False)
        self.assertNotIn("\x1b", panel)
        self.assertNotIn("\x1b", banner)

    def test_env_no_color_disables_color(self):
        import os
        os.environ["NO_COLOR"] = "1"
        try:
            self.assertFalse(tui._color_enabled())
        finally:
            del os.environ["NO_COLOR"]


class GradientTest(unittest.TestCase):
    def test_disabled_gradient_returns_text_unchanged(self):
        self.assertEqual(tui.gradient("ThreeToks", enabled=False), "ThreeToks")

    def test_enabled_gradient_spans_the_ramp(self):
        colored = tui.gradient("ThreeToks", enabled=True)
        self.assertIn(f"38;5;{tui.RAMP[0]}m", colored)   # starts cyan
        self.assertIn(f"38;5;{tui.RAMP[-1]}m", colored)  # ends pink

    def test_hairline_is_panel_width_with_one_node(self):
        line = tui.hairline(6, enabled=False)
        self.assertEqual(len(line), tui.PANEL_WIDTH)
        self.assertEqual(line.count(tui.NODE), 1)


class ResultPanelTest(unittest.TestCase):
    def test_renders_answer_and_stats(self):
        result = {"agent": "web", "answer": "42 lightyears",
                  "notes": "[1] source", "decisions": 9, "tokens": 64,
                  "seconds": 11.2, "rounds": 2}
        panel = tui.render_result(result, enabled=False)
        self.assertIn("42 lightyears", panel)
        self.assertIn("9 decisions", panel)
        self.assertIn("64 tok", panel)
        self.assertIn("2 rounds", panel)

    def test_missing_optional_keys_do_not_crash(self):
        panel = tui.render_result({}, enabled=False)
        self.assertIn("—", panel)  # answer falls back to a dash
        self.assertIn("0 decisions", panel)
        self.assertNotIn("rounds", panel)  # omitted when absent

    def test_every_panel_line_sits_on_the_spine(self):
        panel = tui.render_result({"agent": "web", "answer": "hi",
                                   "notes": "[1] source"}, enabled=False)
        for line in panel.splitlines():
            self.assertTrue(line.startswith(" " + tui.SPINE), repr(line))

    def test_unverified_result_is_labelled(self):
        panel = tui.render_result({"agent": "web", "answer": "hi",
                                   "judged_good": False}, enabled=False)
        self.assertIn("UNVERIFIED", panel)

    def test_error_panel_shows_cross_and_message(self):
        panel = tui.render_error(RuntimeError("ollama down"), enabled=False)
        self.assertIn("✖", panel)
        self.assertIn("RuntimeError: ollama down", panel)
        for line in panel.splitlines():
            self.assertTrue(line.startswith(" " + tui.SPINE), repr(line))

    def test_error_panel_hints_on_known_failures(self):
        missing = tui.render_error(
            FileNotFoundError("casual_replies.json"), enabled=False)
        self.assertIn("reinstall", missing)
        connection = tui.render_error(
            ConnectionError("connection refused"), enabled=False)
        self.assertIn("Ollama", connection)
        plain = tui.render_error(RuntimeError("boom"), enabled=False)
        self.assertNotIn("reinstall", plain)  # no hint without a match

    def test_none_seconds_does_not_crash_the_panel(self):
        panel = tui.render_result({"agent": "web", "answer": "hi",
                                   "seconds": None}, enabled=False)
        self.assertIn("0.0s", panel)

    def test_target_line_rendered_when_present(self):
        panel = tui.render_result(
            {"agent": "web", "answer": "hi",
             "target": {"label": "a github user", "anchor": "the repo"}},
            enabled=False)
        self.assertIn("target: a github user — the repo", panel)

    def test_absent_or_none_target_renders_nothing(self):
        for target in ({}, {"target": None}):
            panel = tui.render_result({"agent": "web", "answer": "hi",
                                       **target}, enabled=False)
            self.assertNotIn("target:", panel)


class TaskReportTest(unittest.TestCase):
    def test_reports_time_model_share_and_per_call_averages(self):
        report = tui.render_task_report(
            {"seconds": 84.2, "calls": 40, "tokens": 220,
             "model_seconds": 16.84, "retries": 3}, enabled=False)
        self.assertIn("task report", report)
        self.assertIn("84.2s", report)
        self.assertIn("model 16.8s (20%)", report)
        self.assertIn("40 calls", report)
        self.assertIn("220 tok", report)
        self.assertIn("avg 5.5 tok/call", report)
        self.assertIn("0.42s/call", report)
        self.assertIn("3 retries", report)
        self.assertEqual(len(report.splitlines()), 2)

    def test_zero_call_task_reports_a_single_line(self):
        report = tui.render_task_report({"seconds": 0.1, "calls": 0},
                                        enabled=False)
        self.assertIn("no model calls", report)
        self.assertEqual(len(report.splitlines()), 1)

    def test_clean_run_omits_the_retry_count(self):
        report = tui.render_task_report(
            {"seconds": 2.0, "calls": 4, "tokens": 12, "model_seconds": 1.0},
            enabled=False)
        self.assertNotIn("retries", report)

    def test_missing_keys_do_not_crash(self):
        report = tui.render_task_report({}, enabled=False)
        self.assertIn("0.0s", report)


class BannerAndPromptTest(unittest.TestCase):
    def test_banner_names_model_and_agent_count(self):
        banner = tui.build_banner("tinyllama", 3, enabled=False)
        self.assertIn("tinyllama", banner)
        self.assertIn("3 agents online", banner)
        self.assertEqual(banner.count(tui.NODE), 2)  # one node per hairline

    def test_banner_art_fits_the_panel(self):
        banner = tui.build_banner("m", 2, enabled=False, rng=random.Random(5))
        longest = max(len(line) for line in banner.splitlines())
        self.assertLessEqual(longest, tui.PANEL_WIDTH)

    def test_banner_shows_every_art_line(self):
        banner = tui.build_banner("m", 2, enabled=False)
        for line in tui.BANNER_ART + (tui.BANNER_SHADOW,):
            self.assertIn(line.lstrip(), banner)


class RandomBannerTest(unittest.TestCase):
    def test_same_seed_reproduces_the_banner(self):
        one = tui.build_banner("m", 2, enabled=True, rng=random.Random(7))
        two = tui.build_banner("m", 2, enabled=True, rng=random.Random(7))
        self.assertEqual(one, two)

    def test_trail_is_light_shades_hugging_the_glyph(self):
        trail = tui._motion_trail(random.Random(3), indent=2)
        self.assertEqual(len(trail), tui.TRAIL_WIDTH + 2)
        self.assertLessEqual(set(trail), set(tui.TRAIL_SHADES))
        self.assertNotEqual(trail[-1], " ")  # flush against the glyph

    def test_every_art_row_gets_a_trail(self):
        rows = tui._banner_rows(random.Random(9))
        self.assertEqual(len(rows), len(tui.BANNER_ART) + 1)
        for row, art in zip(rows, tui.BANNER_ART):
            self.assertTrue(row.endswith(art.lstrip()))
            trail = row[: len(row) - len(art.lstrip())]
            self.assertLessEqual(set(trail), set(tui.TRAIL_SHADES))

    def test_banner_colors_come_from_one_launch_ramp(self):
        banner = tui.build_banner("m", 2, enabled=True, rng=random.Random(1))
        codes = {int(code) for code in re.findall(r"38;5;(\d+)m", banner)}
        art_codes = codes - {tui.GRAY, tui.GRAY_LIGHT}
        self.assertTrue(
            any(art_codes <= set(ramp) for ramp in tui.BANNER_RAMPS))

    def test_prompt_shows_you_and_chevron_without_escapes(self):
        prompt = tui.prompt_text(enabled=False)
        self.assertIn("you", prompt)
        self.assertIn(tui.CHEVRON, prompt)
        self.assertNotIn("\x1b", prompt)


class SlashCommandTest(unittest.TestCase):
    def test_help_lists_commands(self):
        reply = tui.handle_command(make_state(enabled=False), "/help")
        for name in ("/help", "/agents", "/deep", "/model", "/setup",
                     "/quit"):
            self.assertIn(name, reply)

    def test_help_shows_the_voice_arguments(self):
        reply = tui.handle_command(make_state(enabled=False), "/help")
        self.assertIn("/voice [on|off|debug]", reply)

    def test_agents_lists_registered_specs(self):
        reply = tui.handle_command(make_state(enabled=False), "/agents")
        self.assertIn("casual", reply)
        self.assertIn("web", reply)
        self.assertIn("internet research", reply)

    def test_deep_sets_research_rounds(self):
        state = make_state(enabled=False)
        tui.handle_command(state, "/deep 5")
        self.assertEqual(state.services.max_research_rounds, 5)

    def test_deep_rejects_non_numeric(self):
        state = make_state(enabled=False)
        reply = tui.handle_command(state, "/deep lots")
        self.assertEqual(state.services.max_research_rounds, 3)
        self.assertIn("usage", reply)

    def test_model_rebuilds_policy_via_factory(self):
        state = make_state(enabled=False)
        old_policy = state.policy
        tui.handle_command(state, "/model tinyllama")
        self.assertEqual(state.model, "tinyllama")
        self.assertIsNot(state.policy, old_policy)
        self.assertEqual(state._built, ["tinyllama"])

    def test_quit_stops_the_loop(self):
        state = make_state(enabled=False)
        tui.handle_command(state, "/quit")
        self.assertFalse(state.running)

    def test_unknown_command_reports_itself(self):
        reply = tui.handle_command(make_state(enabled=False), "/bogus")
        self.assertIn("unknown", reply)

    def test_browser_http_swaps_to_plain_fetch(self):
        state = make_state(enabled=False)
        reply = tui.handle_command(state, "/browser http")
        self.assertIsNotNone(state.services.fetch_page)
        self.assertIn("http", reply)

    def test_browser_bad_mode_reports_usage(self):
        state = make_state(enabled=False)
        reply = tui.handle_command(state, "/browser on")
        self.assertIn("usage", reply)
        self.assertIsNone(state.services.fetch_page)

    def test_browser_plain_swaps_to_fetcher_and_tracks_it(self):
        state = make_state(enabled=False)
        fake = _RecordingFetcher()
        with mock.patch.object(tui, "_make_browser_fetcher",
                               return_value=fake) as maker:
            reply = tui.handle_command(state, "/browser plain")
        maker.assert_called_once_with("plain", False)
        self.assertIs(state.fetcher, fake)
        self.assertEqual(state.services.fetch_page, fake.fetch_page)
        self.assertIn("plain", reply)

    def test_browser_stealth_on_screen_passes_visible(self):
        state = make_state(enabled=False)
        fake = _RecordingFetcher()
        with mock.patch.object(tui, "_make_browser_fetcher",
                               return_value=fake) as maker:
            tui.handle_command(state, "/browser stealth on-screen")
        maker.assert_called_once_with("stealth", True)

    def test_browser_visibility_defaults_from_the_config(self):
        # [browser] visible = true must show the window without the user
        # having to repeat "on-screen" on every /browser switch.
        state = make_state(enabled=False)
        state.browser_visible = True
        fake = _RecordingFetcher()
        with mock.patch.object(tui, "_make_browser_fetcher",
                               return_value=fake) as maker:
            tui.handle_command(state, "/browser plain")
        maker.assert_called_once_with("plain", True)

    def test_browser_headless_overrides_a_visible_config(self):
        state = make_state(enabled=False)
        state.browser_visible = True
        fake = _RecordingFetcher()
        with mock.patch.object(tui, "_make_browser_fetcher",
                               return_value=fake) as maker:
            tui.handle_command(state, "/browser stealth headless")
        maker.assert_called_once_with("stealth", False)

    def test_browser_switch_closes_previous_fetcher(self):
        state = make_state(enabled=False)
        first, second = _RecordingFetcher(), _RecordingFetcher()
        with mock.patch.object(tui, "_make_browser_fetcher",
                               side_effect=[first, second]):
            tui.handle_command(state, "/browser plain")
            tui.handle_command(state, "/browser stealth")
        self.assertEqual(first.closed, 1)
        self.assertIs(state.fetcher, second)

    def test_browser_failure_reports_styled_message(self):
        state = make_state(enabled=False)
        with mock.patch.object(tui, "_make_browser_fetcher",
                               side_effect=RuntimeError("no chrome")):
            reply = tui.handle_command(state, "/browser stealth")
        self.assertIn("unavailable", reply)
        self.assertIsNone(state.fetcher)

    def test_setup_reruns_the_wizard_and_reports_the_path(self):
        state = make_state(enabled=False)
        saved = "/home/u/.config/threetoks/config.ini"
        with mock.patch("threetoks.onboard.run_onboarding",
                        return_value=saved) as wizard, \
                mock.patch("threetoks.config.find_config_path",
                           return_value=saved):
            reply = tui.handle_command(state, "/setup")
        wizard.assert_called_once_with()
        self.assertIn("saved", reply)
        self.assertIn(saved, reply)
        # /voice on re-reads the file, so it applies without a restart.
        self.assertIn("/voice on", reply)

    def test_setup_warns_when_a_cwd_config_shadows_the_saved_file(self):
        state = make_state(enabled=False)
        with mock.patch("threetoks.onboard.run_onboarding",
                        return_value="/home/u/.config/threetoks/config.ini"), \
                mock.patch("threetoks.config.find_config_path",
                           return_value="config.ini"):
            reply = tui.handle_command(state, "/setup")
        self.assertIn("precedence", reply)
        self.assertIn("config.ini", reply)

    def test_setup_cancel_saves_nothing_and_keeps_the_repl(self):
        state = make_state(enabled=False)
        for interrupt in (EOFError, KeyboardInterrupt):
            with mock.patch("threetoks.onboard.run_onboarding",
                            side_effect=interrupt):
                reply = tui.handle_command(state, "/setup")
            self.assertIn("cancelled", reply)
            self.assertTrue(state.running)


class InitialFetcherFallbackTest(unittest.TestCase):
    def _config(self, mode):
        from types import SimpleNamespace
        return SimpleNamespace(browser=SimpleNamespace(mode=mode,
                                                       visible=True))

    def test_failed_browser_start_reports_instead_of_silently_degrading(self):
        services = FakeServices()
        lines = []
        with mock.patch.object(tui, "_make_browser_fetcher",
                               side_effect=RuntimeError("no chromedriver")):
            fetcher = tui._initial_fetcher(self._config("stealth"), services,
                                           lines.append)
        self.assertIsNone(fetcher)
        self.assertIsNotNone(services.fetch_page)   # degraded to http
        [notice] = lines
        self.assertIn("stealth unavailable", notice)
        self.assertIn("no chromedriver", notice)

    def test_http_mode_stays_quiet(self):
        services = FakeServices()
        lines = []
        fetcher = tui._initial_fetcher(self._config("http"), services,
                                       lines.append)
        self.assertIsNone(fetcher)
        self.assertEqual(lines, [])


class FirstRunOnboardingTest(unittest.TestCase):
    """``main()`` runs the wizard exactly when no config file exists."""

    def _run_main(self, found, wizard_effect=None):
        state = make_state(enabled=False)
        with mock.patch("threetoks.config.find_config_path",
                        return_value=found), \
                mock.patch("threetoks.onboard.run_onboarding",
                           side_effect=wizard_effect) as wizard, \
                mock.patch.object(tui, "build_state",
                                  return_value=state) as builder, \
                mock.patch.object(tui, "repl"), \
                mock.patch("builtins.print"):
            tui.main()
        return wizard, builder

    def test_no_config_anywhere_triggers_onboarding(self):
        wizard, _ = self._run_main(found=None)
        wizard.assert_called_once_with()

    def test_existing_config_skips_onboarding(self):
        wizard, _ = self._run_main(found="config.ini")
        wizard.assert_not_called()

    def test_dry_stdin_skips_setup_and_boots_on_defaults(self):
        _, builder = self._run_main(found=None, wizard_effect=EOFError)
        builder.assert_called_once()

    def test_ctrl_c_during_setup_exits_without_booting(self):
        _, builder = self._run_main(found=None,
                                    wizard_effect=KeyboardInterrupt)
        builder.assert_not_called()


class ReplLoopTest(unittest.TestCase):
    def test_scripted_input_dispatches_then_quits(self):
        state = make_state(enabled=False)
        script = iter(["/deep 4", "/quit"])
        printed = []
        tui.repl(state, read_input=lambda _prompt: next(script),
                 sink=printed.append)
        self.assertEqual(state.services.max_research_rounds, 4)
        self.assertFalse(state.running)
        self.assertTrue(any("mesh offline" in line for line in printed))

    def test_ctrl_d_at_prompt_exits_cleanly(self):
        state = make_state(enabled=False)

        def raise_eof(_prompt):
            raise EOFError

        printed = []
        tui.repl(state, read_input=raise_eof, sink=printed.append)
        self.assertTrue(any("mesh offline" in line for line in printed))

    def test_run_task_streams_ticker_and_panel(self):
        state = make_state(enabled=False)
        events = [{"node": "menu", "value": "go", "valid": True,
                   "out_tokens": 2, "wall_s": 0.1}]

        class TracerStub:
            on_event = None

        class PolicyStub:
            tracer = TracerStub()

        def run(task, services, policy):
            for event in events:
                policy.tracer.on_event(event)
            return {"agent": "web", "answer": "done"}

        def route(task, specs, policy):
            return FakeSpecWithRun("web", "research", run)

        state.policy = PolicyStub()
        printed = []
        tui._run_task(state, "hi", route, printed.append)
        joined = "\n".join(printed)
        self.assertIn("go", joined)       # ticker line for the event
        self.assertIn("done", joined)     # answer in the panel
        self.assertIn("1 decisions", joined)


@dataclass
class FakeSpecWithRun:
    """A spec whose run function drives the ticker in the loop test."""
    name: str
    description: str
    run: object


class ModelSwapOptionalAgentsTest(unittest.TestCase):
    """Regression: /model must re-derive the capability agents."""

    def test_swap_to_vision_model_registers_look_and_back_out(self):
        state = make_state()
        state.services.capture_frame = lambda: "b64"
        tui.handle_command(state, "/model llava:7b")
        self.assertIn("look", [spec.name for spec in state.specs])
        tui.handle_command(state, "/model qwen2.5:1.5b-instruct")
        self.assertNotIn("look", [spec.name for spec in state.specs])

    def test_light_survives_model_swaps_when_relay_is_wired(self):
        state = make_state()
        state.services.relay = object()
        tui.handle_command(state, "/model qwen2.5:1.5b-instruct")
        self.assertIn("light", [spec.name for spec in state.specs])

    def test_without_capabilities_specs_stay_untouched(self):
        state = make_state()
        before = list(state.specs)
        tui.handle_command(state, "/model qwen2.5:1.5b-instruct")
        self.assertEqual(state.specs, before)

    def test_family_notice_only_applies_to_the_ollama_provider(self):
        raw = make_state()
        self.assertIn("unknown template family",
                      tui.handle_command(raw, "/model gpt-4o-mini"))
        chat = make_state()
        chat.llm_provider = "openrouter"
        self.assertNotIn("unknown template family",
                         tui.handle_command(chat, "/model gpt-4o-mini"))


if __name__ == "__main__":
    unittest.main()
