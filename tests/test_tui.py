"""Offline tests for the TUI: no LLM, no network.

Covers NO_COLOR stripping, slash command dispatch with injected specs and
a scripted input function, the task run streaming its progress tree, and
result-panel rendering with missing optional keys.
"""
import unittest
from dataclasses import dataclass, replace
from unittest import mock

from threetoks import session, theme, tui
from threetoks.config import BrowserConfig, LlmConfig
from threetoks.trace import Tracer

_NO_TTY = mock.patch("threetoks.keys.is_interactive", return_value=False)


def setUpModule():
    """Run every test as if piped, so none opens the real config screen."""
    _NO_TTY.start()


def tearDownModule():
    _NO_TTY.stop()


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
        self.blocked_hosts = set()


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
        model="qwen3.5:2b",
        policy_factory=factory,
        enabled=overrides.get("enabled", True),
    )
    state._built = built  # expose for assertions
    return state


class NoColorTest(unittest.TestCase):
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
            self.assertFalse(tui.color_enabled())
        finally:
            del os.environ["NO_COLOR"]


class GradientTest(unittest.TestCase):
    def test_disabled_gradient_returns_text_unchanged(self):
        self.assertEqual(tui.gradient("ThreeToks", enabled=False), "ThreeToks")

    def test_enabled_gradient_spans_the_ramp(self):
        colored = theme.gradient("ThreeToks", enabled=True)
        self.assertIn(f"38;5;{theme.RAMP[0]}m", colored)   # starts cyan
        self.assertIn(f"38;5;{theme.RAMP[-1]}m", colored)  # ends pink

    def test_hairline_is_panel_width_with_one_node(self):
        line = theme.hairline(6, enabled=False)
        self.assertEqual(len(line), theme.PANEL_WIDTH)
        self.assertEqual(line.count(theme.NODE), 1)


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


class PromptTest(unittest.TestCase):
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

    def test_help_lists_debug_apart_from_voice_debug(self):
        # Two unrelated features; /help must not blur them together.
        reply = tui.handle_command(make_state(enabled=False), "/help")
        self.assertIn("/debug [on|off|full]", reply)
        self.assertIn("raw model answer", reply)
        self.assertIn("/voice [on|off|debug]", reply)

    def test_debug_bare_reports_the_current_state(self):
        state = make_state(enabled=False)
        self.assertIn("debug is off", tui.handle_command(state, "/debug"))
        state.debug = True
        self.assertIn("debug is on", tui.handle_command(state, "/debug"))

    def test_debug_on_and_off_set_the_flag(self):
        state = make_state(enabled=False)
        self.assertFalse(state.debug)  # off by default
        tui.handle_command(state, "/debug on")
        self.assertTrue(state.debug)
        tui.handle_command(state, "/debug off")
        self.assertFalse(state.debug)

    def test_debug_garbage_argument_reports_usage(self):
        state = make_state(enabled=False)
        reply = tui.handle_command(state, "/debug loud")
        self.assertIn("usage: /debug [on|off|full]", reply)
        self.assertFalse(state.debug)

    def test_debug_full_sets_both_flags(self):
        state = make_state(enabled=False)
        reply = tui.handle_command(state, "/debug full")
        self.assertTrue(state.debug)
        self.assertTrue(state.debug_full)
        self.assertIn("complete prompt", reply)

    def test_debug_on_clears_full(self):
        state = make_state(enabled=False)
        tui.handle_command(state, "/debug full")
        tui.handle_command(state, "/debug on")
        self.assertTrue(state.debug)
        self.assertFalse(state.debug_full)

    def test_debug_off_clears_full(self):
        state = make_state(enabled=False)
        tui.handle_command(state, "/debug full")
        tui.handle_command(state, "/debug off")
        self.assertFalse(state.debug)
        self.assertFalse(state.debug_full)

    def test_debug_bare_reports_full_state(self):
        state = make_state(enabled=False)
        tui.handle_command(state, "/debug full")
        self.assertIn("debug is full", tui.handle_command(state, "/debug"))

    def test_debug_is_independent_of_voice_debug(self):
        state = make_state(enabled=False)
        tui.handle_command(state, "/debug on")
        reply = tui.handle_command(state, "/voice debug")
        self.assertIn("voice is off", reply)  # no live session to trace
        self.assertTrue(state.debug)          # ... and /debug stays on

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
        with mock.patch.object(session, "make_browser_fetcher",
                               return_value=fake) as maker:
            reply = tui.handle_command(state, "/browser plain")
        maker.assert_called_once_with(False)
        self.assertIs(state.fetcher, fake)
        self.assertEqual(state.services.fetch_page, fake.fetch_page)
        self.assertIn("plain", reply)

    def test_browser_on_screen_passes_visible(self):
        state = make_state(enabled=False)
        fake = _RecordingFetcher()
        with mock.patch.object(session, "make_browser_fetcher",
                               return_value=fake) as maker:
            tui.handle_command(state, "/browser plain on-screen")
        maker.assert_called_once_with(True)

    def test_browser_switch_gives_blocked_sites_a_fresh_chance(self):
        state = make_state(enabled=False)
        state.services.blocked_hosts.add("walled.example")
        with mock.patch.object(session, "make_browser_fetcher",
                               return_value=_RecordingFetcher()):
            tui.handle_command(state, "/browser plain on-screen")
        self.assertEqual(state.services.blocked_hosts, set())

    def test_browser_visibility_defaults_from_the_config(self):
        # [browser] visible = true must show the window without the user
        # having to repeat "on-screen" on every /browser switch.
        state = make_state(enabled=False)
        state.config = replace(state.config, browser=BrowserConfig(
            visible=True))
        fake = _RecordingFetcher()
        with mock.patch.object(session, "make_browser_fetcher",
                               return_value=fake) as maker:
            tui.handle_command(state, "/browser plain")
        maker.assert_called_once_with(True)

    def test_browser_headless_overrides_a_visible_config(self):
        state = make_state(enabled=False)
        state.config = replace(state.config, browser=BrowserConfig(
            visible=True))
        fake = _RecordingFetcher()
        with mock.patch.object(session, "make_browser_fetcher",
                               return_value=fake) as maker:
            tui.handle_command(state, "/browser plain headless")
        maker.assert_called_once_with(False)

    def test_browser_switch_closes_previous_fetcher(self):
        state = make_state(enabled=False)
        first, second = _RecordingFetcher(), _RecordingFetcher()
        with mock.patch.object(session, "make_browser_fetcher",
                               side_effect=[first, second]):
            tui.handle_command(state, "/browser plain")
            tui.handle_command(state, "/browser plain headless")
        self.assertEqual(first.closed, 1)
        self.assertIs(state.fetcher, second)

    def test_browser_failure_reports_styled_message(self):
        state = make_state(enabled=False)
        with mock.patch.object(session, "make_browser_fetcher",
                               side_effect=RuntimeError("no chrome")):
            reply = tui.handle_command(state, "/browser plain")
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
        with mock.patch.object(session, "_searxng_gate"), \
                mock.patch.object(session, "make_browser_fetcher",
                                  side_effect=RuntimeError("no chromedriver")):
            fetcher = session.initial_fetcher(self._config("plain"), services,
                                              lines.append)
        self.assertIsNone(fetcher)
        self.assertIsNotNone(services.fetch_page)   # degraded to http
        [notice] = lines
        self.assertIn("plain unavailable", notice)
        self.assertIn("no chromedriver", notice)

    def test_http_mode_stays_quiet(self):
        services = FakeServices()
        lines = []
        fetcher = session.initial_fetcher(self._config("http"), services,
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

    def test_run_task_streams_a_tree_and_panel(self):
        state = make_state(enabled=False)
        events = [{"node": "menu", "value": "go", "valid": True,
                   "out_tokens": 2, "wall_s": 0.1}]

        class PolicyStub:
            tracer = Tracer(None)

        def run(task, services, policy):
            for event in events:
                policy.tracer.record(event)
            return {"agent": "web", "answer": "done"}

        def route(task, specs, policy):
            return FakeSpecWithRun("web", "research", run)

        state.policy = PolicyStub()
        printed = []
        tui._run_task(state, "hi", route, printed.append)
        self.assertEqual(printed[0], "  ◆ pick an agent")
        self.assertTrue(printed[1].startswith("  ╰ web · 0 calls · 0t"))
        self.assertEqual(printed[2], "  ◆ web agent")
        self.assertIn("go", printed[3])          # the decision row
        self.assertTrue(printed[4].startswith("  ╰ answered · 1 call"))
        self.assertIn("done", printed[5])        # answer in the panel
        self.assertIn("1 decisions", printed[5])

    def test_run_task_ends_the_branch_when_the_agent_fails(self):
        state = make_state(enabled=False)

        class PolicyStub:
            tracer = Tracer(None)

        def run(task, services, policy):
            raise RuntimeError("ollama down")

        state.policy = PolicyStub()
        printed = []
        with self.assertRaises(RuntimeError):
            tui._run_task(state, "hi",
                          lambda *_: FakeSpecWithRun("web", "r", run),
                          printed.append)
        self.assertTrue(printed[-1].startswith("  ╰ stopped"), printed)


@dataclass
class FakeSpecWithRun:
    """A spec whose run function drives the progress tree in the loop test."""
    name: str
    description: str
    run: object


class StartupFailureTest(unittest.TestCase):
    """A misconfigured provider must read as a message, not a traceback."""

    def _main_with_build_error(self, error):
        shown = []
        with mock.patch("threetoks.config.find_config_path",
                        return_value="config.ini"), \
                mock.patch.object(tui, "build_state", side_effect=error), \
                mock.patch.object(tui, "repl") as repl, \
                mock.patch("builtins.print", side_effect=shown.append):
            tui.main()  # must not raise
        return "\n".join(shown), repl

    def test_missing_api_key_prints_a_panel_and_skips_the_repl(self):
        error = RuntimeError("provider 'openrouter' needs the "
                             "OPENROUTER_API_KEY environment variable")
        out, repl = self._main_with_build_error(error)
        self.assertIn("OPENROUTER_API_KEY", out)
        self.assertIn("export the key", out)  # actionable hint
        repl.assert_not_called()

    def test_api_key_hint_names_the_ollama_escape_hatch(self):
        hint = tui._error_hint(RuntimeError("needs the OPENAI_API_KEY"))
        self.assertIn("provider = ollama", hint)


class BadConfigWarningTest(unittest.TestCase):
    """A config that failed to parse must be announced, not swallowed."""

    def test_parse_failure_is_printed_at_startup(self):
        shown = []
        with mock.patch("threetoks.fields.config_warnings",
                        return_value=["⚠ /cfg.ini was ignored — fell back"]):
            session.warn_bad_config(shown.append)
        self.assertEqual(len(shown), 1)
        self.assertIn("was ignored", shown[0])

    def test_healthy_config_prints_nothing(self):
        shown = []
        with mock.patch("threetoks.fields.config_warnings", return_value=[]):
            session.warn_bad_config(shown.append)
        self.assertEqual(shown, [])


class ModelSwapOptionalAgentsTest(unittest.TestCase):
    """Regression: /model must re-derive the capability agents."""

    def test_swap_to_vision_model_registers_look_and_back_out(self):
        state = make_state()
        state.services.capture_frame = lambda: "b64"
        tui.handle_command(state, "/model llava:7b")
        self.assertIn("look", [spec.name for spec in state.specs])
        tui.handle_command(state, "/model qwen3.5:2b")
        self.assertNotIn("look", [spec.name for spec in state.specs])

    def test_light_survives_model_swaps_when_relay_is_wired(self):
        state = make_state()
        state.services.relay = object()
        tui.handle_command(state, "/model qwen3.5:2b")
        self.assertIn("light", [spec.name for spec in state.specs])

    def test_without_capabilities_only_the_planner_is_added(self):
        state = make_state()
        before = list(state.specs)
        tui.handle_command(state, "/model qwen3.5:2b")
        self.assertEqual(state.specs[:-1], before)
        self.assertEqual(state.specs[-1].name, "planner")

    def test_planner_plans_only_with_agents_ready_after_a_swap(self):
        state = make_state()
        state.services.capture_frame = lambda: "b64"
        tui.handle_command(state, "/model llava:7b")
        tui.handle_command(state, "/model qwen3.5:2b")
        names = [spec.name for spec in state.specs]
        self.assertEqual(names.count("planner"), 1)
        planned = [spec.name for spec in state.specs[-1].run.specs]
        self.assertEqual(planned, ["casual", "web"])  # look left, no planner

    def test_family_notice_only_applies_to_the_ollama_provider(self):
        raw = make_state()
        self.assertIn("unknown template family",
                      tui.handle_command(raw, "/model gpt-4o-mini"))
        chat = make_state()
        chat.config = replace(chat.config, llm=LlmConfig(
            provider="openrouter"))
        self.assertNotIn("unknown template family",
                         tui.handle_command(chat, "/model gpt-4o-mini"))


class ConfirmEditTest(unittest.TestCase):
    """The code agent's edit confirm: only an explicit yes applies."""

    DIFF = "--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-x = 1\n+x = 2"

    def _confirm(self, reader):
        printed, prompts = [], []

        def read_input(prompt):
            prompts.append(prompt)
            return reader()

        confirm = tui.make_confirm_edit(make_state(enabled=False),
                                        read_input, printed.append)
        return confirm("m.py", self.DIFF), printed, prompts

    def test_yes_applies_and_the_diff_is_shown_first(self):
        answer, printed, prompts = self._confirm(lambda: " Y ")
        self.assertTrue(answer)
        self.assertIn("+x = 2", printed[0])
        self.assertIn("apply this edit to m.py? [y/N]", prompts[0])

    def test_empty_reply_declines(self):
        self.assertFalse(self._confirm(lambda: "")[0])

    def test_eof_and_ctrl_c_decline(self):
        for error in (EOFError, KeyboardInterrupt):
            def raise_it(error=error):
                raise error
            self.assertFalse(self._confirm(raise_it)[0], error)

    def test_long_diff_is_clipped_with_a_count(self):
        diff = "\n".join(f"+line {n}" for n in
                         range(tui.CONFIRM_DIFF_MAX_LINES + 3))
        shown = []
        confirm = tui.make_confirm_edit(make_state(enabled=False),
                                        lambda prompt: "n", shown.append)
        self.assertFalse(confirm("m.py", diff))
        self.assertIn("3 more lines", shown[0])
        self.assertNotIn(f"+line {tui.CONFIRM_DIFF_MAX_LINES}", shown[0])

    def test_repl_confirm_reads_from_the_repl_input(self):
        state = make_state(enabled=False)
        replies = iter(["/quit", "yes"])
        tui.repl(state, read_input=lambda prompt: next(replies),
                 sink=[].append)
        self.assertTrue(state.services.confirm_edit("m.py", self.DIFF))


class ConfirmCommandTest(unittest.TestCase):
    """The shell-mode confirm: the command is shown, only a yes runs it."""

    def _confirm(self, reply):
        printed, prompts = [], []
        state = make_state(enabled=False)
        state.services.files_root = "/proj"
        confirm = tui.make_confirm_command(
            state, lambda prompt: prompts.append(prompt) or reply,
            printed.append)
        return confirm("rm -rf build"), printed, prompts

    def test_yes_runs_and_the_command_is_shown_first(self):
        answer, printed, prompts = self._confirm("y")
        self.assertTrue(answer)
        self.assertIn("$ rm -rf build", printed[0])
        self.assertIn("in /proj? [y/N]", prompts[0])

    def test_anything_else_declines(self):
        for reply in ("", "n", "sure", None):
            self.assertFalse(self._confirm(reply)[0], reply)

    def test_repl_installs_the_command_confirm(self):
        state = make_state(enabled=False)
        state.services.files_root = "/proj"
        replies = iter(["/quit", "yes"])
        tui.repl(state, read_input=lambda prompt: next(replies),
                 sink=[].append)
        self.assertTrue(state.services.confirm_command("ls"))


class ConfigScreenWiringTest(unittest.TestCase):
    """/setup and the first start open the screen only on a real terminal."""

    def test_setup_opens_the_screen_on_a_terminal(self):
        state = make_state(enabled=False)
        with mock.patch("threetoks.keys.is_interactive", return_value=True), \
                mock.patch("threetoks.config_ui.config_command",
                           return_value="screen reply") as screen, \
                mock.patch("threetoks.onboard.run_onboarding") as wizard:
            self.assertEqual(tui.handle_command(state, "/config"),
                             "screen reply")
        screen.assert_called_once_with(state)
        wizard.assert_not_called()

    def test_first_start_on_a_terminal_opens_the_screen_not_the_wizard(self):
        with mock.patch("threetoks.config_ui.is_interactive",
                        return_value=True), \
                mock.patch("threetoks.config_ui.first_run") as screen, \
                mock.patch("threetoks.onboard.run_onboarding") as wizard:
            from threetoks import config_ui
            self.assertTrue(config_ui.first_start())
        screen.assert_called_once()
        wizard.assert_not_called()

    def test_reader_is_plain_input_without_a_terminal(self):
        from threetoks import line_input
        with mock.patch("threetoks.keys.is_interactive", return_value=False):
            self.assertIs(line_input.terminal_reader(tui._COMMANDS), input)

    def test_closing_without_a_save_changes_nothing(self):
        from threetoks import config_ui
        state = make_state(enabled=False)
        with mock.patch.object(config_ui, "run_screen",
                               return_value=None):
            self.assertIn("nothing changed",
                          config_ui.config_command(state))


if __name__ == "__main__":
    unittest.main()
