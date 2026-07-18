"""Offline tests for the onboarding wizard (scripted console, no model).

The wizard's ``ask``/``sink`` hooks are fed from lists, so every path —
Enter-through defaults, overrides, re-asks on invalid input, conditional
questions, and the save-to-user-file flow — runs without a terminal.
"""
import os
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from threetoks import config, onboard
from threetoks.config import CONFIG_ENV_VAR, ThreetoksConfig, load_config
from threetoks.onboard import run_onboarding, run_wizard

_PIPER_SIDECAR = onboard._PIPER_SIDECAR_SUFFIX  # ".json", beside the .onnx


def _scripted(answers):
    """An ``ask`` that replays ``answers`` and records the prompts."""
    prompts = []
    iterator = iter(answers)

    def ask(prompt):
        prompts.append(prompt)
        return next(iterator)
    return ask, prompts


def _quiet(line):
    return None


def _recorder():
    """A fake ``open_url`` recording URLs instead of launching a browser."""
    opened = []

    def open_url(url):
        opened.append(url)
        return True
    return open_url, opened


def _no_browser(url):
    """A fake ``open_url`` that never launches a browser."""
    return True


class WizardDefaultsTest(unittest.TestCase):
    def test_enter_through_keeps_every_default(self):
        ask, prompts = _scripted([""] * 9)
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        base = ThreetoksConfig()
        self.assertEqual(cfg.llm, base.llm)
        self.assertEqual(cfg.browser, base.browser)
        self.assertEqual(cfg.search, base.search)
        self.assertEqual(cfg.research, base.research)
        self.assertEqual(cfg.files, base.files)
        self.assertTrue(cfg.memory.enabled)
        self.assertEqual(cfg.voice, base.voice)  # declined, unchanged
        self.assertEqual(len(prompts), 9)

    def test_prompts_show_the_defaults(self):
        ask, prompts = _scripted([""] * 9)
        run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertIn(f"[{config.DEFAULT_MODEL}]", prompts[0])
        self.assertIn(f"[{config.DEFAULT_LLM_HOST}]", prompts[1])
        self.assertIn("http/plain/stealth", prompts[2])

    def test_base_config_supplies_the_defaults(self):
        base = ThreetoksConfig(llm=config.LlmConfig(model="tuned", vote_k=5))
        ask, _ = _scripted([""] * 9)
        cfg = run_wizard(base, ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertEqual(cfg.llm.model, "tuned")
        self.assertEqual(cfg.llm.vote_k, 5)  # carried over, never asked

    def test_memory_default_moves_off_the_cwd(self):
        ask, _ = _scripted([""] * 9)
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertNotEqual(cfg.memory.path, config.DEFAULT_MEMORY_PATH)
        self.assertIn("threetoks", cfg.memory.path)
        self.assertTrue(cfg.memory.path.endswith("memory.json"))

    def test_custom_memory_path_is_kept_as_suggestion(self):
        base = ThreetoksConfig(memory=config.MemoryConfig(path="/tuned.json"))
        ask, prompts = _scripted([""] * 9)
        cfg = run_wizard(base, ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertEqual(cfg.memory.path, "/tuned.json")
        self.assertIn("[/tuned.json]", prompts[-2])  # memory-file, before voice


class WizardAnswersTest(unittest.TestCase):
    def test_full_answers_override_every_asked_field(self):
        ask, _ = _scripted(["my-model", "http://h:9", "stealth", "y",
                            "http://s:1", "7", "/data", "yes", "/mem.json",
                            "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertEqual(cfg.llm.model, "my-model")
        self.assertEqual(cfg.llm.host, "http://h:9")
        self.assertEqual(cfg.browser.mode, "stealth")
        self.assertTrue(cfg.browser.visible)
        self.assertEqual(cfg.search.searxng_url, "http://s:1")
        self.assertEqual(cfg.research.max_rounds, 7)
        self.assertEqual(cfg.files.root, "/data")
        self.assertTrue(cfg.memory.enabled)
        self.assertEqual(cfg.memory.path, "/mem.json")

    def test_visible_question_skipped_in_http_mode(self):
        ask, prompts = _scripted(["", "", "http", "", "", "", "y", "", "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertFalse(cfg.browser.visible)
        self.assertFalse(any("window" in prompt for prompt in prompts))

    def test_memory_path_question_skipped_when_disabled(self):
        ask, prompts = _scripted(["", "", "", "", "", "", "n", "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertFalse(cfg.memory.enabled)
        self.assertFalse(any("Memory file" in prompt for prompt in prompts))
        self.assertEqual(len(prompts), 8)

    def test_invalid_browser_mode_reasks(self):
        ask, _ = _scripted(["", "", "teleport", "PLAIN", "n",
                            "", "", "", "", "", "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertEqual(cfg.browser.mode, "plain")

    def test_invalid_round_count_reasks(self):
        ask, _ = _scripted(["", "", "", "", "many", "-2", "0", "4", "", "",
                            "", "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertEqual(cfg.research.max_rounds, 4)

    def test_bool_word_variants_parse(self):
        for word, expected in (("on", True), ("TRUE", True), ("1", True),
                               ("off", False), ("0", False), ("No", False)):
            answers = ["", "", "", "", "", "", word]
            if expected:
                answers.append("")  # memory-file question
            answers.append("")      # voice lead question, declined
            ask, _ = _scripted(answers)
            cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
            self.assertEqual(cfg.memory.enabled, expected, word)

    def test_garbled_bool_reasks(self):
        ask, _ = _scripted(["", "", "", "", "", "", "maybe", "nope", "no",
                            "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertFalse(cfg.memory.enabled)


class RunOnboardingTest(unittest.TestCase):
    def test_saves_to_the_target_and_loads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "config.ini")
            lines = []
            ask, _ = _scripted(["wiz-model", "", "plain", "n", "", "2", "",
                                "y", "/mem.json", "n"])
            with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: target}):
                written = run_onboarding(ask=ask, sink=lines.append,
                                         open_url=_no_browser)
                self.assertEqual(written, target)
                cfg = load_config()
        self.assertEqual(cfg.llm.model, "wiz-model")
        self.assertEqual(cfg.browser.mode, "plain")
        self.assertEqual(cfg.research.max_rounds, 2)
        self.assertEqual(cfg.memory.path, "/mem.json")
        self.assertTrue(any(target in line for line in lines))

    def test_enabling_voice_persists_to_the_saved_ini(self):
        open_url, opened = _recorder()
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "config.ini")
            # ...voice yes, lang, open pages, stt path, keep it, tts
            # path, keep it (both paths are missing on disk).
            ask, _ = _scripted(["", "", "http", "", "", "", "n",
                                "y", "es", "y", "/stt", "n", "/v.onnx", "n"])
            with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: target}):
                run_onboarding(ask=ask, sink=_quiet, open_url=open_url)
                cfg = load_config()
        self.assertIs(cfg.voice.enabled, True)
        self.assertEqual(cfg.voice.lang, "es")
        self.assertEqual(cfg.voice.stt_model_path, "/stt")
        self.assertEqual(cfg.voice.tts_model_path, "/v.onnx")
        self.assertEqual(opened, [onboard.VOSK_MODELS_URL,
                                  onboard.PIPER_VOICES_URL])


class UnaskedSectionsSurviveTest(unittest.TestCase):
    """Regression: /setup must never wipe sections it does not ask about."""

    def test_enter_through_keeps_every_unasked_field(self):
        from threetoks.config import (CameraConfig, CodeConfig, LlmConfig,
                                     RelayConfig, VoiceConfig)
        edited = ThreetoksConfig(
            llm=LlmConfig(provider="anthropic", api_base="http://p"),
            code=CodeConfig(retrieval=True, snippet_cache="/s.json"),
            voice=VoiceConfig(enabled=True, tts_model_path="/v.onnx",
                              tts_speaker=3, post_speak_delay=1.5),
            camera=CameraConfig(index=2),
            relay=RelayConfig(pin=27))
        ask, _ = _scripted(["", "", "http", "", "", "", "n", "n"])
        rerun = run_wizard(edited, ask=ask, sink=_quiet, open_url=_no_browser)
        self.assertEqual(rerun.llm.provider, "anthropic")
        self.assertEqual(rerun.llm.api_base, "http://p")
        self.assertEqual(rerun.code, edited.code)
        # Declining voice keeps every base field but flips enabled off.
        self.assertEqual(rerun.voice, replace(edited.voice, enabled=False))
        self.assertEqual(rerun.camera, edited.camera)
        self.assertEqual(rerun.relay, edited.relay)


class VoiceStepTest(unittest.TestCase):
    """The optional voice step: decline, accept, defaults, and verification."""

    def test_declining_voice_asks_nothing_more(self):
        open_url, opened = _recorder()
        ask, prompts = _scripted(["", "", "http", "", "", "", "n", "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=open_url)
        self.assertFalse(cfg.voice.enabled)
        self.assertEqual(cfg.voice, config.VoiceConfig())  # base carried
        self.assertFalse(any("language" in prompt for prompt in prompts))
        self.assertEqual(opened, [])

    def test_accepting_voice_asks_paths_and_opens_pages(self):
        open_url, opened = _recorder()
        ask, prompts = _scripted(["", "", "http", "", "", "", "n", "y", "fr",
                                  "y", "/models/stt", "n", "/voices/v.onnx",
                                  "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=open_url)
        self.assertTrue(cfg.voice.enabled)
        self.assertEqual(cfg.voice.lang, "fr")
        self.assertEqual(cfg.voice.stt_model_path, "/models/stt")
        self.assertEqual(cfg.voice.tts_model_path, "/voices/v.onnx")
        self.assertTrue(any("language" in prompt for prompt in prompts))
        self.assertEqual(opened, [onboard.VOSK_MODELS_URL,
                                  onboard.PIPER_VOICES_URL])

    def test_yes_path_enter_through_keeps_base_defaults(self):
        open_url, opened = _recorder()
        base = ThreetoksConfig(voice=config.VoiceConfig(
            lang="de", stt_model_path="/base/stt",
            tts_model_path="/base/v.onnx"))
        # Enter keeps each base path; both are missing, so each warns and
        # the retry is declined to keep the configured value.
        ask, _ = _scripted(["", "", "http", "", "", "", "n", "y", "", "y",
                            "", "n", "", "n"])
        cfg = run_wizard(base, ask=ask, sink=_quiet, open_url=open_url)
        self.assertTrue(cfg.voice.enabled)
        self.assertEqual(cfg.voice.lang, "de")
        self.assertEqual(cfg.voice.stt_model_path, "/base/stt")
        self.assertEqual(cfg.voice.tts_model_path, "/base/v.onnx")
        self.assertEqual(opened, [onboard.VOSK_MODELS_URL,
                                  onboard.PIPER_VOICES_URL])

    def test_declining_the_browser_opens_nothing_but_still_prints_links(self):
        open_url, opened = _recorder()
        lines = []
        # ...voice yes, lang, open pages NO, both paths left empty.
        ask, _ = _scripted(["", "", "http", "", "", "", "n", "y", "", "n",
                            "", ""])
        run_wizard(ask=ask, sink=lines.append, open_url=open_url)
        self.assertEqual(opened, [])
        self.assertTrue(any(onboard.VOSK_MODELS_URL in line
                            for line in lines))
        self.assertTrue(any(onboard.PIPER_VOICES_URL in line
                            for line in lines))

    def test_guidance_explains_the_two_piper_files_before_the_paths(self):
        lines = []
        ask, prompts = _scripted(["", "", "http", "", "", "", "n", "y", "",
                                  "n", "", ""])
        run_wizard(ask=ask, sink=lines.append, open_url=_no_browser)
        said = " ".join(lines)
        self.assertIn(".onnx.json", said)  # the sidecar is spelled out
        self.assertIn("auto-download", said)  # vosk needs no download
        self.assertTrue(any("/voice on" in line for line in lines))
        self.assertTrue(any("Piper voice" in prompt for prompt in prompts))

    def test_retrying_a_failing_piper_path_ends_on_the_good_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "voice.onnx")
            for suffix in ("", _PIPER_SIDECAR):  # .onnx plus its sidecar
                with open(good + suffix, "w", encoding="utf-8") as handle:
                    handle.write("x")
            lines = []
            # ...piper path missing, retry yes, second path missing too,
            # retry yes again, then the good pair.
            ask, _ = _scripted(["", "", "http", "", "", "", "n", "y", "",
                                "n", "", "/nope.onnx", "y", "/still-no.onnx",
                                "y", good])
            cfg = run_wizard(ask=ask, sink=lines.append,
                             open_url=_no_browser)
        self.assertEqual(cfg.voice.tts_model_path, good)
        warnings = [line for line in lines if "not found" in line]
        self.assertEqual(len(warnings), 2)  # each failing path re-warns

    def test_a_sidecarless_piper_voice_warns_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            lonely = os.path.join(tmp, "lonely.onnx")  # .onnx, no sidecar
            with open(lonely, "w", encoding="utf-8") as handle:
                handle.write("x")
            lines = []
            ask, _ = _scripted(["", "", "http", "", "", "", "n", "y", "",
                                "n", "", lonely, "n"])
            cfg = run_wizard(ask=ask, sink=lines.append,
                             open_url=_no_browser)
        self.assertEqual(cfg.voice.tts_model_path, lonely)  # kept anyway
        self.assertTrue(any("sidecar" in line for line in lines))

    def test_enter_through_a_stale_failing_path_never_loops(self):
        # Regression: [voice] enabled with a path that has since moved is
        # the /setup re-run case. Enter must keep working the wizard's way
        # ("Enter keeps the default") instead of re-asking forever.
        base = ThreetoksConfig(voice=config.VoiceConfig(
            enabled=True, stt_model_path="/gone/stt",
            tts_model_path="/gone/v.onnx"))
        asked = []

        def enter_only(prompt):
            asked.append(prompt)
            if len(asked) > 40:
                raise AssertionError("wizard trapped the user in a loop")
            return ""

        cfg = run_wizard(base, ask=enter_only, sink=_quiet,
                         open_url=_no_browser)
        self.assertTrue(cfg.voice.enabled)
        self.assertEqual(cfg.voice.stt_model_path, "/gone/stt")  # kept
        self.assertEqual(cfg.voice.tts_model_path, "/gone/v.onnx")

    def test_a_kept_failing_path_is_warned_about_exactly_once(self):
        # Regression: the closing check used to re-run the path checks the
        # ask loop had just done, printing every warning twice.
        lines = []
        ask, _ = _scripted(["", "", "http", "", "", "", "n", "y", "", "n",
                            "/missing/stt", "n", "/missing/v.onnx", "n"])
        run_wizard(ask=ask, sink=lines.append, open_url=_no_browser)
        for path in ("/missing/stt", "/missing/v.onnx"):
            hits = [line for line in lines if path in line]
            self.assertEqual(len(hits), 1, hits)

    def test_declining_the_retry_keeps_the_failing_path(self):
        ask, _ = _scripted(["", "", "http", "", "", "", "n", "y", "", "n",
                            "/missing/stt", "n", "/missing/v.onnx", "n"])
        cfg = run_wizard(ask=ask, sink=_quiet, open_url=_no_browser)
        # The user is never trapped: a failing path they insist on sticks.
        self.assertEqual(cfg.voice.stt_model_path, "/missing/stt")
        self.assertEqual(cfg.voice.tts_model_path, "/missing/v.onnx")

    def test_an_empty_path_skips_verification_entirely(self):
        lines = []
        ask, prompts = _scripted(["", "", "http", "", "", "", "n", "y", "",
                                  "n", "", ""])
        cfg = run_wizard(ask=ask, sink=lines.append, open_url=_no_browser)
        self.assertEqual(cfg.voice.stt_model_path, "")
        self.assertFalse(any("Try a different path" in prompt
                             for prompt in prompts))
        self.assertFalse(any("not found" in line for line in lines))

    def test_piper_problem_walks_missing_file_then_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            onnx = os.path.join(tmp, "voice.onnx")
            self.assertIn("not found", onboard._piper_voice_problem(onnx))
            with open(onnx, "w", encoding="utf-8") as handle:
                handle.write("x")
            # The .onnx alone is the classic half-done piper download.
            self.assertIn("sidecar", onboard._piper_voice_problem(onnx))
            with open(onnx + _PIPER_SIDECAR, "w", encoding="utf-8") as handle:
                handle.write("{}")
            self.assertIsNone(onboard._piper_voice_problem(onnx))

    def test_vosk_problem_is_about_the_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(onboard._vosk_folder_problem(tmp))
        self.assertIn("not found",
                      onboard._vosk_folder_problem("/gone/model"))

    def test_verify_voice_notes_an_empty_piper_path(self):
        lines = []
        onboard._verify_voice(config.VoiceConfig(enabled=True), lines.append)
        self.assertTrue(any("stays off" in line for line in lines), lines)

    def test_verify_voice_leaves_the_path_checks_to_the_ask_loop(self):
        lines = []
        onboard._verify_voice(
            config.VoiceConfig(enabled=True, stt_model_path="/missing/stt",
                               tts_model_path="/missing/v.onnx"),
            lines.append)
        self.assertFalse(any("not found" in line for line in lines), lines)


if __name__ == "__main__":
    unittest.main()
