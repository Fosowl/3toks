"""Offline tests for the onboarding wizard (scripted console, no model).

The wizard's ``ask``/``sink`` hooks are fed from lists, so every path —
Enter-through defaults, overrides, re-asks on invalid input, conditional
questions, and the save-to-user-file flow — runs without a terminal.
"""
import os
import tempfile
import unittest
from unittest import mock

from threetoks import config
from threetoks.config import CONFIG_ENV_VAR, ThreetoksConfig, load_config
from threetoks.onboard import run_onboarding, run_wizard


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


class WizardDefaultsTest(unittest.TestCase):
    def test_enter_through_keeps_every_default(self):
        ask, prompts = _scripted([""] * 8)
        cfg = run_wizard(ask=ask, sink=_quiet)
        base = ThreetoksConfig()
        self.assertEqual(cfg.llm, base.llm)
        self.assertEqual(cfg.browser, base.browser)
        self.assertEqual(cfg.search, base.search)
        self.assertEqual(cfg.research, base.research)
        self.assertEqual(cfg.files, base.files)
        self.assertTrue(cfg.memory.enabled)
        self.assertEqual(len(prompts), 8)

    def test_prompts_show_the_defaults(self):
        ask, prompts = _scripted([""] * 8)
        run_wizard(ask=ask, sink=_quiet)
        self.assertIn(f"[{config.DEFAULT_MODEL}]", prompts[0])
        self.assertIn(f"[{config.DEFAULT_LLM_HOST}]", prompts[1])
        self.assertIn("http/plain/stealth", prompts[2])

    def test_base_config_supplies_the_defaults(self):
        base = ThreetoksConfig(llm=config.LlmConfig(model="tuned", vote_k=5))
        ask, _ = _scripted([""] * 8)
        cfg = run_wizard(base, ask=ask, sink=_quiet)
        self.assertEqual(cfg.llm.model, "tuned")
        self.assertEqual(cfg.llm.vote_k, 5)  # carried over, never asked

    def test_memory_default_moves_off_the_cwd(self):
        ask, _ = _scripted([""] * 8)
        cfg = run_wizard(ask=ask, sink=_quiet)
        self.assertNotEqual(cfg.memory.path, config.DEFAULT_MEMORY_PATH)
        self.assertIn("threetoks", cfg.memory.path)
        self.assertTrue(cfg.memory.path.endswith("memory.json"))

    def test_custom_memory_path_is_kept_as_suggestion(self):
        base = ThreetoksConfig(memory=config.MemoryConfig(path="/tuned.json"))
        ask, prompts = _scripted([""] * 8)
        cfg = run_wizard(base, ask=ask, sink=_quiet)
        self.assertEqual(cfg.memory.path, "/tuned.json")
        self.assertIn("[/tuned.json]", prompts[-1])


class WizardAnswersTest(unittest.TestCase):
    def test_full_answers_override_every_asked_field(self):
        ask, _ = _scripted(["my-model", "http://h:9", "stealth", "y",
                            "http://s:1", "7", "/data", "yes", "/mem.json"])
        cfg = run_wizard(ask=ask, sink=_quiet)
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
        ask, prompts = _scripted(["", "", "http", "", "", "", "y", ""])
        cfg = run_wizard(ask=ask, sink=_quiet)
        self.assertFalse(cfg.browser.visible)
        self.assertFalse(any("window" in prompt for prompt in prompts))

    def test_memory_path_question_skipped_when_disabled(self):
        ask, prompts = _scripted(["", "", "", "", "", "", "n"])
        cfg = run_wizard(ask=ask, sink=_quiet)
        self.assertFalse(cfg.memory.enabled)
        self.assertEqual(len(prompts), 7)

    def test_invalid_browser_mode_reasks(self):
        ask, _ = _scripted(["", "", "teleport", "PLAIN", "n",
                            "", "", "", "", ""])
        cfg = run_wizard(ask=ask, sink=_quiet)
        self.assertEqual(cfg.browser.mode, "plain")

    def test_invalid_round_count_reasks(self):
        ask, _ = _scripted(["", "", "", "", "many", "-2", "0", "4", "", "",
                            ""])
        cfg = run_wizard(ask=ask, sink=_quiet)
        self.assertEqual(cfg.research.max_rounds, 4)

    def test_bool_word_variants_parse(self):
        for word, expected in (("on", True), ("TRUE", True), ("1", True),
                               ("off", False), ("0", False), ("No", False)):
            answers = ["", "", "", "", "", "", word]
            if expected:
                answers.append("")
            ask, _ = _scripted(answers)
            cfg = run_wizard(ask=ask, sink=_quiet)
            self.assertEqual(cfg.memory.enabled, expected, word)

    def test_garbled_bool_reasks(self):
        ask, _ = _scripted(["", "", "", "", "", "", "maybe", "nope", "no"])
        cfg = run_wizard(ask=ask, sink=_quiet)
        self.assertFalse(cfg.memory.enabled)


class RunOnboardingTest(unittest.TestCase):
    def test_saves_to_the_target_and_loads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "nested", "config.ini")
            lines = []
            ask, _ = _scripted(["wiz-model", "", "plain", "n", "", "2", "",
                                "y", "/mem.json"])
            with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: target}):
                written = run_onboarding(ask=ask, sink=lines.append)
                self.assertEqual(written, target)
                cfg = load_config()
        self.assertEqual(cfg.llm.model, "wiz-model")
        self.assertEqual(cfg.browser.mode, "plain")
        self.assertEqual(cfg.research.max_rounds, 2)
        self.assertEqual(cfg.memory.path, "/mem.json")
        self.assertTrue(any(target in line for line in lines))


if __name__ == "__main__":
    unittest.main()
