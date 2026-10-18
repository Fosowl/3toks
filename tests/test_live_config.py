"""Offline tests: a saved config reaches the running chat without a restart.

Each test starts a real session with ``session.build_state`` on a temp
config file (``$THREETOKS_CONFIG``), saves a change, and checks the live
part it controls. SearXNG, the camera and the relay are switched off so
the tests never touch the network or hardware.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from threetoks import session as session_mod
from threetoks.config import CONFIG_ENV_VAR
from threetoks.live_config import sync_config
from threetoks.session import build_state


class LiveCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.path = os.path.join(self.folder, "c.ini")
        for patcher in (
                mock.patch.dict(os.environ, {CONFIG_ENV_VAR: self.path}),
                mock.patch("threetoks.web.search._searxng_answers",
                           return_value=False),
                mock.patch.object(session_mod, "make_capture",
                                  return_value=None),
                mock.patch.object(session_mod, "make_relay",
                                  return_value=None)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.save(f"[memory]\npath = {self.folder}/one.json\n")
        self.state = build_state(sink=[].append)
        self.state.enabled = False

    def save(self, text):
        """Write the config and move its mtime, like a real later save."""
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(text)
        stamp = getattr(self, "_mtime", 1_000_000_000_000_000_000) + 1_000
        os.utime(self.path, ns=(stamp, stamp))
        self._mtime = stamp


class SyncTest(LiveCase):
    def test_saved_changes_reach_policy_and_services(self):
        self.save(f"[memory]\npath = {self.folder}/one.json\n"
                  f"[llm]\nvote_k = 3\n[research]\nmax_rounds = 5\n"
                  f"max_steps = 9\n[files]\nroot = {self.folder}\n")
        report = sync_config(self.state)
        self.assertIn("config applied (4 of 4)", report)
        self.assertEqual(self.state.policy.config.vote_k, 3)
        services = self.state.services
        self.assertEqual((services.max_research_rounds,
                          services.max_research_steps), (5, 9))
        self.assertEqual(str(services.files_root), self.folder)

    def test_an_unchanged_file_is_not_applied_again(self):
        policy = self.state.policy
        self.assertEqual(sync_config(self.state), "")
        self.assertIs(self.state.policy, policy)

    def test_a_model_chosen_this_session_survives_other_llm_edits(self):
        self.state.model = "picked:1b"  # from -m or /model
        self.save(f"[memory]\npath = {self.folder}/one.json\n"
                  "[llm]\nvote_k = 2\n")
        sync_config(self.state)
        self.assertEqual(self.state.policy.config.spec.name, "picked:1b")
        self.save(f"[memory]\npath = {self.folder}/one.json\n"
                  "[llm]\nvote_k = 2\nmodel = saved:2b\n")
        sync_config(self.state)
        self.assertEqual(self.state.model, "saved:2b")

    def test_a_broken_file_keeps_the_running_settings(self):
        self.save("[llm]\nvote_k = 3\nno delimiter here\n")
        report = sync_config(self.state)
        self.assertIn("keeps its settings", report)
        self.assertEqual(self.state.policy.config.vote_k, 1)

    def test_a_provider_that_cannot_start_keeps_the_old_llm_only(self):
        memory = f"[memory]\npath = {self.folder}/one.json\n"
        self.save(memory + "[llm]\nprovider = openai\n[research]\n"
                  "max_rounds = 6\n")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            report = sync_config(self.state)
        self.assertIn("[llm] not applied", report)
        self.assertEqual(self.state.config.llm.provider, "ollama")
        self.assertEqual(self.state.services.max_research_rounds, 6)
        self.save(memory + "[llm]\nprovider = openai\n[research]\n"
                  "max_rounds = 6\n\n")  # saved again, key now set
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            sync_config(self.state)
        self.assertEqual(self.state.config.llm.provider, "openai")

    def test_memory_moves_to_the_new_file_and_keeps_the_old(self):
        self.state.memory.remember("web", "q", "a")
        self.save(f"[memory]\npath = {self.folder}/two.json\n")
        sync_config(self.state)
        with open(os.path.join(self.folder, "one.json"),
                  encoding="utf-8") as handle:
            self.assertIn("q", json.dumps(json.load(handle)))
        self.assertEqual(self.state.config.memory.path,
                         f"{self.folder}/two.json")
        self.assertEqual(self.state.memory.count(), 0)

    def test_saved_searxng_url_is_the_one_probed(self):
        probed = []
        self.save(f"[memory]\npath = {self.folder}/one.json\n"
                  "[search]\nsearxng_url = http://sx:9\n")
        with mock.patch("threetoks.web.search._searxng_answers",
                        side_effect=lambda url: probed.append(url)):
            sync_config(self.state)
        self.assertEqual(probed, ["http://sx:9"])

    def test_voice_follows_a_saved_enabled_switch(self):
        session = mock.Mock()
        self.save(f"[memory]\npath = {self.folder}/one.json\n"
                  "[voice]\nenabled = true\n")
        with mock.patch("threetoks.voice.session.make_voice_session",
                        return_value=(session, [])) as factory:
            sync_config(self.state)
        self.assertIs(self.state.voice, session)
        self.assertTrue(factory.call_args.args[0].enabled)
        self.save(f"[memory]\npath = {self.folder}/one.json\n")
        sync_config(self.state)
        self.assertIsNone(self.state.voice)
        session.close.assert_called_once_with()

    def test_a_browser_that_cannot_start_is_tried_again_next_save(self):
        memory = f"[memory]\npath = {self.folder}/one.json\n"
        self.save(memory + "[browser]\nmode = plain\n")
        with mock.patch.object(session_mod, "make_browser_fetcher",
                               side_effect=RuntimeError("no chrome")):
            report = sync_config(self.state)
        self.assertIn("[browser] not applied", report)
        self.assertEqual(self.state.config.browser.mode, "http")
        self.save(memory + "[browser]\nmode = plain\n[llm]\nvote_k = 2\n")
        fetcher = mock.Mock()
        with mock.patch.object(session_mod, "make_browser_fetcher",
                               return_value=fetcher):
            sync_config(self.state)
        self.assertIs(self.state.fetcher, fetcher)

    def test_a_deep_choice_survives_an_edit_of_the_other_budget(self):
        self.state.services.max_research_rounds = 7  # from /deep 7
        self.save(f"[memory]\npath = {self.folder}/one.json\n"
                  "[research]\nmax_steps = 30\n")
        sync_config(self.state)
        self.assertEqual(self.state.services.max_research_rounds, 7)
        self.assertEqual(self.state.services.max_research_steps, 30)

    def test_ctrl_c_mid_apply_keeps_the_session_and_retries_the_rest(self):
        self.save(f"[memory]\npath = {self.folder}/one.json\n"
                  "[research]\nmax_rounds = 5\n[voice]\nenabled = true\n")
        with mock.patch("threetoks.voice.session.make_voice_session",
                        side_effect=KeyboardInterrupt):
            report = sync_config(self.state)
        self.assertIn("next line", report)
        self.assertEqual(self.state.config.research.max_rounds, 5)
        with mock.patch("threetoks.voice.session.make_voice_session",
                        return_value=(mock.Mock(), [])):
            sync_config(self.state)
        self.assertIsNotNone(self.state.voice)


class ApplierCoverageTest(unittest.TestCase):
    def test_every_config_section_has_an_applier(self):
        from dataclasses import fields

        from threetoks.config import ThreetoksConfig
        from threetoks.live_config import APPLIERS
        sections = {item.name for item in fields(ThreetoksConfig)}
        self.assertEqual({name for name, _ in APPLIERS}, sections)


if __name__ == "__main__":
    unittest.main()
