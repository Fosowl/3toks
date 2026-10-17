"""Tests for the health probes and the hints built on them."""
import http.server
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from threetoks import probes
from threetoks.config import (FilesConfig, LlmConfig, ThreetoksConfig,
                              BrowserConfig)
from threetoks.fields import find_field
from threetoks.hints import LEVEL_BAD, LEVEL_OK, LEVEL_WARN, hint_for
from threetoks.probes import Probes


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/tags":
            body = json.dumps({"models": [{"name": "m:1"}, {}]}).encode()
            self.send_response(200)
        else:
            body, _ = b"nope", self.send_response(404)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_models_are_listed_and_nameless_entries_dropped(self):
        self.assertEqual(probes.ollama_models(self.url), ("m:1",))

    def test_an_error_page_still_counts_as_reachable(self):
        self.assertTrue(probes.http_reachable(self.url + "/missing"))

    def test_dead_port_is_not_reachable(self):
        self.assertFalse(probes.http_reachable("http://127.0.0.1:9"))
        self.assertIsNone(probes.ollama_models("http://127.0.0.1:9"))

    def test_non_web_schemes_are_refused(self):
        self.assertFalse(probes.http_reachable("file:///etc/passwd"))
        self.assertIsNone(probes.ollama_models("ftp://x"))

    def test_sweep_asks_ollama_only_for_the_ollama_provider(self):
        local = ThreetoksConfig(llm=LlmConfig(host=self.url))
        self.assertEqual(probes.sweep(local).models, ("m:1",))
        cloud = ThreetoksConfig(llm=LlmConfig(provider="openai",
                                              host=self.url))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            found = probes.sweep(cloud)
        self.assertIsNone(found.models)
        self.assertEqual(found.key, ("OPENAI_API_KEY", True))


class KeyStatusTest(unittest.TestCase):
    def test_local_provider_needs_no_key(self):
        self.assertIsNone(probes.key_status("ollama"))

    def test_missing_key_is_reported_by_name(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(probes.key_status("anthropic"),
                             ("ANTHROPIC_API_KEY", False))


class HintTest(unittest.TestCase):
    def hint(self, dotted, config=None, **probe_args):
        return hint_for(find_field(dotted), config or ThreetoksConfig(),
                        Probes(**probe_args))

    def test_model_hint_follows_ollama_state(self):
        self.assertEqual(self.hint("llm.model").level, LEVEL_WARN)
        up = dict(ollama_up=True, models=("other",))
        bad = self.hint("llm.model", **up)
        self.assertEqual(bad.level, LEVEL_BAD)
        self.assertIn("ollama pull qwen3.5:2b", bad.text)

    def test_bare_model_name_matches_latest_tag(self):
        config = ThreetoksConfig(llm=LlmConfig(model="qwen3"))
        hint = self.hint("llm.model", config, ollama_up=True,
                         models=("qwen3:latest",))
        self.assertEqual(hint.level, LEVEL_OK)

    def test_chat_provider_has_no_ollama_hints(self):
        config = ThreetoksConfig(llm=LlmConfig(provider="openai"))
        self.assertIsNone(self.hint("llm.model", config))
        self.assertIsNone(self.hint("llm.host", config))

    def test_custom_provider_requires_an_api_base(self):
        config = ThreetoksConfig(llm=LlmConfig(provider="custom"))
        self.assertEqual(self.hint("llm.api_base", config).level, LEVEL_BAD)

    def test_plain_browser_warns_when_searxng_is_down(self):
        config = ThreetoksConfig(browser=BrowserConfig(mode="plain"))
        self.assertEqual(self.hint("browser.mode", config).level, LEVEL_WARN)
        self.assertIsNone(self.hint("browser.mode", config, searxng_up=True))

    def test_files_root_must_be_a_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            good = ThreetoksConfig(files=FilesConfig(root=folder))
            self.assertEqual(self.hint("files.root", good).level, LEVEL_OK)
        bad = ThreetoksConfig(files=FilesConfig(root="/no/such/dir"))
        self.assertEqual(self.hint("files.root", bad).level, LEVEL_BAD)


if __name__ == "__main__":
    unittest.main()
