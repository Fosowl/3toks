"""Offline tests for the chat-API transports and the backend factory.

Every test injects a canned ``_post`` — no network, no API keys leave
the process environment (which is patched per test).
"""
import os
import unittest
from dataclasses import dataclass
from unittest import mock

from threetoks.backend.base import ChatPrompt, GenOpts
from threetoks.backend.ollama import OllamaBackend
from threetoks.backend.providers import (OPENAI_COMPAT, PROVIDERS,
                                         AnthropicBackend, OpenAIChatBackend,
                                         _strip_prefill, make_backend)
from threetoks.config import PROVIDER_CHOICES


@dataclass
class _Llm:
    provider: str
    model: str = "m"
    host: str = ""
    api_base: str = ""


class _CannedOpenAI(OpenAIChatBackend):
    reply = {"choices": [{"message": {"content": " 2"},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 11, "completion_tokens": 3}}

    def _post(self, path, payload):
        self.sent = (path, payload)
        return self.reply


class _CannedAnthropic(AnthropicBackend):
    reply = {"content": [{"type": "text", "text": "AN"},
                         {"type": "text", "text": "SWER"}],
             "usage": {"input_tokens": 9, "output_tokens": 2},
             "stop_reason": "end_turn"}

    def _post(self, path, payload):
        self.sent = (path, payload)
        return self.reply


class RegistrySyncTest(unittest.TestCase):
    def test_config_choices_match_provider_registry(self):
        self.assertEqual(set(PROVIDERS), set(PROVIDER_CHOICES))


class MakeBackendTest(unittest.TestCase):
    def test_ollama_uses_raw_backend_with_configured_host(self):
        backend = make_backend(_Llm("ollama", host="http://box:1234"))
        self.assertIsInstance(backend, OllamaBackend)
        self.assertEqual(backend.host, "http://box:1234")

    def test_anthropic_needs_its_key_and_honours_api_base(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                make_backend(_Llm("anthropic"))
        env = {"ANTHROPIC_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            backend = make_backend(_Llm("anthropic", api_base="http://p/"))
        self.assertIsInstance(backend, AnthropicBackend)
        self.assertEqual(backend.base_url, "http://p")

    def test_every_openai_compat_provider_resolves(self):
        env = {key_env: "k" for _, key_env, _ in OPENAI_COMPAT.values()}
        with mock.patch.dict(os.environ, env, clear=True):
            for name in OPENAI_COMPAT:
                llm = _Llm(name, api_base="http://x" if name == "custom"
                           else "")
                self.assertIsInstance(make_backend(llm), OpenAIChatBackend,
                                      name)

    def test_missing_required_key_raises_naming_the_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                make_backend(_Llm("openrouter"))

    def test_lm_studio_works_without_a_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            backend = make_backend(_Llm("lm-studio"))
        self.assertEqual(backend.api_key, "")
        self.assertIn("1234", backend.base_url)

    def test_custom_without_api_base_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "api_base"):
                make_backend(_Llm("custom"))

    def test_unknown_provider_raises(self):
        with self.assertRaisesRegex(RuntimeError, "unknown llm provider"):
            make_backend(_Llm("nope"))


class OpenAIChatBackendTest(unittest.TestCase):
    def chat(self, prompt, opts):
        backend = _CannedOpenAI("https://api.example.test/v1", api_key="k")
        result = backend.chat("m", prompt, opts)
        return backend.sent, result

    def test_payload_shape_and_prefill_hint(self):
        (path, payload), _ = self.chat(ChatPrompt("sys", "pick", "ANSWER:"),
                                       GenOpts(max_tokens=3, seed=7))
        self.assertEqual(path, "/chat/completions")
        self.assertEqual(payload["messages"][0],
                         {"role": "system", "content": "sys"})
        self.assertIn('Begin your reply with "ANSWER:"',
                      payload["messages"][1]["content"])
        self.assertEqual(payload["max_tokens"], 3)
        self.assertEqual(payload["seed"], 7)
        self.assertNotIn("stop", payload)

    def test_no_system_and_no_prefill_stay_absent(self):
        (_, payload), _ = self.chat(ChatPrompt("", "pick"),
                                    GenOpts(max_tokens=3))
        roles = [message["role"] for message in payload["messages"]]
        self.assertEqual(roles, ["user"])
        self.assertEqual(payload["messages"][0]["content"], "pick")
        self.assertNotIn("seed", payload)

    def test_stops_capped_at_four(self):
        (_, payload), _ = self.chat(ChatPrompt("", "u"),
                                    GenOpts(max_tokens=3,
                                            stop=("a", "b", "c", "d", "e")))
        self.assertEqual(payload["stop"], ["a", "b", "c", "d"])

    def test_images_become_data_uri_content_parts(self):
        (_, payload), _ = self.chat(ChatPrompt("", "what is this?"),
                                    GenOpts(max_tokens=9, images=("QUJD",)))
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "what is this?"})
        self.assertEqual(parts[1]["image_url"]["url"],
                         "data:image/jpeg;base64,QUJD")

    def test_result_mapping_and_prefill_strip(self):
        _, result = self.chat(ChatPrompt("", "u", "ANSWER:"),
                              GenOpts(max_tokens=3))
        self.assertEqual((result.text, result.prompt_tokens,
                          result.out_tokens, result.done_reason),
                         (" 2", 11, 3, "stop"))

    def test_echoed_prefill_is_stripped(self):
        backend = _CannedOpenAI("https://x", api_key="k")
        backend.reply = {"choices": [{"message":
                                      {"content": "ANSWER: 3"}}]}
        result = backend.chat("m", ChatPrompt("", "u", "ANSWER:"),
                              GenOpts(max_tokens=3))
        self.assertEqual(result.text, " 3")
        self.assertEqual(result.out_tokens, 0)  # absent usage -> zero

    def test_empty_choices_yield_empty_text(self):
        backend = _CannedOpenAI("https://x")
        backend.reply = {"choices": []}
        result = backend.chat("m", ChatPrompt("", "u"), GenOpts(max_tokens=3))
        self.assertEqual((result.text, result.done_reason), ("", ""))


class AnthropicBackendTest(unittest.TestCase):
    def chat(self, prompt, opts):
        backend = _CannedAnthropic("k")
        result = backend.chat("m", prompt, opts)
        return backend.sent, result

    def test_prefill_rides_as_trailing_assistant_message(self):
        (path, payload), _ = self.chat(ChatPrompt("sys", "pick", "ANSWER: "),
                                       GenOpts(max_tokens=3))
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(payload["system"], "sys")
        self.assertEqual(payload["messages"][-1],
                         {"role": "assistant", "content": "ANSWER:"})

    def test_whitespace_only_stops_are_dropped(self):
        (_, payload), _ = self.chat(ChatPrompt("", "u"),
                                    GenOpts(max_tokens=3,
                                            stop=("\n", "END")))
        self.assertEqual(payload["stop_sequences"], ["END"])
        self.assertNotIn("system", payload)

    def test_no_prefill_keeps_single_user_message(self):
        (_, payload), _ = self.chat(ChatPrompt("", "u"),
                                    GenOpts(max_tokens=3, stop=("\n",)))
        self.assertEqual(len(payload["messages"]), 1)
        self.assertNotIn("stop_sequences", payload)

    def test_images_become_source_blocks_before_text(self):
        (_, payload), _ = self.chat(ChatPrompt("", "look"),
                                    GenOpts(max_tokens=9, images=("QUJD",)))
        parts = payload["messages"][0]["content"]
        self.assertEqual(parts[0]["source"],
                         {"type": "base64", "media_type": "image/jpeg",
                          "data": "QUJD"})
        self.assertEqual(parts[1], {"type": "text", "text": "look"})

    def test_result_joins_text_blocks_and_maps_usage(self):
        _, result = self.chat(ChatPrompt("", "u"), GenOpts(max_tokens=3))
        self.assertEqual((result.text, result.prompt_tokens,
                          result.out_tokens, result.done_reason),
                         ("ANSWER", 9, 2, "end_turn"))


class TransportErrorTest(unittest.TestCase):
    def test_unreachable_host_raises_a_clean_runtime_error(self):
        from threetoks.backend.providers import _http_post_json
        with self.assertRaisesRegex(RuntimeError, "cannot reach"):
            _http_post_json("http://127.0.0.1:9/none", {}, {}, timeout_s=1)


class StripPrefillTest(unittest.TestCase):
    def test_strips_echo_even_after_leading_whitespace(self):
        self.assertEqual(_strip_prefill("  ANSWER: 2", "ANSWER:"), " 2")

    def test_leaves_unechoed_text_alone(self):
        self.assertEqual(_strip_prefill("2", "ANSWER:"), "2")

    def test_empty_prefill_is_identity(self):
        self.assertEqual(_strip_prefill("ANSWER: 2", ""), "ANSWER: 2")


class OllamaImagesTest(unittest.TestCase):
    class _Canned(OllamaBackend):
        def _post(self, path, payload):
            self.sent = payload
            return {"response": "ok"}

    def test_images_ride_in_the_payload_only_when_present(self):
        backend = self._Canned()
        backend.complete("m", "p", GenOpts(max_tokens=3, images=("QUJD",)))
        self.assertEqual(backend.sent["images"], ["QUJD"])
        backend.complete("m", "p", GenOpts(max_tokens=3))
        self.assertNotIn("images", backend.sent)


if __name__ == "__main__":
    unittest.main()
