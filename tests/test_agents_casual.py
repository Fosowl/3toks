"""Offline tests for the casual agent (no LLM; scripted backend)."""
import unittest

from threetoks.agents import casual
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.policy import Policy, PolicyConfig


class CountingBackend:
    """Scripted backend that records how many completions it served."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, model, raw_prompt, opts):
        self.calls += 1
        text = self.texts.pop(0) if self.texts else " 9"
        return GenResult(text, 5, 2, 0.0, "stop")


def make_policy(texts):
    return Policy(CountingBackend(texts),
                  PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


class ExactShortcutTest(unittest.TestCase):
    def test_exact_hint_returns_reply_with_zero_model_calls(self):
        policy = make_policy([])
        result = casual.SPEC.run("hello", None, policy)
        self.assertEqual(result["answer"],
                         "Hello! How can I help you today?")
        self.assertEqual(result["agent"], "casual")
        self.assertEqual(policy.backend.calls, 0)

    def test_shortcut_is_case_and_whitespace_insensitive(self):
        policy = make_policy([])
        result = casual.SPEC.run("  Thank You  ", None, policy)
        self.assertEqual(result["answer"], "You're welcome! Happy to help.")
        self.assertEqual(policy.backend.calls, 0)

    def test_who_are_you_shortcut_identifies_threetoks(self):
        policy = make_policy([])
        result = casual.SPEC.run("who are you", None, policy)
        self.assertIn("ThreeToks", result["answer"])
        self.assertEqual(policy.backend.calls, 0)


class MenuPathTest(unittest.TestCase):
    def test_menu_returns_full_reply_for_the_chosen_option(self):
        # "hi there friend" has no exact hint but ranks the greeting reply;
        # the menu is offered and digit 1 maps back to the FULL text.
        policy = make_policy([" 1"])
        result = casual.SPEC.run("hi there friend", None, policy)
        self.assertEqual(result["answer"],
                         "Hello! How can I help you today?")
        self.assertEqual(result["agent"], "casual")
        self.assertEqual(policy.backend.calls, 1)

    def test_menu_reply_is_not_the_truncated_option(self):
        long = "Sorry, that's beyond what a tiny local agent can do right now."
        policy = make_policy([" 1"])
        result = casual.SPEC.run("sorry about the trouble", None, policy)
        self.assertEqual(result["answer"], long)
        self.assertGreater(len(result["answer"]), casual.OPTION_CHARS)


class EscapePathTest(unittest.TestCase):
    def test_no_candidates_falls_through_to_free_reply(self):
        policy = make_policy([" Nice to hear from you!"])
        result = casual.SPEC.run("zxqw foobar plugh", None, policy)
        self.assertEqual(result["answer"], "Nice to hear from you!")
        self.assertEqual(result["agent"], "casual")

    def test_menu_escape_falls_through_to_free_reply(self):
        # "hi thanks bye" ranks 3 candidates, so the escape option is digit
        # 4; picking it drops through to a bounded free reply.
        policy = make_policy([" 4", " A friendly free reply."])
        result = casual.SPEC.run("hi thanks bye", None, policy)
        self.assertEqual(result["answer"], "A friendly free reply.")

    def test_free_reply_uses_fallback_when_model_is_empty(self):
        policy = make_policy(["", "", ""])
        result = casual.SPEC.run("zxqw foobar plugh", None, policy)
        self.assertEqual(result["answer"], casual.FALLBACK_REPLY)


class FreeReplyBleedTest(unittest.TestCase):
    def test_digit_bleed_falls_back_instead_of_answering_a_number(self):
        policy = make_policy([" 2"])  # menu-format bleed on a text node
        result = casual.SPEC.run("zxqw foobar plugh", None, policy)
        self.assertEqual(result["answer"], casual.FALLBACK_REPLY)


class MissingReplyBankTest(unittest.TestCase):
    def test_missing_reply_file_degrades_to_builtins(self):
        from pathlib import Path
        original = casual.REPLIES_PATH
        casual.REPLIES_PATH = Path("/nonexistent/casual_replies.json")
        try:
            policy = make_policy([])
            result = casual.SPEC.run("hello", None, policy)
        finally:
            casual.REPLIES_PATH = original
        self.assertEqual(result["answer"],
                         "Hello! How can I help you today?")
        self.assertEqual(policy.backend.calls, 0)  # shortcut still free


if __name__ == "__main__":
    unittest.main()
