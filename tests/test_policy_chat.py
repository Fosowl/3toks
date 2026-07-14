"""Offline tests: the policy's dispatch between raw and chat transports."""
import unittest

from threetoks.backend.base import (FAMILY_CHATML, GenResult, ModelSpec)
from threetoks.nodes import MenuNode, PickManyNode, ShortTextNode
from threetoks.policy import Policy, PolicyConfig
from threetoks.render import Episode
from threetoks.trace import Tracer


class FakeChatBackend:
    """Chat-capable fake recording every call."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def chat(self, model, prompt, opts):
        self.calls.append((model, prompt, opts))
        return GenResult(self.scripted.pop(0), 10, 2, 0.01, "stop")


class FakeRawBackend:
    """Raw-only fake recording prompts and opts."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []

    def complete(self, model, raw_prompt, opts):
        self.calls.append((raw_prompt, opts))
        return GenResult(self.scripted.pop(0), 10, 2, 0.01, "stop")


def make_policy(backend, vote_k=1):
    spec = ModelSpec("qwen2.5:1.5b-instruct", FAMILY_CHATML)
    return Policy(backend, PolicyConfig(spec, system="SYS", vote_k=vote_k))


class ChatDispatchTest(unittest.TestCase):
    def test_menu_goes_through_chat_with_untemplated_parts(self):
        backend = FakeChatBackend([" 1"])
        policy = make_policy(backend)
        decision = policy.decide(Episode("SYS", "t"),
                                 MenuNode("Pick.", ["a", "b"]))
        self.assertTrue(decision.valid)
        model, prompt, opts = backend.calls[0]
        self.assertEqual(model, "qwen2.5:1.5b-instruct")
        self.assertEqual(prompt.system, "SYS")
        self.assertEqual(prompt.prefill, "ANSWER:")
        self.assertIn("ACTIONS:", prompt.user)
        self.assertNotIn("<|im_start|>", prompt.user)  # no template junk

    def test_chat_gets_node_stops_only_never_family_stops(self):
        backend = FakeChatBackend(["1"])
        policy = make_policy(backend)
        policy.decide(Episode("SYS", "t"), PickManyNode("Which?", 3))
        _, _, opts = backend.calls[0]
        self.assertEqual(opts.stop, ("\n",))

    def test_raw_backend_still_gets_family_stops_and_template(self):
        backend = FakeRawBackend([" 1"])
        policy = make_policy(backend)
        policy.decide(Episode("SYS", "t"), MenuNode("Pick.", ["a", "b"]))
        raw_prompt, opts = backend.calls[0]
        self.assertIn("<|im_start|>", raw_prompt)
        self.assertIn("<|im_end|>", opts.stop)

    def test_node_images_reach_both_transports(self):
        node = ShortTextNode("What is it?", "", max_tokens=8)
        node.images = ("QUJD",)
        chat_backend = FakeChatBackend(["a cat"])
        make_policy(chat_backend).decide(Episode("SYS", "t"), node)
        self.assertEqual(chat_backend.calls[0][2].images, ("QUJD",))
        raw_backend = FakeRawBackend(["a cat"])
        make_policy(raw_backend).decide(Episode("SYS", "t"), node)
        self.assertEqual(raw_backend.calls[0][1].images, ("QUJD",))

    def test_trace_records_chat_mode_marker(self):
        events = []
        for backend in (FakeChatBackend([" 1"]), FakeRawBackend([" 1"])):
            policy = make_policy(backend)
            policy.tracer = Tracer(None)
            policy.tracer.on_event = events.append
            policy.decide(Episode("SYS", "t"), MenuNode("P.", ["a", "b"]))
        self.assertEqual(events[0].get("mode"), "chat")
        self.assertNotIn("mode", events[1])

    def test_vote_path_also_dispatches_to_chat(self):
        backend = FakeChatBackend([" 1", " 1", " 2"])
        policy = make_policy(backend, vote_k=3)
        decision = policy.decide(Episode("SYS", "t"),
                                 MenuNode("Pick.", ["a", "b"]))
        self.assertTrue(decision.valid)
        self.assertEqual(len(backend.calls), 3)

    def test_retry_ladder_runs_over_chat(self):
        backend = FakeChatBackend(["junk", " 2"])
        policy = make_policy(backend)
        decision = policy.decide(Episode("SYS", "t"),
                                 MenuNode("Pick.", ["a", "b"]))
        self.assertTrue(decision.valid)
        self.assertEqual(len(backend.calls), 2)

    def test_non_callable_chat_attribute_falls_back_to_raw(self):
        backend = FakeRawBackend([" 1"])
        backend.chat = "not a method"
        policy = make_policy(backend)
        decision = policy.decide(Episode("SYS", "t"),
                                 MenuNode("Pick.", ["a", "b"]))
        self.assertTrue(decision.valid)
        self.assertEqual(len(backend.calls), 1)


if __name__ == "__main__":
    unittest.main()
