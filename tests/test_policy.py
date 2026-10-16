"""Unit tests for the policy retry ladder, permutation, and think phase."""
import unittest

from threetoks.backend.base import FAMILY_R1, GenResult, ModelSpec
from threetoks.nodes import MenuNode, PickManyNode, ShortTextNode
from threetoks.policy import TEMPERATURE_LADDER, Policy, PolicyConfig
from threetoks.render import Episode
from threetoks.trace import Tracer


class FakeBackend:
    """Scripted backend that records every prompt and option set."""

    def __init__(self, scripted_texts, done_reason="stop"):
        self.scripted = list(scripted_texts)
        self.done_reason = done_reason
        self.calls = []

    def complete(self, model, raw_prompt, opts):
        self.calls.append((raw_prompt, opts))
        return GenResult(self.scripted.pop(0), 10, 2, 0.01, self.done_reason)


def make_policy(backend, **overrides):
    config = PolicyConfig(ModelSpec("m", FAMILY_R1), **overrides)
    return Policy(backend, config)


class RetryLadderTest(unittest.TestCase):
    def test_invalid_then_valid_uses_two_attempts(self):
        backend = FakeBackend(["garbage", " 2"])
        decision = make_policy(backend).decide(
            Episode("S", "t"), MenuNode("Pick.", ["a", "b", "c"]))
        self.assertTrue(decision.valid)
        self.assertEqual(len(backend.calls), 2)

    def test_temperature_ladder_applied_per_attempt(self):
        backend = FakeBackend(["x", "x", "x"])
        decision = make_policy(backend).decide(
            Episode("S", "t"), MenuNode("Pick.", ["a", "b"]))
        self.assertFalse(decision.valid)
        temps = [opts.temperature for _, opts in backend.calls]
        self.assertEqual(temps, list(TEMPERATURE_LADDER))

    def test_permutations_differ_across_attempts(self):
        backend = FakeBackend(["x", "x", " 1"])
        node = MenuNode("Pick.", ["a", "b", "c", "d", "e"])
        policy = make_policy(backend)
        perms = [policy._permutation(node, 0, attempt) for attempt in range(3)]
        self.assertEqual(len(set(perms)), 3)

    def test_menu_prompt_ends_with_prefill(self):
        backend = FakeBackend([" 1"])
        make_policy(backend).decide(Episode("S", "t"),
                                    MenuNode("Pick.", ["a", "b"]))
        self.assertTrue(backend.calls[0][0].endswith("ANSWER:"))


class ThinkPhaseTest(unittest.TestCase):
    def test_think_budget_adds_reasoning_phase(self):
        backend = FakeBackend(["hmm options...", " 1"])
        make_policy(backend).decide(Episode("S", "t"),
                                    MenuNode("Pick.", ["a", "b"]),
                                    think_budget=50)
        self.assertEqual(len(backend.calls), 2)
        think_prompt, think_opts = backend.calls[0]
        answer_prompt, _ = backend.calls[1]
        self.assertTrue(think_prompt.rstrip().endswith("<think>"))
        self.assertEqual(think_opts.max_tokens, 50)
        self.assertIn("hmm options...", answer_prompt)
        self.assertIn("</think>", answer_prompt)
        self.assertTrue(answer_prompt.endswith("ANSWER:"))


class VotingTest(unittest.TestCase):
    def test_majority_wins_across_permuted_samples(self):
        node = MenuNode("Pick.", ["a", "b", "c"])
        policy = make_policy(FakeBackend([]), vote_k=3)
        perms = [policy._permutation(node, 0, sample) for sample in range(3)]
        digit_for = {perm: str(perm.index(node.options.index("b")) + 1)
                     for perm in perms}
        backend = FakeBackend([digit_for[perms[0]], digit_for[perms[1]],
                               " 1"])  # two votes for "b", one stray
        decision = make_policy(backend, vote_k=3).decide(
            Episode("S", "t"), node)
        self.assertTrue(decision.valid)
        self.assertEqual(decision.value, "b")
        self.assertEqual(len(backend.calls), 3)

    def test_all_invalid_votes_fall_back_to_ladder(self):
        backend = FakeBackend(["x", "x", "x", " 2", "x", "x"])
        episode = Episode("S", "t")
        decision = make_policy(backend, vote_k=3).decide(
            episode, MenuNode("Pick.", ["a", "b"]))
        self.assertTrue(decision.valid)
        self.assertEqual(episode.step_count, 1)  # no double increment

    def test_vote_success_counts_one_step(self):
        backend = FakeBackend([" 1", " 1", " 1"])
        episode = Episode("S", "t")
        make_policy(backend, vote_k=3).decide(episode,
                                              MenuNode("Pick.", ["a", "b"]))
        self.assertEqual(episode.step_count, 1)

    def test_vote_only_applies_to_menu_nodes(self):
        backend = FakeBackend(["3,7"])
        decision = make_policy(backend, vote_k=3).decide(
            Episode("S", "t"), PickManyNode("Which?", 10))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(decision.value, [3, 7])


class TraceDoneReasonTest(unittest.TestCase):
    """The completion's finish reason must survive into the trace event.

    It is what separates a wrong answer from a truncated one: a chat
    provider that spends the node's 3-token cap on preamble stops with
    ``length`` and never reaches its digit, which the TUI's /debug line
    then shows instead of a bare "—".
    """

    def _traced(self, texts, done_reason):
        events = []
        backend = FakeBackend(texts, done_reason=done_reason)
        policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_R1)),
                        Tracer(None, on_event=events.append))
        policy.decide(Episode("S", "t"), MenuNode("Pick.", ["a", "b"]))
        return events

    def test_truncated_completions_trace_length(self):
        events = self._traced(["Okay, so the"] * 3, "length")
        self.assertEqual(len(events), 3)  # whole ladder, nothing parsed
        self.assertEqual([event["done_reason"] for event in events],
                         ["length"] * 3)
        self.assertFalse(any(event["valid"] for event in events))

    def test_ordinary_completion_traces_its_own_reason(self):
        [event] = self._traced([" 1"], "stop")
        self.assertEqual(event["done_reason"], "stop")

    def test_existing_event_keys_are_untouched(self):
        [event] = self._traced([" 1"], "stop")
        for key in ("step", "node", "question", "perm", "attempt",
                    "raw_text", "value", "valid", "prompt_tokens",
                    "out_tokens", "wall_s"):
            self.assertIn(key, event)


class NonMenuNodeTest(unittest.TestCase):
    def test_pick_many_gets_no_permutation(self):
        backend = FakeBackend(["3,7"])
        decision = make_policy(backend).decide(Episode("S", "t"),
                                               PickManyNode("Which?", 10))
        self.assertEqual(decision.value, [3, 7])


class StrictTextNode(ShortTextNode):
    """A text node that refuses completions cut by the token cap."""

    reject_truncated = True


def temps_of(backend):
    return [opts.temperature for _, opts in backend.calls]


class NonMenuRetryTemperatureTest(unittest.TestCase):
    def test_short_text_retries_hot_after_first_attempt(self):
        backend = FakeBackend(["", "", ""])
        make_policy(backend).decide(Episode("S", "t"),
                                    ShortTextNode("Query?", "Q:"))
        self.assertEqual(temps_of(backend), [0.0, 0.4, 0.4])

    def test_menu_keeps_the_ladder(self):
        backend = FakeBackend(["x", "x", "x"])
        make_policy(backend).decide(Episode("S", "t"),
                                    MenuNode("Pick.", ["a", "b"]))
        self.assertEqual(temps_of(backend), [0.0, 0.0, 0.4])

    def test_node_temperature_override_wins(self):
        node = ShortTextNode("Query?", "Q:")
        node.temperature = 0.9
        backend = FakeBackend(["", "", ""])
        make_policy(backend).decide(Episode("S", "t"), node)
        self.assertEqual(temps_of(backend), [0.9, 0.9, 0.9])


class RejectTruncatedTest(unittest.TestCase):
    def _decide(self, node, texts, reasons):
        events = []
        backend = FakeBackend(texts)
        reason_iter = iter(reasons)
        complete = backend.complete
        backend.complete = lambda *args: GenResult(
            complete(*args).text, 10, 2, 0.01, next(reason_iter))
        policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_R1)),
                        Tracer(None, on_event=events.append))
        return policy.decide(Episode("S", "t"), node), events

    def test_truncated_completion_is_retried(self):
        decision, events = self._decide(StrictTextNode("Q?", "Q:"),
                                        [" cut", " whole"], ["length", "stop"])
        self.assertTrue(decision.valid)
        self.assertEqual(len(events), 2)
        self.assertFalse(events[0]["valid"])
        self.assertEqual(events[0]["done_reason"], "length")

    def test_all_truncated_ends_invalid(self):
        decision, events = self._decide(StrictTextNode("Q?", "Q:"),
                                        [" a", " b", " c"], ["max_tokens"] * 3)
        self.assertFalse(decision.valid)
        self.assertEqual(len(events), 3)

    def test_node_without_flag_accepts_truncated(self):
        decision, events = self._decide(ShortTextNode("Q?", "Q:"),
                                        [" cut"], ["length"])
        self.assertTrue(decision.valid)
        self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
