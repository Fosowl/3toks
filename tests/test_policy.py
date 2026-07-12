"""Unit tests for the policy retry ladder, permutation, and think phase."""
import unittest

from threetoks.backend.base import FAMILY_R1, GenResult, ModelSpec
from threetoks.nodes import MenuNode, PickManyNode
from threetoks.policy import TEMPERATURE_LADDER, Policy, PolicyConfig
from threetoks.render import Episode


class FakeBackend:
    """Scripted backend that records every prompt and option set."""

    def __init__(self, scripted_texts):
        self.scripted = list(scripted_texts)
        self.calls = []

    def complete(self, model, raw_prompt, opts):
        self.calls.append((raw_prompt, opts))
        return GenResult(self.scripted.pop(0), 10, 2, 0.01, "stop")


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


class NonMenuNodeTest(unittest.TestCase):
    def test_pick_many_gets_no_permutation(self):
        backend = FakeBackend(["3,7"])
        decision = make_policy(backend).decide(Episode("S", "t"),
                                               PickManyNode("Which?", 10))
        self.assertEqual(decision.value, [3, 7])


if __name__ == "__main__":
    unittest.main()
