"""Unit tests for the episode loop and vertical contract."""
import unittest

from threetoks.backend.base import FAMILY_R1, GenResult, ModelSpec
from threetoks.engine import run_episode
from threetoks.nodes import MenuNode
from threetoks.policy import Policy, PolicyConfig
from threetoks.render import Episode


class CountingVertical:
    """Serves `total` menus, then finishes."""

    def __init__(self, total):
        self.episode = Episode("S", "count")
        self.steps_left = 0
        self.total = total
        self.applied = []

    def next_node(self):
        if len(self.applied) >= self.total:
            return None
        return MenuNode("Continue?", ["go", "stop"])

    def apply(self, node, decision):
        self.applied.append(decision.value if decision.valid else "<invalid>")

    def result(self):
        return {"applied": self.applied, "steps_left": self.steps_left}


class AlwaysOneBackend:
    def complete(self, model, raw_prompt, opts):
        return GenResult(" 1", 5, 2, 0.0, "stop")


def make_policy():
    return Policy(AlwaysOneBackend(), PolicyConfig(ModelSpec("m", FAMILY_R1)))


class RunEpisodeTest(unittest.TestCase):
    def test_runs_until_vertical_is_done(self):
        outcome = run_episode(CountingVertical(3), make_policy(), max_steps=10)
        self.assertEqual(len(outcome["applied"]), 3)

    def test_step_budget_caps_the_loop(self):
        outcome = run_episode(CountingVertical(99), make_policy(), max_steps=5)
        self.assertEqual(len(outcome["applied"]), 5)

    def test_vertical_sees_remaining_budget(self):
        outcome = run_episode(CountingVertical(2), make_policy(), max_steps=10)
        self.assertEqual(outcome["steps_left"], 8)


if __name__ == "__main__":
    unittest.main()
