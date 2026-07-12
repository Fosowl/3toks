"""Episode loop: the harness walks the tree, the model only answers nodes.

A Vertical owns the domain state machine (web research, file ops, ...).
Contract:
- `episode`     the Episode whose prompt it maintains,
- `next_node()` returns the next decision node, or None when done,
- `apply(node, decision)` executes the decision deterministically —
  including handling invalid decisions (treat as escape/no-op),
- `result()`    returns the episode outcome (best-effort if the step
  budget ran out; the vertical sees the budget via `steps_left`).
"""
from typing import Protocol

from threetoks.nodes import Decision
from threetoks.policy import Policy

DEFAULT_MAX_STEPS = 40


class Vertical(Protocol):
    """Domain state machine driven by the engine."""
    episode: object
    steps_left: int

    def next_node(self):
        """Next decision node, or None when the episode is complete."""
        ...

    def apply(self, node, decision: Decision) -> None:
        """Execute one decision (must handle decision.valid == False)."""
        ...

    def result(self) -> dict:
        """Outcome of the episode."""
        ...


def run_episode(vertical: Vertical, policy: Policy,
                max_steps: int = DEFAULT_MAX_STEPS) -> dict:
    """Drive the vertical until done or out of steps; return its result."""
    for remaining in range(max_steps, 0, -1):
        vertical.steps_left = remaining
        node = vertical.next_node()
        if node is None:
            break
        decision = policy.decide(vertical.episode, node)
        vertical.apply(node, decision)
    return vertical.result()


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_R1, GenResult, ModelSpec
    from threetoks.nodes import MenuNode
    from threetoks.policy import PolicyConfig
    from threetoks.render import Episode

    class _OneMenuVertical:
        """Toy vertical: a single menu, then done."""

        def __init__(self):
            self.episode = Episode("SYS", "toy")
            self.steps_left = 0
            self.picked = None

        def next_node(self):
            return None if self.picked else MenuNode("Pick.", ["go", "stop"])

        def apply(self, node, decision):
            self.picked = decision.value if decision.valid else "<invalid>"

        def result(self):
            return {"picked": self.picked}

    class _FakeBackend:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" 1", 5, 2, 0.0, "stop")

    policy = Policy(_FakeBackend(), PolicyConfig(ModelSpec("m", FAMILY_R1)))
    outcome = run_episode(_OneMenuVertical(), policy)
    assert outcome["picked"] in {"go", "stop"}
    print("smoke OK")
