"""The plug-and-play agent contract.

An agent is a named capability with a one-line description (shown to the
router model as a menu option) and a run function. Adding an agent =
one module exposing a SPEC; the registry in threetoks/agents/__init__.py
lists it.
"""
from dataclasses import dataclass
from typing import Callable

from threetoks.policy import Policy
from threetoks.services import Services


@dataclass(frozen=True)
class AgentSpec:
    """One pluggable agent.

    run(task, services, policy) returns a result dict with at least
    {"answer": str}; optional keys: "notes", "agent", "rounds".
    """
    name: str
    description: str
    run: Callable[[str, Services, Policy], dict]


if __name__ == "__main__":
    spec = AgentSpec("echo", "repeats the task",
                     lambda task, services, policy: {"answer": task})
    result = spec.run("hello", Services(), None)
    assert result["answer"] == "hello"
    print("smoke OK")
