"""The plug-and-play agent contract.

An agent is a named capability with a one-line description (shown to the
router model as a menu option) and a run function. Adding an agent =
one module exposing a SPEC; the registry in threetoks/agents/__init__.py
lists it.
"""
from collections.abc import Callable
from dataclasses import dataclass

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


def outcome_status(outcome: dict) -> str:
    """Name the state of one agent result in a word or two.

    The TUI and the planner put this on the end row of an agent branch
    in the progress tree. They call it with the result dict of one run.

    Args:
        outcome (dict): the agent result, with "answer", an optional
            "judged_good" (bool) from a judge, and an optional "ran"
            (bool) from the code agent's shell mode.

    Returns:
        str: "not run" when a shell command did not run, "unverified"
        when a judge or a failed run rejected it, "no answer" when the
        answer is empty, else "answered".
    """
    if outcome.get("ran") is False:
        return "not run"
    if outcome.get("judged_good") is False:
        return "unverified"
    answer = str(outcome.get("answer") or "").strip()
    return "answered" if answer and answer != "(no answer)" else "no answer"


if __name__ == "__main__":
    spec = AgentSpec("echo", "repeats the task",
                     lambda task, services, policy: {"answer": task})
    result = spec.run("hello", Services(), None)
    assert result["answer"] == "hello"
    assert outcome_status(result) == "answered"
    assert outcome_status({"answer": "x", "judged_good": False}) == "unverified"
    assert outcome_status({"answer": "(no answer)"}) == "no answer"
    assert outcome_status({"answer": "x", "ran": False,
                           "judged_good": False}) == "not run"
    print("smoke OK")
