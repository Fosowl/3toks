"""Light agent: switch the desk lamp or report whether it is on.

A regex settles clearly-phrased requests with ZERO model calls — "turn on
the light", "lights off", "toggle the lamp", "is the light on?". Only a
request that matches none of those spends ONE MenuNode decision. The relay
lives on ``services.relay``; without it the agent says so.

Classification over generation (docs/AGENTS.md rule 7): never spend a model
call where a word-boundary regex settles the intent.
"""
import re

from threetoks.agents.base import AgentSpec
from threetoks.nodes import ESCAPE, MenuNode
from threetoks.render import Episode

AGENT_NAME = "light"
AGENT_DESCRIPTION = "switch the desk lamp on or off, or check whether it is on"

NO_RELAY = "No relay is connected, so I can't control the light."
CONFUSED = "I didn't understand what to do with the light."

# Word-boundary patterns. Only the STATUS-FIRST ordering is load-bearing:
# questions about state ("is it on?", "are the lights off?") contain
# on/off words and must never DRIVE the relay. The off/on/toggle order
# is not load-bearing.
_STATUS_RE = re.compile(r"\bstatus\b|\bis the (?:light|lamp)\b"
                        r"|\b(?:is|are)\b[\w\s]*\b(?:on|off)\b")
_OFF_RE = re.compile(r"\boff\b")
_ON_RE = re.compile(r"\bon\b")
_TOGGLE_RE = re.compile(r"\b(?:toggle|switch|flip)\b")

OPT_ON = "turn the light on"
OPT_OFF = "turn the light off"
OPT_TOGGLE = "toggle the light"
OPT_STATUS = "say whether the light is on"

PREFIX = """You pick one light action for an unclear request.
Example:
ACTIONS:
1 = turn the light on
2 = turn the light off
Reply with exactly ONE digit.
ANSWER: 1"""


def _changed(state: str) -> str:
    """Spoken result after switching the lamp."""
    return f"The light is now {state}."


def _reported(state: str) -> str:
    """Spoken current lamp state, without claiming a change."""
    return f"The light is {state}."


def _deterministic(task: str, relay) -> str | None:
    """Answer a clearly-phrased command with zero model calls, else None."""
    text = task.lower()
    if _STATUS_RE.search(text):
        return _reported(relay.get_status())
    if _OFF_RE.search(text):
        return _changed(relay.set_light(False))
    if _ON_RE.search(text):
        return _changed(relay.set_light(True))
    if _TOGGLE_RE.search(text):
        return _changed(relay.toggle())
    return None


def _apply(choice: str, relay) -> str:
    """Run the chosen menu action and speak the result."""
    if choice == OPT_ON:
        return _changed(relay.set_light(True))
    if choice == OPT_OFF:
        return _changed(relay.set_light(False))
    if choice == OPT_TOGGLE:
        return _changed(relay.toggle())
    return _reported(relay.get_status())


def _from_menu(task: str, relay, policy) -> str:
    """One MenuNode decision maps an unclear request to an action."""
    episode = Episode(PREFIX, task)
    episode.open_observation(f"REQUEST:\n{task}")
    node = MenuNode("What should I do with the light?",
                    [OPT_ON, OPT_OFF, OPT_TOGGLE, OPT_STATUS], escape=True)
    decision = policy.decide(episode, node)
    if not decision.valid or decision.value == ESCAPE:
        return CONFUSED
    return _apply(decision.value, relay)


def run(task: str, services, policy) -> dict:
    """Control the lamp: guard, deterministic match, then one menu."""
    relay = getattr(services, "relay", None)
    if relay is None:
        return {"answer": NO_RELAY, "agent": AGENT_NAME}
    answer = _deterministic(task, relay) or _from_menu(task, relay, policy)
    return {"answer": answer, "agent": AGENT_NAME}


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)


if __name__ == "__main__":
    class _StubRelay:
        """Records nothing; just tracks state for the smoke test."""

        def __init__(self):
            self.on = False

        def set_light(self, on):
            self.on = bool(on)
            return "on" if self.on else "off"

        def get_status(self):
            return "on" if self.on else "off"

        def toggle(self):
            self.on = not self.on
            return "on" if self.on else "off"

    def _services(relay):
        return type("S", (), {"relay": relay})()

    relay = _StubRelay()
    services = _services(relay)
    # Deterministic paths pass policy=None: a regex must never touch a model.
    assert run("turn on the light", services, None)["answer"] == _changed("on")
    assert relay.on is True
    assert run("lights off please", services, None)["answer"] == _changed("off")
    assert run("is the light on?", services, None)["answer"] == _reported("off")
    assert run("toggle the lamp", services, None)["answer"] == _changed("on")
    assert run("anything", _services(None), None)["answer"] == NO_RELAY
    print("smoke OK")
