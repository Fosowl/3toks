"""Casual agent: small talk answered from a small canned-reply bank.

Small talk is not worth a generation. The harness ranks predefined
replies by hint-keyword overlap with the user's message, and:

- if a hint matches the whole normalized input, returns that reply with
  ZERO model calls (deterministic shortcut);
- otherwise offers the top candidates as a numbered menu (one decision),
  mapping the chosen option back to its FULL reply text;
- and only falls through to a bounded free reply when nothing fits.

Classification over generation (docs/DESIGN.md §1) all the way down.
"""
import json
import re
from pathlib import Path

from threetoks.agents.base import AgentSpec
from threetoks.nodes import ESCAPE, MenuNode, ShortTextNode
from threetoks.render import Episode

AGENT_NAME = "casual"
AGENT_DESCRIPTION = "chat, greetings, thanks, and other small talk"

REPLIES_PATH = Path(__file__).with_name("casual_replies.json")
# Safety net when the reply bank is missing or unreadable (e.g. a wheel
# built without package data): small talk must degrade, never crash.
BUILTIN_REPLIES = (
    {"reply": "Hello! How can I help you today?",
     "hints": ["hello", "hi", "hey"]},
    {"reply": "You're welcome! Happy to help.",
     "hints": ["thanks", "thank you"]},
    {"reply": "Goodbye! Come back anytime.",
     "hints": ["bye", "goodbye", "see you"]},
)
MENU_CANDIDATES = 5
OPTION_CHARS = 60
FREE_REPLY_MAX_TOKENS = 48
FREE_REPLY_PREFILL = "REPLY:"
FALLBACK_REPLY = "I'm not sure how to respond to that."
_MENU_BLEED_RE = re.compile(r"\d+(\s*,\s*\d+)*\s*,?")  # "1" / "1,2" is no reply

PREFIX = """You pick the best short reply to a casual message.
Example:
ACTIONS:
1 = Hello! How can I help you today?
2 = Goodbye! Come back anytime.
Reply with exactly ONE digit.
ANSWER: 1"""


def _load_replies() -> list[dict]:
    """Load the canned reply bank; a broken install gets the built-ins."""
    try:
        return json.loads(REPLIES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return list(BUILTIN_REPLIES)


def _normalize(text: str) -> str:
    """Lowercase and collapse surrounding whitespace for matching."""
    return " ".join(text.lower().split())


def _exact_reply(normalized: str, replies: list[dict]) -> str | None:
    """Reply whose hint equals the whole message, else None (zero-cost)."""
    for entry in replies:
        if any(_normalize(hint) == normalized for hint in entry["hints"]):
            return entry["reply"]
    return None


def _score(normalized: str, entry: dict) -> int:
    """Count hints appearing as substrings of the normalized message."""
    return sum(1 for hint in entry["hints"] if _normalize(hint) in normalized)


def _ranked_replies(normalized: str, replies: list[dict]) -> list[str]:
    """Reply texts with at least one hint hit, best overlap first."""
    scored = [(_score(normalized, entry), entry["reply"]) for entry in replies]
    hits = sorted((s for s in scored if s[0] > 0), key=lambda s: -s[0])
    return [reply for _, reply in hits[:MENU_CANDIDATES]]


def _pick_from_menu(task: str, candidates: list[str], policy) -> str | None:
    """One menu decision; map the truncated option back to the full reply."""
    option_map = {reply[:OPTION_CHARS]: reply for reply in candidates}
    episode = Episode(PREFIX, task)
    episode.open_observation(f"MESSAGE:\n{task}")
    node = MenuNode("Which reply fits best?", list(option_map), escape=True)
    decision = policy.decide(episode, node)
    if decision.valid and decision.value in option_map:
        return option_map[decision.value]
    return None  # escape or invalid -> caller writes a free reply


def _free_reply(task: str, policy) -> str:
    """Bounded free-text reply when no canned answer fits.

    A bare digit or index list is menu-format bleed, not a reply — it
    falls back rather than being shown to the user.
    """
    episode = Episode(PREFIX, task)
    episode.open_observation(f"MESSAGE:\n{task}")
    node = ShortTextNode(
        "Reply to this casual message in one short, friendly sentence.",
        FREE_REPLY_PREFILL, max_tokens=FREE_REPLY_MAX_TOKENS)
    decision = policy.decide(episode, node)
    value = decision.value if decision.valid else ""
    if not value or _MENU_BLEED_RE.fullmatch(value.strip()):
        return FALLBACK_REPLY
    return value


def run(task: str, services, policy) -> dict:
    """Answer small talk: exact shortcut, then menu, then free reply."""
    replies = _load_replies()
    normalized = _normalize(task)
    exact = _exact_reply(normalized, replies)
    if exact is not None:
        return {"answer": exact, "agent": AGENT_NAME}
    candidates = _ranked_replies(normalized, replies)
    chosen = _pick_from_menu(task, candidates, policy) if candidates else None
    reply = chosen if chosen is not None else _free_reply(task, policy)
    return {"answer": reply, "agent": AGENT_NAME}


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    class _Backend:
        def __init__(self, text):
            self.text, self.calls = text, 0

        def complete(self, model, raw_prompt, opts):
            self.calls += 1
            return GenResult(self.text, 5, 2, 0.0, "stop")

    shortcut_backend = _Backend(" 1")
    policy = Policy(shortcut_backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    out = SPEC.run("hello", None, policy)
    assert out == {"answer": "Hello! How can I help you today?",
                   "agent": "casual"}, out
    assert shortcut_backend.calls == 0, "exact hint must skip the model"
    menu = SPEC.run("hi there friend", None, _pick_policy := policy)
    assert menu["agent"] == "casual" and menu["answer"], menu
    assert ESCAPE  # referenced so the escape constant stays imported
    print("smoke OK")
