"""Decision nodes: the only shapes in which the model is ever consulted.

Every node renders itself as text the harness appends to the episode
prompt, and parses the model's (tiny) completion back into a Decision.
Menu options are permuted per attempt to defeat position bias; parsing
maps digits back through the permutation to canonical option text.
"""
import re
from dataclasses import dataclass

ESCAPE = "<none>"
ESCAPE_LABEL = "none of these fit — do something else"
MENU_MAX_OPTIONS = 9
MENU_PREFILL = "ANSWER:"
PICK_PREFILL = "RELEVANT:"


@dataclass(frozen=True)
class Decision:
    """Parsed model output for one node."""
    kind: str
    value: object
    raw_text: str
    valid: bool


def _parse_first_digit(text: str, n_options: int) -> int | None:
    """First digit in text if within 0..n_options, else None."""
    for char in text:
        if char.isdigit():
            value = int(char)
            return value if value <= n_options else None
    return None


class MenuNode:
    """Single-choice menu answered with one digit.

    The escape ("none of these fit") is rendered as an ordinary numbered
    option in the LAST position, not as "0": E1/E4 showed tiny models
    almost never emit 0 for a menu (out-of-distribution), which made the
    escape blind. Digit 0 is always an invalid parse.
    """

    kind = "menu"
    prefill = MENU_PREFILL
    max_tokens = 3

    def __init__(self, question: str, options: list[str], escape: bool = True):
        max_own = MENU_MAX_OPTIONS - (1 if escape else 0)
        if not 1 <= len(options) <= max_own:
            raise ValueError(f"menu needs 1..{max_own} options")
        self.question = question
        self.options = list(options)
        self.escape = escape

    def render(self, perm: tuple[int, ...]) -> str:
        """Render the numbered menu with options in permuted order."""
        lines = [self.question, "ACTIONS:"]
        for shown, canonical in enumerate(perm, start=1):
            lines.append(f"{shown} = {self.options[canonical]}")
        if self.escape:
            lines.append(f"{len(perm) + 1} = {ESCAPE_LABEL}")
        lines.append("Reply with exactly ONE digit.")
        return "\n".join(lines)

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Map the first digit back to canonical option text."""
        escape_digit = len(perm) + 1 if self.escape else None
        digit = _parse_first_digit(text, escape_digit or len(perm))
        if not digit:
            return Decision(self.kind, None, text, valid=False)
        if digit == escape_digit:
            return Decision(self.kind, ESCAPE, text, valid=True)
        return Decision(self.kind, self.options[perm[digit - 1]], text,
                        valid=True)


class PickManyNode:
    """Select several numbered items (e.g. sentences worth noting).

    Defaults tuned by spike E3: ~24-token cap with a newline stop lets
    short lists finish cleanly, and max_picks caps over-picking (tiny
    models fill whatever budget they get; gold is rarely >3 items).
    """

    kind = "pick_many"
    prefill = PICK_PREFILL
    max_tokens = 24
    stop = ("\n",)

    def __init__(self, question: str, n_items: int, max_picks: int = 4,
                 items: list[str] | None = None):
        self.question = question
        self.n_items = n_items
        self.max_picks = max_picks
        self.items = items  # texts the indices point to (for tracing)

    def render(self, perm: tuple[int, ...]) -> str:
        """Items are shown elsewhere in canonical order; perm is unused."""
        return (f"{self.question}\n"
                "Reply with the FEWEST item numbers that answer, "
                "separated by commas (e.g. 3,7).")

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Parse indices from the first line: in-range, deduped, capped."""
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        picked: list[int] = []
        for token in re.findall(r"\d+", first_line):
            value = int(token)
            if 1 <= value <= self.n_items and value not in picked:
                picked.append(value)
        picked = picked[:self.max_picks]
        return Decision(self.kind, picked, text, valid=bool(picked))


TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "</s>")


class ShortTextNode:
    """Bounded free text: search queries, form fields, final answers."""

    kind = "short_text"
    prefill = ""
    stop = ("\n",)

    def __init__(self, question: str, prefill: str, max_tokens: int = 24):
        self.question = question
        self.prefill = prefill
        self.max_tokens = max_tokens

    def render(self, perm: tuple[int, ...]) -> str:
        """Perm is unused; free text has no options."""
        return self.question

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Accept any non-empty first line, chat-template markers removed.

        Raw-mode models sometimes emit their own end marker inline
        ("1<|im_end|>"); those must never leak into an answer.
        """
        for marker in TEMPLATE_MARKERS:
            text = text.replace(marker, "")
        value = text.strip().splitlines()[0].strip() if text.strip() else ""
        return Decision(self.kind, value, text, valid=bool(value))


def confirm_node(question: str) -> MenuNode:
    """Yes/no as a two-option menu without escape."""
    return MenuNode(question, ["yes", "no"], escape=False)


if __name__ == "__main__":
    menu = MenuNode("Pick.", ["alpha", "beta", "gamma"])
    perm = (2, 0, 1)
    assert "1 = gamma" in menu.render(perm)
    assert menu.parse(" 1", perm).value == "gamma"
    assert menu.parse(" 4", perm).value == ESCAPE
    assert not menu.parse(" 0", perm).valid
    assert not menu.parse("x", perm).valid
    pick = PickManyNode("Which?", 10)
    assert pick.parse(" 3, 7, 12", ()).value == [3, 7]
    short = ShortTextNode("Query?", "QUERY:")
    assert short.parse(" france population \n junk", ()).value == "france population"
    assert short.parse("1<|im_end|>", ()).value == "1"
    assert confirm_node("Sure?").parse("2", (0, 1)).value == "no"
    print("smoke OK")
