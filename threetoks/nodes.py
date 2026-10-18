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
SEQUENCE_PREFILL = "PLAN:"
SEQUENCE_PUNCTUATION = ".;:()[]-"  # stripped around each number token


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

    An optional ``tag`` attribute (e.g. ``node.tag = "judge"``) names the
    node in traces and in the TUI progress tree. It also changes the tree
    shape: a menu WITHOUT a tag is an action, and the decisions after it
    nest under it (see :mod:`threetoks.progress`). Tag every menu that
    is a check or a classification, not a step the agent takes.
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


class SequenceNode:
    """Ordered list of item numbers; one item can come back more than once.

    The planner agent uses this node to put agents in order for a plan.
    The planner calls it with the number of agents it shows in the prompt.
    The model gives numbers in order, so "2,3,2" means agent 2, then
    agent 3, then agent 2 again. Order is kept and repeats are allowed;
    numbers out of range are dropped, and only the first ``max_len``
    numbers of the first line are kept.
    """

    kind = "sequence"
    prefill = SEQUENCE_PREFILL
    max_tokens = 16
    stop = ("\n",)

    def __init__(self, question: str, n_items: int, max_len: int,
                 items: list[str] | None = None):
        """Store the question and the limits of the list.

        Args:
            question (str): the line shown above the reply rule, from the
                planner.
            n_items (int): how many numbered items the prompt shows.
            max_len (int): the most numbers to keep (the step limit).
            items (list[str] | None): item texts, for the trace only.
        """
        self.question = question
        self.n_items = n_items
        self.max_len = max_len
        self.items = items

    def render(self, perm: tuple[int, ...]) -> str:
        """Give the question and the reply rule; ``perm`` is not used.

        Args:
            perm (tuple[int, ...]): option order from the policy (empty).

        Returns:
            str: the node text that goes after the episode prompt.
        """
        return (f"{self.question}\n"
                f"Reply with at most {self.max_len} numbers in order, "
                "separated by commas (e.g. 2,1).")

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Read the numbers on the first line, in the order given.

        Args:
            text (str): the model completion, from the policy.
            perm (tuple[int, ...]): option order (not used).

        Returns:
            Decision: value is a list[int] of 1-based item numbers; the
            decision is valid when at least one number is in range.
        """
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        tokens = first_line.replace(",", " ").split()
        numbers = [token.strip(SEQUENCE_PUNCTUATION) for token in tokens]
        picked = [int(number) for number in numbers if number.isdecimal()
                  and 1 <= int(number) <= self.n_items][:self.max_len]
        return Decision(self.kind, picked, text, valid=bool(picked))


# End/turn markers any supported family might leak into a completion
# (raw-mode models sometimes emit their own end token inline).
TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "</s>",
                    "<end_of_turn>", "<|eot_id|>", "<|end|>",
                    "<｜end▁of▁sentence｜>")


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
    sequence = SequenceNode("Which?", 3, max_len=3)
    assert sequence.parse(" 2, 3, 2, 9, 1", ()).value == [2, 3, 2]
    assert not sequence.parse("none", ()).valid
    short = ShortTextNode("Query?", "QUERY:")
    assert short.parse(" france population \n junk", ()).value == "france population"
    assert short.parse("1<|im_end|>", ()).value == "1"
    assert confirm_node("Sure?").parse("2", (0, 1)).value == "no"
    print("smoke OK")
