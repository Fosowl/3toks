"""Paged menus over a flat item list, and their one-line labels.

Built entirely on the core MenuNode. Menus have <= 9 options including
escape (load-bearing rule), so a list with more entries is paged: the
last real slot becomes "show more options" rather than widening the menu.
A list with zero or one entry never asks the model at all — the harness
takes the only possible answer for free. Labels double as lookup keys,
so they are clipped to one line and made unique.
"""
from threetoks.nodes import ESCAPE, MenuNode

PAGE_SIZE = 8            # MENU_MAX_OPTIONS(9) - 1 for escape
MORE_LABEL = "show more options"
LABEL_CLIP = 72
MIN_DOC_CHARS = 12      # less doc room than this: no doc
ELLIPSIS = "…"


def paginate(items: list[str], page: int) -> tuple[list[str], bool]:
    """The slice of items shown on ``page``, and whether more remain."""
    per_page = PAGE_SIZE - 1     # reserve one slot for "show more"
    remaining = items[page * per_page:]
    if len(remaining) <= PAGE_SIZE:
        return remaining, False
    return remaining[:per_page], True


def level_menu(question: str, items: list[str], page: int) -> MenuNode | None:
    """A MenuNode for this page, or None when no menu is needed (0 or 1
    item: nothing to ask; the caller free-takes or gives up)."""
    if len(items) <= 1:
        return None
    shown, has_more = paginate(items, page)
    options = list(shown) + ([MORE_LABEL] if has_more else [])
    return MenuNode(question, options, escape=True)


class LevelChoice:
    """One paged navigation level over a fixed item list.

    ``node()`` returns the current page's MenuNode (None when free-take
    applies); ``apply(decision)`` returns the chosen item, MORE_LABEL
    (advance a page, stay on this level), or None (escape / nothing).
    """

    def __init__(self, question: str, items: list[str]):
        self.question = question
        self.items = items
        self.page = 0

    def free_take(self) -> str | None:
        """The single item when no menu is needed, else None."""
        return self.items[0] if len(self.items) == 1 else None

    def node(self) -> MenuNode | None:
        """The menu for the current page, or None for a free level."""
        return level_menu(self.question, self.items, self.page)

    def apply(self, decision) -> str | None:
        """Advance the page on "more"; otherwise the pick or None."""
        if not decision.valid or decision.value in (ESCAPE, None):
            return None
        if decision.value == MORE_LABEL:
            self.page += 1
            return MORE_LABEL
        return decision.value


def symbol_label(symbol, with_kind: bool = False) -> str:
    """Build the menu label of one symbol: [kind] qualname — path:line (doc).

    The edit vertical (shortlist) and the code agent (navigate menu) call
    this for each candidate; it is the only Symbol label builder. The
    line number tells same-named defs apart (property setter, overload).
    The path is clipped from the LEFT so the file name survives; the doc
    gets the room that is left.

    Args:
        symbol: index.Symbol, one candidate (kind, qualname, file, lineno,
            doc).
        with_kind: bool, True to prefix "class ", "function " or "method ".
    Returns:
        str: a one-line label of about LABEL_CLIP characters.
    """
    kind = f"{symbol.kind} " if with_kind else ""
    head = f"{kind}{symbol.qualname} — "
    where = f":{symbol.lineno}"
    path = _clip_left(symbol.file, LABEL_CLIP - len(head) - len(where))
    label = head + path + where
    room = LABEL_CLIP - len(label) - len(" ()")
    doc = " ".join(symbol.doc.split())
    if doc and room >= MIN_DOC_CHARS:
        label += f" ({clip(doc, room)})"
    return label


def _clip_left(path: str, limit: int) -> str:
    """Shorten a path from the left; the file name always survives.

    Only symbol_label calls this.

    Args:
        path: str, a posix path from Symbol.file.
        limit: int, the wanted maximum length.
    Returns:
        str: the path, or ELLIPSIS plus its tail.
    """
    if len(path) <= limit:
        return path
    keep = max(limit - len(ELLIPSIS), len(path.rsplit("/", 1)[-1]))
    return ELLIPSIS + path[-keep:]


def clip(text: str, limit: int = LABEL_CLIP) -> str:
    """Make text one line and cut it on the right.

    symbol_label and the edit vertical's delete menu call this.

    Args:
        text: str, any label text.
        limit: int, the maximum length.
    Returns:
        str: one line of at most ``limit`` characters.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + ELLIPSIS


def clip_lines(lines: list[str], limit: int) -> tuple[list[str], int]:
    """Keep the first lines of a text and count the rest.

    The edit vertical, the code agent, and the TUI call this before they
    show code, diffs, or output; each caller writes its own "N more
    lines" note from the count.

    Args:
        lines: list[str], the lines to show, in order.
        limit: int, the maximum number of lines to keep.
    Returns:
        tuple[list[str], int]: (kept lines, number of hidden lines).
    """
    shown = lines[:max(0, limit)]
    return shown, len(lines) - len(shown)


def uniquify(labels: list[str]) -> list[str]:
    """Deduplicate menu labels that clipping made identical.

    The edit vertical calls this before it builds a LevelChoice, because
    labels double as lookup keys back to the picked item.

    Args:
        labels: list[str], labels in menu order.
    Returns:
        list[str]: unique labels (counter suffix on repeats), same order.
    """
    counts: dict[str, int] = {}
    out = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        out.append(label if counts[label] == 1
                   else f"{label} ({counts[label]})")
    return out


if __name__ == "__main__":
    assert level_menu("pick", [], 0) is None
    assert level_menu("pick", ["only"], 0) is None
    menu = level_menu("pick a function", ["a", "b", "c"], 0)
    assert menu is not None and len(menu.options) == 3

    many = [f"item{i}" for i in range(12)]
    first = level_menu("pick", many, 0)
    assert MORE_LABEL in first.options and len(first.options) == PAGE_SIZE
    assert MORE_LABEL not in level_menu("pick", many, 1).options

    choice = LevelChoice("pick", many)
    node = choice.node()
    perm = tuple(range(len(node.options)))
    more_digit = node.options.index(MORE_LABEL) + 1
    assert choice.apply(node.parse(f" {more_digit}", perm)) == MORE_LABEL
    assert choice.page == 1

    lone = LevelChoice("pick", ["solo"])
    assert lone.node() is None and lone.free_take() == "solo"

    pair = LevelChoice("pick", ["x", "y"])
    n = pair.node()
    p = tuple(range(len(n.options)))
    assert pair.apply(n.parse(f" {len(n.options) + 1}", p)) is None  # escape

    from types import SimpleNamespace
    deep = SimpleNamespace(qualname="convert", doc="", kind="function",
                           lineno=12,
                           file="a/very/long/package/path/" * 3 + "money.py")
    assert symbol_label(deep).endswith("money.py:12")
    assert len(symbol_label(deep)) <= LABEL_CLIP
    assert symbol_label(deep, with_kind=True).startswith("function convert")
    assert clip_lines(["a", "b", "c"], 2) == (["a", "b"], 1)
    assert uniquify(["x", "x"]) == ["x", "x (2)"]
    print("smoke OK")
