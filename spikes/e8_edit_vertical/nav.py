"""Tree navigation: folder -> file -> def, for when no symbol is named.

Built entirely on threetoks' real MenuNode (nodes.py) -- no bespoke node
type. Menus have <= 9 options including escape (load-bearing rule), so a
level with more entries than fit is paged: the harness reserves the last
slot for "show more options" rather than widening the menu (research.py
already does this for search-result pages; this mirrors it).

A level with zero or one entry never asks the model at all -- a one-option
"menu" is a decision with only one possible answer, so the harness takes it
for free (the same free-take rule threetoks/research.py applies to a
single-result menu).
"""
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_DIR.parent.parent
for _p in (str(SPIKE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from threetoks.nodes import ESCAPE, MenuNode  # noqa: E402

PAGE_SIZE = 8            # MENU_MAX_OPTIONS(9) - 1 for escape
MORE_LABEL = "show more options"


def paginate(items: list[str], page: int) -> tuple[list[str], bool]:
    """The slice of items to show on `page`, and whether more pages remain."""
    per_page = PAGE_SIZE - 1     # reserve one slot for "show more"
    start = page * per_page
    remaining = items[start:]
    if len(remaining) <= PAGE_SIZE:
        return remaining, False
    return remaining[:per_page], True


def level_menu(question: str, items: list[str], page: int) -> MenuNode | None:
    """A MenuNode for this page of items, or None when no menu is needed
    (0 items: nothing to pick; 1 item: free-take, caller auto-selects it).
    """
    if len(items) <= 1:
        return None
    shown, has_more = paginate(items, page)
    options = list(shown) + ([MORE_LABEL] if has_more else [])
    return MenuNode(question, options, escape=True)


class LevelChoice:
    """One paged navigation level; wraps level_menu with page-tracking.

    Usage: build with the full item list; call `node()` for the current
    page's MenuNode (or None if free-take applies); call `apply(decision)`
    with the model's answer -- returns the chosen item, MORE_LABEL (stay on
    this level, next page), or None (escape / nothing usable).
    """

    def __init__(self, question: str, items: list[str]):
        self.question = question
        self.items = items
        self.page = 0

    def free_take(self) -> str | None:
        """The single item when no menu is needed, else None."""
        if len(self.items) == 1:
            return self.items[0]
        if not self.items:
            return None
        return None

    def node(self) -> MenuNode | None:
        return level_menu(self.question, self.items, self.page)

    def apply(self, decision) -> str | None:
        """Advance the page on "more"; otherwise return the pick or None."""
        if not decision.valid or decision.value in (ESCAPE, None):
            return None
        if decision.value == MORE_LABEL:
            self.page += 1
            return MORE_LABEL
        return decision.value


def list_folders(root: Path) -> list[str]:
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.rglob("*.py")):
            out.append(child.name)
    return out


def list_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()
                 if p.is_file() and p.suffix == ".py")


if __name__ == "__main__":
    from threetoks.nodes import Decision

    # zero/one item: free, no menu
    assert level_menu("pick", [], 0) is None
    assert level_menu("pick", ["only"], 0) is None

    # a handful of items: one ordinary menu, no paging
    menu = level_menu("pick a folder", ["a", "b", "c"], 0)
    assert menu is not None and len(menu.options) == 3

    # more items than fit: paged, "show more" occupies the last real slot
    many = [f"item{i}" for i in range(12)]
    first_page = level_menu("pick", many, 0)
    assert MORE_LABEL in first_page.options
    assert len(first_page.options) == PAGE_SIZE   # 7 items + "show more"
    second_page = level_menu("pick", many, 1)
    assert MORE_LABEL not in second_page.options  # remaining 5 items fit

    choice = LevelChoice("pick", many)
    node = choice.node()
    perm = tuple(range(len(node.options)))
    more_idx = node.options.index(MORE_LABEL) + 1
    picked = choice.apply(node.parse(f" {more_idx}", perm))
    assert picked == MORE_LABEL and choice.page == 1
    node2 = choice.node()
    assert MORE_LABEL not in node2.options

    esc = LevelChoice("pick", ["x", "y"])
    n = esc.node()
    p = tuple(range(len(n.options)))
    escape_digit = len(n.options) + 1
    assert esc.apply(n.parse(f" {escape_digit}", p)) is None
    print("smoke OK")
