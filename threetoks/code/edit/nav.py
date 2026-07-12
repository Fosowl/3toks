"""Tree navigation: folder -> file -> def, for when no symbol is named.

Built entirely on the core MenuNode. Menus have <= 9 options including
escape (load-bearing rule), so a level with more entries is paged: the
last real slot becomes "show more options" rather than widening the menu.
A level with zero or one entry never asks the model at all — the harness
takes the only possible answer for free.
"""
from pathlib import Path

from threetoks.nodes import ESCAPE, MenuNode

PAGE_SIZE = 8            # MENU_MAX_OPTIONS(9) - 1 for escape
MORE_LABEL = "show more options"


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


def list_folders(root: Path) -> list[str]:
    """Immediate subdirectories of root containing at least one .py file."""
    return sorted(child.name for child in root.iterdir()
                  if child.is_dir() and any(child.rglob("*.py")))


def list_files(folder: Path) -> list[str]:
    """.py filenames directly inside one folder, sorted."""
    return sorted(p.name for p in folder.iterdir()
                  if p.is_file() and p.suffix == ".py")


if __name__ == "__main__":
    assert level_menu("pick", [], 0) is None
    assert level_menu("pick", ["only"], 0) is None
    menu = level_menu("pick a folder", ["a", "b", "c"], 0)
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
    print("smoke OK")
