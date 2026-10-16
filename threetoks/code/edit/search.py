"""Find and confirm the edit target: shortlist pick + confirm by pointing.

The edit vertical delegates the LOCATE and CONFIRM phases to one
TargetSearch. A unique exact function/method name in the request skips
the pick menu; otherwise rank.shortlist orders the corpus symbols and
the model picks one from ONE paged menu (a lone candidate is free). Every
target is then confirmed: its source becomes the menu observation and
one menu asks "here / in a callee (hop, bounded) / somewhere else".
"Somewhere else" strikes the pick and its hops and goes back to the
shortlist. The search ends ACCEPTED (``target`` is the confirmed symbol)
or FAILED (``failure`` says why).
"""
from dataclasses import dataclass, field

from threetoks.code.edit import nav, rank
from threetoks.code.edit.spans import dedent_block, leading_indent, span_text
from threetoks.nodes import MenuNode

MAX_CALLEE_OPTIONS = 6      # yes + 6 callees + elsewhere <= 9
MAX_CONFIRM_HOPS = 2        # callee hops before a target is accepted
CONFIRM_MAX_LINES = 60      # target source lines shown to menus
PICK_QUESTION = "Which function must change to do what the task asks?"
CONFIRM_YES = "yes — the problem is in this function"
CONFIRM_ELSEWHERE = "no — it is somewhere else"
CALLEE_LABEL = "no — it is in {name}() which this function calls"
NO_MATCH = "no function matches the request"
NO_FIT = "none of the shortlisted functions fits the request"
NOT_CONFIRMED = "could not confirm the target"
NO_REMAINING = "no remaining function matches the request"
PICK, CONFIRM, ACCEPTED, FAILED = "pick", "confirm", "accepted", "failed"


@dataclass
class _Menus:
    """The open pick menu and the label lookups of the current menus."""
    pick: nav.LevelChoice | None = None
    labels: dict = field(default_factory=dict)   # pick label -> Symbol
    callees: dict = field(default_factory=dict)  # confirm label -> Symbol


class TargetSearch:
    """The locate + confirm state machine of one edit episode.

    The vertical calls ``start()`` once, then ``node()`` / ``apply()``
    while ``state`` is PICK or CONFIRM, and ``strike(why)`` when a
    confirmed target turns out wrong. Public data: ``state``, ``target``,
    ``path_taken``, ``failure``.
    """

    def __init__(self, index, request: str, episode):
        """Keep the corpus, the request, and the menu episode.

        The edit vertical creates one search per episode.

        Args:
            index: SymbolIndex, the corpus index.
            request: str, the user's request text.
            episode: Episode, the menu episode (observation + session log).
        """
        self.index = index
        self.request = request
        self.episode = episode
        self.state = PICK
        self.target = None
        self.path_taken = ""
        self.failure = ""
        self._shortlist: list | None = None   # ranked once, lazily
        self._struck: set = set()
        self._chain: list = []                # pick, then callee hops
        self._menus = _Menus()

    def start(self) -> None:
        """Confirm a unique exact name, else go to the shortlist.

        The vertical calls this once, after its free precheck. A unique
        exact name skips the pick menu but is still confirmed: a request
        word can be a common function name ("update", "run") by chance.
        """
        hits = rank.exact_hits(self.index, self.request)
        if len(hits) == 1:
            self.path_taken = "direct"
            self._choose(hits[0])
            return
        self._back_to_shortlist(NO_MATCH)

    def strike(self, why: str) -> None:
        """Drop the current target and go back to the shortlist.

        The vertical calls this when the model found nothing to change or
        the repair ladder gave up on the target.

        Args:
            why: str, the failure reason when no candidate remains.
        """
        self._struck.add(self.target)
        self._back_to_shortlist(why)

    def node(self):
        """Give the next pick or confirm menu.

        A lone remaining candidate is taken for free, so the confirm menu
        follows directly.

        Returns:
            MenuNode | None: None when the search is ACCEPTED or FAILED.
        """
        if self.state == PICK:
            remaining = self._remaining()
            if len(remaining) == 1:
                self._choose(remaining[0])
            else:
                return self._pick_node(remaining)
        if self.state == CONFIRM:
            return self._confirm_node()
        return None

    def apply(self, decision) -> None:
        """Apply the model's answer to the current menu.

        Args:
            decision: Decision from the Policy for the last node().
        """
        if self.state == PICK:
            self._apply_pick(decision)
        elif self.state == CONFIRM:
            self._apply_confirm(decision)

    def _remaining(self) -> list:
        """Give the shortlist candidates that are not struck yet.

        Returns:
            list[Symbol]: candidates, best first.
        """
        if self._shortlist is None:
            self._shortlist = rank.shortlist(self.index, self.request)
        return [s for s in self._shortlist if s not in self._struck]

    def _back_to_shortlist(self, empty_reason: str) -> None:
        """Show the shortlist menu again, or fail when it is empty.

        Args:
            empty_reason: str, the failure reason when no candidate remains.
        """
        if not self._remaining():
            self._fail(empty_reason)
            return
        self.path_taken = "shortlist"
        self._menus.pick = None
        self.episode.open_observation("")   # struck source leaves view
        self.state = PICK

    def _pick_node(self, remaining: list) -> MenuNode | None:
        """Give the current page of the shortlist menu.

        Args:
            remaining: list[Symbol], the unstruck candidates (two or more).
        Returns:
            MenuNode | None: the page.
        """
        if self._menus.pick is None:
            labels = nav.uniquify([nav.symbol_label(s) for s in remaining])
            self._menus.labels = dict(zip(labels, remaining))
            self._menus.pick = nav.LevelChoice(PICK_QUESTION, labels)
        return self._menus.pick.node()

    def _apply_pick(self, decision) -> None:
        """Take the picked candidate, page on "more", fail on escape.

        Args:
            decision: Decision from the Policy.
        """
        picked = self._menus.pick.apply(decision)
        if picked is None:
            self._fail(NO_FIT)
        elif picked != nav.MORE_LABEL:
            self._choose(self._menus.labels[picked])

    def _choose(self, symbol) -> None:
        """Start a new confirm chain on one candidate.

        Args:
            symbol: Symbol, the picked candidate.
        """
        self._chain = []
        self._show(symbol)
        self.state = CONFIRM

    def _show(self, symbol) -> None:
        """Make one symbol the current target and show it to menus.

        Args:
            symbol: Symbol, the new target.
        """
        self.target = symbol
        self._chain.append(symbol)
        self.episode.open_observation(target_view(self.index, symbol))
        self.episode.log_session(
            f"> target: {symbol.qualname} in {symbol.file}")

    def _confirm_node(self) -> MenuNode:
        """Ask whether the problem is in the shown function.

        Returns:
            MenuNode: yes / a callee / somewhere else (no escape).
        """
        visited = self._struck | set(self._chain)
        callees = [c for c in self.index.callees_of(self.target)
                   if c not in visited][:MAX_CALLEE_OPTIONS]
        self._menus.callees = {CALLEE_LABEL.format(name=c.qualname): c
                               for c in callees}
        options = [CONFIRM_YES, *self._menus.callees, CONFIRM_ELSEWHERE]
        question = (f"Is the change the task asks for inside "
                    f"{self.target.qualname}()?")
        return MenuNode(question, options, escape=False)

    def _apply_confirm(self, decision) -> None:
        """Accept the target, hop to a callee, or strike the chain.

        Args:
            decision: Decision from the Policy.
        """
        if not decision.valid:
            self._fail(NOT_CONFIRMED)
            return
        if decision.value == CONFIRM_YES:
            self.episode.log_session("> confirmed")
            self.state = ACCEPTED
            return
        callee = self._menus.callees.get(decision.value)
        if callee is not None:
            self._hop(callee)
            return
        self.episode.log_session(f"> not in {self.target.qualname}")
        self._struck.update(self._chain)
        self._back_to_shortlist(NO_REMAINING)

    def _hop(self, callee) -> None:
        """Move the target to a callee; accept after MAX_CONFIRM_HOPS.

        Args:
            callee: Symbol, a function the current target calls.
        """
        self._show(callee)
        if len(self._chain) > MAX_CONFIRM_HOPS:
            self.state = ACCEPTED

    def _fail(self, reason: str) -> None:
        """End the search without a target.

        Args:
            reason: str, one line for the user.
        """
        self.failure = reason
        self.state = FAILED


def target_code(index, symbol) -> str:
    """Give a symbol's verbatim span, dedented to column 0.

    The confirm observation and the edit vertical's prompts use this.

    Args:
        index: SymbolIndex, gives the cached file text.
        symbol: Symbol, a function or method.
    Returns:
        str: the function source (decorators excluded).
    """
    source = index.source_of(symbol.file)
    def_line = source.split("\n")[symbol.lineno - 1]
    span = span_text(source, symbol.lineno, symbol.end_lineno)
    return dedent_block(span, leading_indent(def_line))


def target_view(index, symbol) -> str:
    """Give the menu observation that shows one target.

    Only TargetSearch._show calls this.

    Args:
        index: SymbolIndex, gives the cached file text.
        symbol: Symbol, the target to show.
    Returns:
        str: a header line plus at most CONFIRM_MAX_LINES source lines.
    """
    lines = target_code(index, symbol).split("\n")
    shown, hidden = nav.clip_lines(lines, CONFIRM_MAX_LINES)
    if hidden:
        shown.append(f"... ({hidden} more lines)")
    head = f"FUNCTION {symbol.qualname} in {symbol.file}:{symbol.lineno}:"
    return "\n".join([head, *shown])


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from threetoks.code.edit.index import SymbolIndex
    from threetoks.nodes import Decision
    from threetoks.render import Episode

    def _answer(node, label: str) -> Decision:
        return Decision("menu", label, "", valid=label in node.options)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "m.py").write_text(
            "def order_total(items):\n    return sum_prices(items)\n\n\n"
            "def sum_prices(items):\n    return sum(items) - 1\n")
        index = SymbolIndex(root)
        search = TargetSearch(index, "fix order_total", Episode("S", "t"))
        search.start()
        assert search.path_taken == "direct" and search.state == CONFIRM
        node = search.node()
        hop = CALLEE_LABEL.format(name="sum_prices")
        search.apply(_answer(node, hop))
        assert search.target.name == "sum_prices"
        search.apply(_answer(search.node(), CONFIRM_YES))
        assert search.state == ACCEPTED
        search.strike("gave up")
        assert search.state == PICK                  # order_total is left
        search.apply(_answer(search.node(), CONFIRM_ELSEWHERE))
        assert search.state == FAILED and search.failure == NO_REMAINING
        assert "def sum_prices" in target_code(index, index.symbols[1])
    print("smoke OK")
