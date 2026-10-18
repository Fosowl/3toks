"""Live progress tree: how the TUI shows one task while it runs.

The agent makes small decisions several times a second. The TUI shows
each decision when it arrives, as one row of a tree, so the user sees
what the model does and why. A row carries a token bar (▁▂▃…): its
height is the token cost of the decision and its color is the speed.

Span events from :func:`threetoks.trace.span` open and close named
branches, such as "pick an agent", "round 1" or "check the answer".
Inside a branch, a plain menu choice (a menu without a tag) is an
action: the decisions after it are its children until the next plain
menu choice. Each branch ends with a ``╰`` row that gives its result,
its model calls, its tokens and its time.

Rows stream: a row is never changed after it is shown. So the last
child of an action keeps its ``├`` joint; only branches end with ``╰``.
"""
import time
from dataclasses import dataclass

from threetoks.theme import (
    AMBER,
    CROSS,
    CYAN,
    GRAY_LIGHT,
    GREEN,
    PANEL_WIDTH,
    PINK,
    RAMP,
    VIOLET,
    ZAP,
    dim,
    fg,
)

TOKEN_BARS = "▁▂▃▄▅▆▇█"     # token-cost bar, 1..8+ tokens
STAT_COLUMN = PANEL_WIDTH   # stats start where the result panel ends
STATS_WIDTH = 12            # room for stats such as "24t·0.88s"
ROW_WIDTH = STAT_COLUMN + STATS_WIDTH  # picked sentences end here
MIN_STAT_GAP = 2            # spaces kept before the stats
FAST_S, STEADY_S = 0.4, 1.2 # decision-speed thresholds for bar color
VALUE_CLIP = 40             # chosen value clipped to this
MIN_PICKED_CLIP = 24        # picked sentence keeps at least this
LABEL_CLIP = 60             # branch label clipped to this
KIND_WIDTH = 7              # width of the node-kind column
RAW_CLIP = 120              # /debug: raw completion clipped to this
DEBUG_PROMPT_HEAD = 80      # /debug: prompt head shown before the seam
DEBUG_PROMPT_TAIL = 220     # /debug: prompt tail shown after the seam
DEBUG_RESULT_CLIP = 120     # /debug: result text clipped to this
MARGIN = "  "               # left margin of every tree row
RAIL_WIDTH = 2              # columns per depth level: glyph + space
FOLLOW_INDENT = "  "        # detail lines start under the token bar
PICK_PREFIX = "· "          # bullet of one picked sentence
RAIL, JOINT, END = "│", "├", "╰"
BRANCH_MARK = "◆"
_KIND_LABELS = {"menu": "choose", "pick_many": "pick", "short_text": "write",
                "sequence": "plan", "fetch": "fetch"}


def as_number(value, cast):
    """Read a number from a trace event or a result dict, or give 0.

    The tree and the TUI panels use this on values that can be missing
    or malformed. They call it with one raw value and a number type.

    Args:
        value: the raw value (any type; None and "" count as 0).
        cast (type): int or float.

    Returns:
        int | float: the number, or 0 when the value cannot be read.
    """
    try:
        return cast(value or 0)
    except (TypeError, ValueError):
        return cast(0)


def is_model_call(event: dict) -> bool:
    """Tell if a decision event came from the model.

    The tree uses this to count calls and to draw the token stats.
    It receives one decision event. A page fetch row is not a model call.

    Args:
        event (dict): one decision event from Tracer.record.

    Returns:
        bool: False for a "fetch" row, True for any other decision.
    """
    return event.get("node") != "fetch"


def _clip(text: str, limit: int) -> str:
    """Collapse to one line and clip to ``limit`` chars with an ellipsis."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def gutter(depth: int, joint: str, enabled: bool = True) -> str:
    """Draw the dim tree rails for one row inside open branches.

    The progress tree uses this before every row it shows.
    The tree calls it with the branch depth of the row and a joint glyph.
    It draws one rail per open branch; the deepest rail becomes the joint.

    Args:
        depth (int): number of open branches around the row (0 = none).
        joint (str): glyph of the deepest rail: JOINT for a child row,
            END for the last row of a branch, RAIL for a follow-up line.
        enabled (bool): True to add ANSI color, from the REPL state.

    Returns:
        str: the margin plus the rails, RAIL_WIDTH columns per level.
    """
    rails = [RAIL] * depth
    if rails:
        rails[-1] = joint
    return MARGIN + dim("".join(cell.ljust(RAIL_WIDTH) for cell in rails),
                        enabled)


def _follow_up(depth: int, text: str, enabled: bool) -> str:
    """Put one dim detail line under a row at ``depth``.

    The tree uses this for picked sentences and for /debug lines.
    The tree calls it with the depth of the row the detail belongs to.
    It keeps the rails open and indents the text under the token bar.

    Args:
        depth (int): branch depth of the row above the detail.
        text (str): the detail text; color is added here.
        enabled (bool): True to add ANSI color, from the REPL state.

    Returns:
        str: one tree row.
    """
    return gutter(depth, RAIL, enabled) + FOLLOW_INDENT + dim(text, enabled)


def _compress_value(event: dict) -> str:
    """Human-readable, compressed form of a decision's chosen value."""
    kind = event.get("node")
    value = event.get("value")
    if not event.get("valid", False) or value is None:
        return "—"  # em dash for no/invalid decision
    if kind in ("pick_many", "sequence"):
        return ",".join(str(index) for index in value) or "—"
    if kind == "plan":  # code plan: ParsedMethod rows, show their names
        return _clip(", ".join(getattr(item, "name", str(item))
                               for item in value), VALUE_CLIP)
    return _clip(value, VALUE_CLIP)


def _token_bar(tokens: int) -> str:
    """One glyph whose height encodes the decision's token cost."""
    if tokens <= 0:
        return "·"
    return TOKEN_BARS[min(tokens, len(TOKEN_BARS)) - 1]


def _pace_color(wall_s: float) -> int:
    """Bar color by decision speed: green fast, amber steady, pink slow."""
    if wall_s < FAST_S:
        return GREEN
    return AMBER if wall_s < STEADY_S else PINK


def _lead_glyph(event: dict, enabled: bool) -> str:
    """Draw the first glyph of a decision row.

    build_decision_row uses this for every row. It receives the event.
    A model call gets its token bar in its speed color; a page fetch
    row gets an amber cross.

    Args:
        event (dict): one decision event.
        enabled (bool): True to add ANSI color, from the REPL state.

    Returns:
        str: one colored glyph.
    """
    if not is_model_call(event):
        return fg(AMBER, CROSS, enabled)
    tokens = as_number(event.get("out_tokens"), int)
    wall = as_number(event.get("wall_s"), float)
    return fg(_pace_color(wall), _token_bar(tokens), enabled)


def _stats_text(event: dict) -> str:
    """Give the token and time stats of a model call, or "" for a fetch."""
    if not is_model_call(event):
        return ""
    tokens = as_number(event.get("out_tokens"), int)
    return f"{tokens:>2}t·{as_number(event.get('wall_s'), float):.2f}s"


def build_decision_row(event: dict, enabled: bool = True,
                       depth: int = 0) -> str:
    """Show one decision event as a tree row, with its picked sentences.

    The progress tree uses this for every decision the model makes.
    The tree calls it with a trace event from Policy and the row depth.
    It draws the rails, the lead glyph, the node kind, the chosen value
    and, for a model call, the token and time stats at STAT_COLUMN.

    Args:
        event (dict): the decision event; missing keys are tolerated.
        enabled (bool): True to add ANSI color, from the REPL state.
        depth (int): number of open branches around the row.

    Returns:
        str: one row, plus one follow-up row per picked sentence.
    """
    kind = event.get("tag") or _KIND_LABELS.get(event.get("node"),
                                                str(event.get("node", "?")))
    value = _compress_value(event)
    stats = _stats_text(event)
    used = (len(MARGIN) + RAIL_WIDTH * depth
            + len(f"x {kind:<{KIND_WIDTH}} {value}"))
    pad = " " * max(MIN_STAT_GAP, STAT_COLUMN - used) if stats else ""
    line = (gutter(depth, JOINT, enabled) + _lead_glyph(event, enabled)
            + " " + fg(GRAY_LIGHT, f"{kind:<{KIND_WIDTH}}", enabled) + " "
            + fg(CYAN, value, enabled) + pad + dim(stats, enabled))
    room = max(MIN_PICKED_CLIP, ROW_WIDTH - len(MARGIN) - RAIL_WIDTH * depth
               - len(FOLLOW_INDENT) - len(PICK_PREFIX))
    picked = [_follow_up(depth, PICK_PREFIX + _clip(text, room), enabled)
              for text in event.get("resolved") or []]
    return "\n".join([line] + picked)


def build_branch_line(label: str, depth: int, enabled: bool = True) -> str:
    """Show the first row of a branch: a colored mark and its label.

    The progress tree uses this when a span opens.
    The tree calls it with the span label and the depth of the new row.
    It colors the mark by depth along the theme ramp.

    Args:
        label (str): what the branch does, e.g. "round 1 · search and read".
        depth (int): number of open branches around the row.
        enabled (bool): True to add ANSI color, from the REPL state.

    Returns:
        str: one tree row.
    """
    mark = fg(RAMP[min(depth, len(RAMP) - 1)], BRANCH_MARK, enabled)
    return (gutter(depth, JOINT, enabled) + mark + " "
            + fg(VIOLET, _clip(label, LABEL_CLIP), enabled))


def _plural(count: int, word: str) -> str:
    """Return "1 call" or "2 calls"."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


@dataclass
class Branch:
    """One branch of the tree: what it cost and, once closed, its end."""
    started: float           # time.time() when the branch opened
    calls: int = 0           # model calls inside, children included
    tokens: int = 0          # output tokens inside, children included
    in_action: bool = False  # True: rows nest under the last action
    result: str | None = None  # span result, set when the branch closes
    seconds: float = 0.0     # time open, set when the branch closes


def build_end_line(branch: Branch, depth: int, enabled: bool = True) -> str:
    """Show the last row of a branch: its result, calls, tokens and time.

    The progress tree uses this when a span closes.
    The tree calls it with the closed branch, which holds the result.
    It writes "done" when the span gave no result.

    Args:
        branch (Branch): the closed branch, with tallies, result, seconds.
        depth (int): depth of the rows inside the branch.
        enabled (bool): True to add ANSI color, from the REPL state.

    Returns:
        str: one tree row.
    """
    stats = (f" · {_plural(branch.calls, 'call')} · {branch.tokens}t"
             f" · {branch.seconds:.1f}s")
    return (gutter(depth, END, enabled)
            + fg(CYAN, _clip(branch.result or "done", VALUE_CLIP), enabled)
            + dim(stats, enabled))


def _clip_repr(text, limit: int) -> str:
    """``repr`` of ``text``, clipped to ``limit`` chars with an ellipsis.

    The quotes make an empty or whitespace-only completion visible (``''``,
    ``'\\n'``) where a bare string would look like a rendering bug; the
    ellipsis sits outside them so it never reads as characters the model
    actually emitted.
    """
    raw = str(text or "")
    if len(raw) <= limit:
        return repr(raw)
    return repr(raw[:limit]) + "…"


def _clip_repr_ends(text, head: int, tail: int) -> str:
    """``repr`` of ``text`` with the middle elided: head, ``…``, tail, ``…``.

    Menu prompts carry their identity at the head (family template, system
    block, few-shots) and their decision inputs at the tail (user request,
    ACTIONS menu, prefill); a head-only clip showed nothing but boilerplate,
    which is exactly the part you do not need when debugging a misroute.
    Short prompts pass through as a plain ``repr``.
    """
    raw = str(text or "")
    if len(raw) <= head + tail:
        return repr(raw)
    return repr(raw[:head]) + "…" + repr(raw[-tail:]) + "…"


def build_debug_line(event: dict) -> str:
    """One plain line showing the model's RAW answer for a decision.

    Printed under every decision while ``/debug`` is on. It exists to
    make a ``—`` (unparseable) decision explain itself: ``done: length``
    means the node's token cap ended the completion before the model got
    to its digit — the failure mode of a chat transport, where the
    ``ANSWER:`` prefill is only a hint and a chatty model spends the cap
    on preamble. ``attempt`` is 1-based, so it counts up to the retry
    ladder's length. An empty completion at ``done: stop`` gets its own
    hint: the model's first token already hit a stop (its end-of-turn
    token or the node's newline stop) — usually a judgment that nothing
    fits, not a crash.
    """
    raw = _clip_repr(event.get("raw_text"), RAW_CLIP)
    reason = event.get("done_reason") or "?"
    attempt = as_number(event.get("attempt"), int) + 1
    line = f"· raw {raw} · done: {reason} · attempt {attempt}"
    if reason == "stop" and not str(event.get("raw_text") or "").strip():
        line += " — empty answer: the model stopped before answering"
    return line


def build_debug_tokens_line(event: dict) -> str:
    """One plain line showing the prompt and result tokens of a model call.

    Printed for every LLM request while ``/debug`` is on. Menu prompts
    carry boilerplate (family template, system block, few-shots) at the
    head and the decision inputs (user request, ACTIONS menu, prefill)
    at the tail, so the prompt is shown as head + tail with the middle
    elided; ``/debug full`` prints everything.
    """
    prompt = _clip_repr_ends(event.get("prompt_text"),
                             DEBUG_PROMPT_HEAD, DEBUG_PROMPT_TAIL)
    result = _clip_repr(event.get("result_text"), DEBUG_RESULT_CLIP)
    pt = as_number(event.get("prompt_tokens"), int)
    ot = as_number(event.get("out_tokens"), int)
    reason = event.get("done_reason") or "?"
    return f"· in {prompt} → out {result} · {pt}pt/{ot}ot · {reason}"


def build_debug_full_lines(event: dict) -> list[str]:
    """The complete prompt and result of one model call (``/debug full``).

    For reading exactly what a misroute saw: the system block (chat
    transports only — a raw template embeds it in ``prompt_text``), every
    prompt line, then the unclipped completion with its token counts.
    Nothing is clipped or wrapped, so a dump can be pasted into a bug
    report exactly as the model saw it.
    """
    lines = []
    system = str(event.get("system_text") or "")
    if system:
        lines.append("· system:")
        lines.extend(f"· {row}" for row in system.splitlines() or [""])
    lines.append("· prompt:")
    prompt = str(event.get("prompt_text") or "")
    lines.extend(f"· {row}" for row in prompt.splitlines() or [""])
    stats = (f"{as_number(event.get('prompt_tokens'), int)}pt/"
             f"{as_number(event.get('out_tokens'), int)}ot · "
             f"{event.get('done_reason') or '?'}")
    lines.append(f"· out {str(event.get('result_text') or '')!r} · {stats}")
    return lines


class ProgressTree:
    """Streams one task as a tree and counts its model calls and tokens.

    The TUI sets an instance as ``tracer.on_event`` for one task, and
    its :meth:`note` as ``services.on_progress``.
    """

    def __init__(self, view, sink):
        """Keep the display flags and the output callable.

        tui._run_task makes one tree per task; demos and tests make one
        with a small stand-in view.

        Args:
            view: the REPL state; reads ``enabled``, ``debug`` and
                ``debug_full`` (bool) at each event.
            sink (callable): takes one str and shows it, e.g. print.
        """
        self.view = view
        self.sink = sink
        self.branches = [Branch(time.time())]  # root: never shown
        self.model_seconds = 0.0
        self.retries = 0
        self.pending_debug: list[str] = []  # dumps wait for their row

    def __call__(self, event: dict) -> None:
        """Show one trace event: a debug dump, a span edge or a decision.

        Args:
            event (dict): one event from Tracer.record.
        """
        if event.get("debug"):
            self._hold_debug(event)
        elif event.get("span") == "open":
            self._open(str(event.get("label") or ""))
        elif event.get("span") == "close":
            self._close(event.get("result"))
        else:
            self._show_decision(event)

    def note(self, line: str) -> None:
        """Show one progress line from an agent as a leaf row.

        The planner sends its plan lines here through on_progress.

        Args:
            line (str): the progress text, without color.
        """
        self._leave_action()
        self.sink(gutter(self._depth(), JOINT, self.view.enabled)
                  + dim(f"{ZAP} {line}", self.view.enabled))

    def stats(self, seconds: float) -> dict:
        """Give the task tallies for the panel and the task report.

        tui._run_task calls this once, after the agent returns.

        Args:
            seconds (float): wall time of the whole task.

        Returns:
            dict: "seconds", "calls", "tokens", "model_seconds" and
            "retries" (numbers).
        """
        root = self.branches[0]
        return {"seconds": seconds, "calls": root.calls,
                "tokens": root.tokens, "model_seconds": self.model_seconds,
                "retries": self.retries}

    def _depth(self) -> int:
        """Depth of a new row: open branches, plus one inside an action."""
        top = self.branches[-1]
        return len(self.branches) - 1 + (1 if top.in_action else 0)

    def _leave_action(self) -> None:
        """Make the next row a sibling of the open action, not its child."""
        self.branches[-1].in_action = False

    def _open(self, label: str) -> None:
        """Show the branch row and push the new branch."""
        self._leave_action()
        self.sink(build_branch_line(label, self._depth(), self.view.enabled))
        self.branches.append(Branch(time.time()))

    def _close(self, result) -> None:
        """Show the end row of the deepest branch and pop it."""
        if len(self.branches) == 1:
            return  # a close without an open: nothing to end
        depth = len(self.branches) - 1
        branch = self.branches.pop()
        branch.result = result
        branch.seconds = time.time() - branch.started
        self.sink(build_end_line(branch, depth, self.view.enabled))

    def _show_decision(self, event: dict) -> None:
        """Count one decision and show it; a plain menu opens an action."""
        if is_model_call(event):
            self._count(event)
        is_action = event.get("node") == "menu" and not event.get("tag")
        if is_action:
            self._leave_action()
        depth = self._depth()
        self.sink(build_decision_row(event, self.view.enabled, depth))
        self._flush_debug(event, depth)
        if is_action:
            self.branches[-1].in_action = True

    def _count(self, event: dict) -> None:
        """Add one model call to every open branch and to the task."""
        tokens = as_number(event.get("out_tokens"), int)
        self.model_seconds += as_number(event.get("wall_s"), float)
        self.retries += as_number(event.get("attempt"), int) > 0
        for branch in self.branches:
            branch.calls += 1
            branch.tokens += tokens

    def _hold_debug(self, event: dict) -> None:
        """Keep the prompt dump of one model call while /debug is on.

        Policy records the dump before the decision it belongs to, so the
        rows wait and show under that decision row.
        """
        if not self.view.debug:
            return
        self.pending_debug.extend(build_debug_full_lines(event)
                                  if self.view.debug_full
                                  else [build_debug_tokens_line(event)])

    def _flush_debug(self, event: dict, depth: int) -> None:
        """Show the raw answer and the held dumps under a model-call row."""
        if not (self.view.debug and is_model_call(event)):
            return
        rows = [build_debug_line(event)] + self.pending_debug
        self.pending_debug = []
        for row in rows:
            self.sink(_follow_up(depth, row, self.view.enabled))


# A scripted web task: route, one research round, the judge.
DEMO_EVENTS = (
    {"span": "open", "label": "pick an agent"},
    {"node": "menu", "value": "web — look up facts or current information",
     "valid": True, "out_tokens": 2, "wall_s": 0.31},
    {"span": "close", "result": "web"},
    {"span": "open", "label": "web agent"},
    {"span": "open", "label": "round 1 · search and read"},
    {"node": "menu", "value": "open result 2: Anthropic pricing",
     "valid": True, "out_tokens": 2, "wall_s": 0.31},
    {"node": "fetch", "value": "failed to open 'Anthropic pricing' (403)",
     "valid": True},
    {"node": "menu", "value": "open result 4: Claude docs",
     "valid": True, "out_tokens": 2, "wall_s": 0.52},
    {"node": "pick_many", "value": [3, 7], "valid": True, "resolved":
     ["Opus pricing was updated in June.", "Input tokens cost $15/M."],
     "out_tokens": 5, "wall_s": 0.62},
    {"node": "menu", "value": "answer the task now",
     "valid": True, "out_tokens": 2, "wall_s": 0.4},
    {"node": "short_text", "value": "Claude Opus costs $15 per million "
     "input tokens.", "valid": True, "out_tokens": 14, "wall_s": 1.55},
    {"span": "close"},
    {"span": "open", "label": "check the answer"},
    {"tag": "judge", "node": "menu", "value": "good — it directly answers",
     "valid": True, "out_tokens": 2, "wall_s": 0.3},
    {"span": "close", "result": "accepted"},
    {"span": "close", "result": "answered"},
)


if __name__ == "__main__":
    from types import SimpleNamespace
    shown: list[str] = []
    flags = SimpleNamespace(enabled=False, debug=False, debug_full=False)
    tree = ProgressTree(flags, shown.append)
    for demo_event in DEMO_EVENTS:
        tree(demo_event)
    print("\n".join(shown))
    assert shown[0] == "  ◆ pick an agent" and shown[2].startswith("  ╰ web")
    assert tree.stats(1.0)["calls"] == 7 and len(tree.branches) == 1
    print("smoke OK")
