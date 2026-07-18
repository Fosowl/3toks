"""Neon-dawn cyberpunk terminal UI for the ThreeToks agent mesh.

The killer view: the agent makes ~1-token menu decisions several times a
second, and this TUI streams each one live as a ticker line, so the user
literally watches the model think. Each ticker line carries a tiny token
bar (▁▂▃…) whose height is the decision's token cost and whose color is
its speed — the frugality of the mesh made visible.

One visual identity holds it together: a cyan → lavender → hot-pink
gradient ramp on hairline rules and the vertical spine of result panels,
while the startup wordmark draws a random dark-neon ramp per launch. All
structural chrome sits in dim gray so only the model's decisions and
answers carry color.

Rendering is stdlib-only raw ANSI; every escape sequence hides behind a
small helper (:func:`fg`, :func:`dim`, :func:`gradient`) so there is no
inline escape soup. ``NO_COLOR`` and non-tty output fall back to plain
text. Pure formatting/dispatch functions (:func:`build_ticker_line`,
:func:`render_result`, :func:`handle_command`) are kept free of the input
loop so they can be unit-tested offline without an LLM or a network.
"""
import os
import random
import textwrap
import sys
import time
from dataclasses import dataclass, field

# --- palette (xterm-256) -------------------------------------------------
CYAN = 51        # decision values, prompt chevron
VIOLET = 141     # agent identity, headings
PINK = 205       # answer labels, slow decisions
GREEN = 84       # fast decisions
AMBER = 214      # warnings / errors
GRAY = 240       # structural chrome
GRAY_LIGHT = 246 # secondary text
WHITE = 253      # answer text

# Signature ramp: neon dawn, cyan → lavender → hot pink.
RAMP = (51, 81, 111, 141, 171, 205)

# --- glyphs --------------------------------------------------------------
SPINE = "▍"                 # ▍ panel gutter
NODE = "◈"                  # ◈ hairline node
ZAP = "⌁"                   # ⌁ mesh events
CHEVRON = "❯"               # ❯ prompt
NOTE_BULLET = "▪"           # ▪
CROSS = "✖"                 # ✖
TOKEN_BARS = "▁▂▃▄▅▆▇█"     # ticker token-cost bar, 1..8+ tokens

# --- misc constants ------------------------------------------------------
_ESC = "\x1b"
PANEL_WIDTH = 64
STAT_COLUMN = 56            # ticker column where the trailing stats start
FAST_S, STEADY_S = 0.4, 1.2 # decision-speed thresholds for bar color
SHORT_TEXT_CLIP = 40
PICKED_LINE_CLIP = 70
QUESTION_CLIP = 48
RAW_CLIP = 120              # /debug: raw completion clipped to this
DEBUG_PROMPT_CLIP = 200     # /debug: prompt text clipped to this
DEBUG_RESULT_CLIP = 120     # /debug: result text clipped to this
UNDER_INDENT = " " * 12     # column where a ticker line's value starts
SIGN_OFF = "⌁ mesh offline — tokens saved, see you"
_KIND_LABELS = {"menu": "menu", "pick_many": "pick", "short_text": "text"}


def build_debug_tokens_line(event: dict, enabled: bool = True) -> str:
    """One dim line showing the full prompt and result tokens for a model call.

    Printed for every LLM request while ``/debug`` is on, aligned under the
    ticker's value column. Shows the clipped prompt text, result text, and
    token counts.
    """
    prompt = _clip_repr(event.get("prompt_text"), DEBUG_PROMPT_CLIP)
    result = _clip_repr(event.get("result_text"), DEBUG_RESULT_CLIP)
    pt = _num(event.get("prompt_tokens"), int)
    ot = _num(event.get("out_tokens"), int)
    reason = event.get("done_reason") or "?"
    return dim(f"{UNDER_INDENT}· in {prompt} → out {result}"
               f" · {pt}pt/{ot}ot · {reason}", enabled)


def _color_enabled(stream=None) -> bool:
    """True when ANSI color is appropriate for the given stream."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def fg(code: int, text: str, enabled: bool = True) -> str:
    """Wrap ``text`` in a 256-color foreground escape (or return it plain)."""
    if not enabled:
        return text
    return f"{_ESC}[38;5;{code}m{text}{_ESC}[0m"


def dim(text: str, enabled: bool = True) -> str:
    """Render ``text`` in dim gray, or plain when color is disabled."""
    return fg(GRAY, text, enabled)


def _ramp_at(position: int, total: int, ramp=RAMP) -> int:
    """Color code for step ``position`` of ``total`` along ``ramp``."""
    if total <= 1:
        return ramp[0]
    return ramp[round(position * (len(ramp) - 1) / (total - 1))]


def gradient(text: str, enabled: bool = True, ramp=RAMP) -> str:
    """Color each character through ``ramp`` (plain when disabled)."""
    if not enabled:
        return text
    return "".join(fg(_ramp_at(index, len(text), ramp), char)
                   for index, char in enumerate(text))


def hairline(node_at: int, enabled: bool = True, ramp=RAMP) -> str:
    """A gradient hairline rule with a ◈ node at column ``node_at``."""
    line = "─" * PANEL_WIDTH
    return gradient(line[:node_at] + NODE + line[node_at + 1:], enabled, ramp)


def _clip(text: str, limit: int) -> str:
    """Collapse to one line and clip to ``limit`` chars with an ellipsis."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _compress_value(event: dict) -> str:
    """Human-readable, compressed form of a decision's chosen value."""
    kind = event.get("node")
    value = event.get("value")
    if not event.get("valid", False) or value is None:
        return "—"  # em dash for no/invalid decision
    if kind == "pick_many":
        return ",".join(str(index) for index in value) or "—"
    if kind == "short_text":
        return _clip(value, SHORT_TEXT_CLIP)
    return _clip(value, SHORT_TEXT_CLIP)


def _resolved_lines(event: dict, enabled: bool) -> list[str]:
    """One dim line per picked sentence, printed under a pick event."""
    return [dim(f"{UNDER_INDENT}· {_clip(text, PICKED_LINE_CLIP)}", enabled)
            for text in event.get("resolved") or []]


def _num(value, cast):
    """Coerce a trace/result value to a number; 0 on anything malformed."""
    try:
        return cast(value or 0)
    except (TypeError, ValueError):
        return cast(0)


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


def build_ticker_line(event: dict, enabled: bool = True) -> str:
    """One ticker line for a streamed decision event.

    A speed-colored token bar leads, then the dim node kind, the chosen
    value in cyan, and a column-aligned dim stat, e.g.
    ``  ▂ menu    open result 2                    2t·0.31s``.
    Tolerates missing optional keys from the trace event.
    """
    kind = event.get("tag") or _KIND_LABELS.get(event.get("node"),
                                                str(event.get("node", "?")))
    value = _compress_value(event)
    tokens = _num(event.get("out_tokens"), int)
    wall = _num(event.get("wall_s"), float)
    plain_width = len(f"  {_token_bar(tokens)} {kind:<7} {value}")
    pad = " " * max(2, STAT_COLUMN - plain_width)
    line = ("  " + fg(_pace_color(wall), _token_bar(tokens), enabled) + " "
            + dim(f"{kind:<7}", enabled) + " "
            + fg(CYAN, value, enabled) + pad
            + dim(f"{tokens}t·{wall:.2f}s", enabled))
    return "\n".join([line] + _resolved_lines(event, enabled))


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


def build_debug_line(event: dict, enabled: bool = True) -> str:
    """One dim line under the ticker showing the model's RAW answer.

    Printed for every decision while ``/debug`` is on, aligned under the
    ticker's value column like a resolved pick. It exists to make a ``—``
    (unparseable) decision explain itself: ``done: length`` means the
    node's token cap ended the completion before the model got to its
    digit — the failure mode of a chat transport, where the ``ANSWER:``
    prefill is only a hint and a chatty model spends the cap on preamble.
    ``attempt`` is 1-based, so it counts up to the retry ladder's length.
    """
    raw = _clip_repr(event.get("raw_text"), RAW_CLIP)
    reason = event.get("done_reason") or "?"
    attempt = _num(event.get("attempt"), int) + 1
    return dim(f"{UNDER_INDENT}· raw {raw} · done: {reason}"
               f" · attempt {attempt}", enabled)


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Wrap plain text to the panel width; never truncates content."""
    return textwrap.wrap(text, width=width,
                         subsequent_indent=indent) or [""]


def _spined(rows: list[str], enabled: bool, color: int = None) -> list[str]:
    """Prefix each row with a ▍ spine, gradient top→bottom unless fixed.

    The spine is the panel's only frame: a thin neon edge stepping through
    the ramp, or a single ``color`` for uniform panels (e.g. errors).
    """
    out = []
    for index, row in enumerate(rows):
        code = color if color is not None else _ramp_at(index, len(rows))
        out.append((fg(code, " " + SPINE, enabled) + " " + row).rstrip())
    return out


def _stats_line(result: dict, enabled: bool) -> str:
    """Dim ``N decisions · T tok · S.s`` line, rounds appended if present."""
    parts = [f"{result.get('decisions', 0)} decisions",
             f"{result.get('tokens', 0)} tok",
             f"{_num(result.get('seconds'), float):.1f}s"]
    if result.get("rounds") is not None:
        parts.append(f"{result['rounds']} rounds")
    return dim(" · ".join(parts), enabled)


def _source_lines(result: dict, enabled: bool) -> list[str]:
    """Dim, bullet-prefixed lines for notes/sources (may be empty)."""
    raw = result.get("sources") or result.get("notes") or ""
    entries = raw if isinstance(raw, list) else str(raw).splitlines()
    lines = []
    for entry in entries:
        text = str(entry).strip()
        for seg in (_wrap(f"{NOTE_BULLET} {text}", PANEL_WIDTH - 4, "  ")
                    if text else []):
            lines.append(dim(seg, enabled))
    return lines


def _panel_rows(result: dict, enabled: bool) -> list[str]:
    """Content rows of the panel: label, target, answer, output, sources,
    stats."""
    label = fg(PINK, str(result.get("agent", "agent")).upper(), enabled)
    if result.get("judged_good") is False:
        label += fg(AMBER, " · UNVERIFIED", enabled)
    rows = [label]
    target = result.get("target")
    if isinstance(target, dict) and target.get("label"):
        note = f" — {target['anchor']}" if target.get("anchor") else ""
        rows.extend(dim(seg, enabled) for seg in _wrap(
            f"target: {target['label']}{note}", PANEL_WIDTH - 3))
    answer = str(result.get("answer") or "—")
    for answer_line in answer.splitlines() or [""]:
        rows.extend(fg(WHITE, seg, enabled)
                    for seg in _wrap(answer_line, PANEL_WIDTH - 3))
    rows.extend(_output_rows(result, enabled))
    sources = _source_lines(result, enabled)
    if sources:
        rows.append("")
        rows.extend(sources)
    rows.extend(["", _stats_line(result, enabled)])
    return rows


OUTPUT_MAX_LINES = 12


def _output_rows(result: dict, enabled: bool) -> list[str]:
    """The script's captured stdout, shown under the code it came from.

    The runner already executed the delivered entry (§7b smoke call);
    hiding what it printed made a working script look inert. Long output
    is clipped with an honest truncation count.
    """
    output = str(result.get("output") or "").rstrip()
    if not output:
        return []
    lines = output.splitlines()
    shown, hidden = lines[:OUTPUT_MAX_LINES], len(lines) - OUTPUT_MAX_LINES
    rows = ["", dim("output when run:", enabled)]
    for line in shown:
        rows.extend(fg(GREEN, seg, enabled)
                    for seg in _wrap(line, PANEL_WIDTH - 3))
    if hidden > 0:
        rows.append(dim(f"… {hidden} more line(s)", enabled))
    return rows


def render_result(result: dict, enabled: bool = True) -> str:
    """Render the result panel down a neon gradient spine.

    Label pink, answer bright, sources and stats dim. Never crashes on
    missing optional keys — ``answer`` defaults to a dash and every stat
    falls back to zero.
    """
    return "\n".join(_spined(_panel_rows(result, enabled), enabled))


def render_task_report(stats: dict, enabled: bool = True) -> str:
    """Dim task report shown under the result panel.

    First line: task wall time and the model's share of it. Second
    line: per-call averages — the frugality headline (tokens per model
    call) made explicit — plus a retry count when any attempt needed a
    reshuffle. A task answered without any model call reports one line.
    """
    seconds = _num(stats.get("seconds"), float)
    calls = _num(stats.get("calls"), int)
    model_s = _num(stats.get("model_seconds"), float)
    head = f"  {ZAP} task report {CHEVRON} {seconds:.1f}s"
    if model_s and seconds:
        share = min(100, round(100 * model_s / seconds))
        head += f" · model {model_s:.1f}s ({share}%)"
    if not calls:
        return dim(head + " · no model calls", enabled)
    tokens = _num(stats.get("tokens"), int)
    parts = [f"{calls} calls", f"{tokens} tok",
             f"avg {tokens / calls:.1f} tok/call",
             f"{(model_s or seconds) / calls:.2f}s/call"]
    retries = _num(stats.get("retries"), int)
    if retries:
        parts.append(f"{retries} retries")
    return "\n".join((dim(head, enabled),
                      dim("    " + " · ".join(parts), enabled)))


# "3 TOKS" wordmark: a glitch-dissolve "3" beside DOS-shadow "TOKS".
# Rendered rows are led by random motion trails (:func:`_motion_trail`)
# hugging each line's first glyph, plus a streak under the 3, as if the
# wordmark were speeding rightward; the TOKS shadow row is static.
BANNER_ART = (
    "▓█████▙   ████████╗ ██████╗ ██╗  ██╗ ███████╗",
    " ░░▒▓██▌  ╚══██╔══╝██╔═══██╗██║ ██╔╝ ██╔════╝",
    "  ▟███▛░     ██║   ██║   ██║█████╔╝  ███████╗",
    " ░▒▀▜██▌     ██║   ██║   ██║██╔═██╗  ╚════██║",
    "▓█████▛      ██║   ╚██████╔╝██║  ██╗ ███████║",
)
BANNER_SHADOW = "╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚══════╝"
STREAK_COLUMN = 13   # art column where the TOKS shadow row starts
BAR_INDENT = 7       # under-3 trail may reach the bottom bar's right edge
TRAIL_WIDTH = 6      # columns left of the art reserved for motion trails
TRAIL_SHADES = " ░▒"

# Launch ramps: build_banner draws one per call; all run dark to neon so
# the left trail zone stays in shadow and the wordmark brightens rightward.
BANNER_RAMPS = (
    (54, 55, 56, 92, 93, 129),     # midnight: deep purple → electric violet
    (19, 25, 26, 32, 38, 44),      # blade rain: navy → azure → turquoise
    (23, 29, 30, 36, 37, 43),      # deep sea: dark teal → aqua
    (53, 89, 125, 161, 162, 197),  # noir crimson: wine → neon crimson
    (22, 28, 29, 35, 41, 42),      # ghost circuit: dark green → sea green
)


def _motion_trail(rng, indent: int) -> str:
    """A motion trail ``TRAIL_WIDTH + indent`` columns wide: light shades
    densening rightward against the glyphs, flickering per column."""
    span = TRAIL_WIDTH + indent
    top = len(TRAIL_SHADES) - 1
    length = rng.randint(2, span)
    shades = []
    for column in range(length):
        rise = top * (column + 1) / length
        flicker = rng.choice((-1, -1, 0, 0, 1))
        shades.append(TRAIL_SHADES[max(0, min(top, round(rise) + flicker))])
    return "".join(shades).rjust(span)


def _banner_rows(rng) -> list:
    """The art rows, each led by a fresh trail hugging its first glyph."""
    rows = []
    for row in BANNER_ART:
        indent = len(row) - len(row.lstrip())
        rows.append(_motion_trail(rng, indent) + row.lstrip())
    streak = _motion_trail(rng, BAR_INDENT)
    rows.append(streak.ljust(TRAIL_WIDTH + STREAK_COLUMN) + BANNER_SHADOW)
    return rows


def build_banner(model: str, agent_count: int, enabled: bool = True,
                 rng=None) -> str:
    """The startup banner: the gradient "3 TOKS" art between hairline rules.

    Each call draws a random dark-neon ramp from ``BANNER_RAMPS`` plus
    fresh motion trails; pass a seeded ``rng`` for a reproducible banner.
    Rows are padded to a common width before :func:`gradient` so the ramp
    colors align into vertical bands across all rows.
    """
    rng = rng or random.Random()
    ramp = rng.choice(BANNER_RAMPS)
    rows = _banner_rows(rng)
    width = max(len(row) for row in rows)
    art = ["  " + gradient(row.ljust(width), enabled, ramp) for row in rows]
    tagline = "3 tokens per action or bust — local · frugal · fast"
    meta = f"{ZAP} {model} · {agent_count} agents online"
    return "\n".join([
        hairline(6, enabled, ramp),
        *art,
        "  " + dim(tagline, enabled),
        "  " + fg(GRAY_LIGHT, meta, enabled),
        hairline(PANEL_WIDTH - 7, enabled, ramp),
    ])


def prompt_text(enabled: bool = True) -> str:
    """The two-line input prompt ``╭╴you`` / ``╰╴❯ ``."""
    return (dim("╭╴", enabled) + fg(GRAY_LIGHT, "you", enabled) + "\n"
            + dim("╰╴", enabled) + fg(CYAN, CHEVRON + " ", enabled))


# --- REPL state + slash-command dispatch --------------------------------
@dataclass
class ReplState:
    """Mutable REPL state passed to command handlers.

    ``policy_factory`` rebuilds the decision policy from a model name so
    ``/model`` can swap models without importing a backend at test time.
    """
    services: object
    specs: list
    policy: object
    model: str
    policy_factory: object
    enabled: bool = True
    running: bool = True
    lines: list = field(default_factory=list)
    fetcher: object = None  # live browser fetcher to close on mode switch
    memory: object = None   # MemoryStore for this session, or None if disabled
    memory_path: str = ""   # where memory is dumped when the session ends
    browser_visible: bool = False  # config default for /browser visibility
    voice: object = None    # live VoiceSession, or None for keyboard input
    voice_config: object = None  # [voice] config section for /voice on
    llm_provider: str = "ollama"  # LLM transport name (NOT services.provider,
    # which is the web SearchProvider); families are raw-mode only
    debug: bool = False     # /debug: show each decision's raw model answer
    # (NOT voice.debug, which traces microphone transcripts)


def _cmd_help(state: ReplState, arg: str) -> str:
    """Styled list of the available slash commands."""
    rows = [
        ("/help", "show this command list"),
        ("/agents", "list registered agents"),
        ("/deep N", "set research rounds to N"),
        ("/browser MODE [on-screen]", "fetch via http|plain|stealth"),
        ("/model NAME", "swap the deciding model"),
        ("/voice [on|off|debug]", "talk instead of typing (voice extras)"),
        ("/debug [on|off]", "show the raw model answer per decision"),
        ("/setup", "re-run the config wizard"),
        ("/quit", "leave the mesh"),
    ]
    out = [fg(VIOLET, f"{ZAP} commands", state.enabled)]
    out.extend("  " + fg(CYAN, name.ljust(28), state.enabled)
               + dim(desc, state.enabled) for name, desc in rows)
    return "\n".join(out)


def _cmd_agents(state: ReplState, arg: str) -> str:
    """List registered agents with their one-line descriptions."""
    out = [fg(VIOLET, f"{ZAP} agents", state.enabled)]
    for spec in state.specs:
        out.append("  " + fg(VIOLET, spec.name.ljust(12), state.enabled)
                   + dim(spec.description, state.enabled))
    return "\n".join(out)


def _cmd_deep(state: ReplState, arg: str) -> str:
    """Set ``services.max_research_rounds`` from the numeric argument."""
    if not arg.strip().isdigit():
        return fg(AMBER, "usage: /deep N (a positive integer)", state.enabled)
    state.services.max_research_rounds = int(arg.strip())
    return dim(f"deep research rounds = {arg.strip()}", state.enabled)


def _refresh_optional_agents(state: ReplState, model: str) -> None:
    """Re-derive the capability agents for ``model`` (look needs vision)."""
    from threetoks.agents import OPTIONAL_AGENT_NAMES, optional_agents
    kept = [spec for spec in state.specs
            if spec.name not in OPTIONAL_AGENT_NAMES]
    state.specs = kept + optional_agents(state.services, model)


def _cmd_model(state: ReplState, arg: str) -> str:
    """Rebuild the policy for a new model name via the policy factory.

    The optional agents are re-derived too: swapping to/from a
    vision-capable model registers/unregisters the ``look`` agent.
    """
    name = arg.strip()
    if not name:
        return fg(AMBER, "usage: /model NAME", state.enabled)
    state.model = name
    state.policy = state.policy_factory(name)
    _refresh_optional_agents(state, name)
    notice = unknown_family_notice(name) if state.llm_provider == "ollama" \
        else None
    if notice:
        return fg(AMBER, notice, state.enabled)
    return dim(f"model = {name}", state.enabled)


_BROWSER_MODES = ("http", "plain", "stealth")


def _close_active_fetcher(state: ReplState) -> None:
    """Quit any browser fetcher from a previous ``/browser`` selection."""
    if state.fetcher is not None:
        try:
            state.fetcher.close()
        except Exception:  # noqa: BLE001 - teardown must never crash the REPL
            pass
        state.fetcher = None


def _make_browser_fetcher(mode: str, visible: bool):
    """Lazily build a fetcher for ``plain`` / ``stealth`` (imports deferred)."""
    if mode == "plain":
        from threetoks.web.browser import make_fetcher
        return make_fetcher(visible=visible)
    from threetoks.web.stealth_browser import make_stealth_fetcher
    return make_stealth_fetcher(visible=visible)


def _cmd_browser(state: ReplState, arg: str) -> str:
    """Swap ``services.fetch_page`` between http / plain / stealth fetchers.

    ``/browser http`` restores the plain-HTTP fetch. ``plain`` and
    ``stealth`` drive a real Chrome; visibility defaults to the config's
    ``[browser] visible`` and an explicit ``on-screen`` / ``headless``
    argument overrides it. The browser modules are optional, so imports
    are lazy and every failure surfaces as a styled message, not a
    traceback.
    """
    parts = arg.split()
    mode = parts[0].lower() if parts else ""
    if mode not in _BROWSER_MODES:
        return fg(AMBER,
                  "usage: /browser http|plain|stealth [on-screen|headless]",
                  state.enabled)
    visible = state.browser_visible
    if len(parts) > 1 and parts[1].lower() == "on-screen":
        visible = True
    elif len(parts) > 1 and parts[1].lower() == "headless":
        visible = False
    _close_active_fetcher(state)
    if mode == "http":
        from threetoks.cli import fetch_page
        state.services.fetch_page = fetch_page
        return dim("browser http — plain HTTP fetch", state.enabled)
    try:
        fetcher = _make_browser_fetcher(mode, visible)
    except Exception as error:  # noqa: BLE001 - optional module, any failure
        from threetoks.cli import fetch_page
        state.services.fetch_page = fetch_page  # never leave a dead driver bound
        return fg(AMBER, f"browser unavailable: {error} — reverted to http",
                  state.enabled)
    state.fetcher = fetcher
    state.services.fetch_page = fetcher.fetch_page
    window = "on-screen" if visible else "headless"
    return dim(f"browser {mode} — Chrome ({window})", state.enabled)


def _cmd_setup(state: ReplState, arg: str) -> str:
    """Re-run the onboarding wizard and save the user config file.

    The saved file is read at the next start; the running session keeps
    its current services, model, and browser mode untouched. Ctrl-C or a
    closed stdin cancels without writing, and a shadowing ``./config.ini``
    is called out so the save is never silently ineffective.
    """
    from threetoks.config import find_config_path
    from threetoks.onboard import run_onboarding
    try:
        path = run_onboarding()
    except (EOFError, KeyboardInterrupt):
        return fg(AMBER, "setup cancelled — nothing saved", state.enabled)
    if find_config_path() != path:
        return fg(AMBER, f"saved — {path}, but {find_config_path()} takes "
                  "precedence here and will be read instead", state.enabled)
    return dim(f"saved — {path} (applies next start; /model, /browser and "
               "/voice on pick up changes this session)", state.enabled)


_VOICE_ARGS = ("", "on", "off", "debug")  # accepted /voice arguments


def _close_voice(state: ReplState) -> None:
    """Stop any live voice session and return to keyboard input."""
    if state.voice is not None:
        state.voice.close()
        state.voice = None


def _reload_voice_config(state: ReplState) -> None:
    """Re-read the saved ``[voice]`` section so a fresh /setup applies now.

    Without this the session would keep the config captured at startup and
    ``/voice on`` would ignore paths the user just saved. ``load_config``
    is total — a missing or malformed file degrades to the built-in
    defaults rather than raising, exactly as the next start would — so the
    guard here only covers genuinely unexpected errors, and keeps the
    startup section when one happens.
    """
    try:
        from threetoks.config import load_config
        state.voice_config = load_config().voice
    except Exception:  # noqa: BLE001 - a bad config must not block voice
        return


def _toggle_voice_debug(state: ReplState) -> str:
    """Flip per-transcript tracing on the live session; report the state."""
    if state.voice is None:
        return dim("voice is off — /voice on first", state.enabled)
    state.voice.debug = not state.voice.debug
    status = "on" if state.voice.debug else "off"
    return dim(f"voice debug {status}", state.enabled)


def _voice_on(state: ReplState) -> str:
    """Build a session from the freshly reloaded config; report the notes.

    The config is only re-read on the off-to-on edge, so a live session
    is told how to pick up a change rather than being left to imply it
    already did.
    """
    if state.voice is not None:
        return dim("voice is already on — /voice off then on to re-read "
                   "the config", state.enabled)
    _reload_voice_config(state)
    from threetoks.voice.session import make_voice_session
    try:
        state.voice, notes = make_voice_session(state.voice_config)
    except Exception as error:  # noqa: BLE001 - optional extras, any failure
        return fg(AMBER, f"voice unavailable: {error}", state.enabled)
    rows = [dim("voice on — speak; Ctrl-C returns to the keyboard",
                state.enabled)]
    rows.extend(fg(AMBER, note, state.enabled) for note in notes)
    return "\n".join(rows)


def _cmd_voice(state: ReplState, arg: str) -> str:
    """Toggle voice mode: microphone in and/or spoken answers out.

    ``/voice on`` re-reads the saved ``[voice]`` config, then builds a
    session from whichever voice extras are installed (stt and tts degrade
    independently); ``/voice off`` returns to the keyboard; ``/voice
    debug`` traces raw transcripts and gate drops; bare ``/voice`` reports
    the current state. Failures surface as a styled message.
    """
    want = arg.strip().lower()
    if want not in _VOICE_ARGS:
        return fg(AMBER, "usage: /voice [on|off|debug]", state.enabled)
    if want == "":
        status = "on" if state.voice is not None else "off"
        return dim(f"voice is {status}", state.enabled)
    if want == "off":
        _close_voice(state)
        return dim("voice off — keyboard input", state.enabled)
    if want == "debug":
        return _toggle_voice_debug(state)
    return _voice_on(state)


_DEBUG_ARGS = ("", "on", "off")  # accepted /debug arguments


def _cmd_debug(state: ReplState, arg: str) -> str:
    """Toggle the raw-answer and token-dump lines under every ticker decision.

    Distinct from ``/voice debug``, which traces microphone transcripts:
    this one shows what the model actually completed, so a decision that
    renders ``—`` names its own cause instead of just looking broken.
    Bare ``/debug`` reports the current state.
    """
    want = arg.strip().lower()
    if want not in _DEBUG_ARGS:
        return fg(AMBER, "usage: /debug [on|off]", state.enabled)
    if want == "":
        status = "on" if state.debug else "off"
        return dim(f"debug is {status}", state.enabled)
    state.debug = want == "on"
    if state.policy is not None and hasattr(state.policy, "debug"):
        state.policy.debug = state.debug
    if not state.debug:
        return dim("debug off", state.enabled)
    return dim("debug on — raw model answer + token dump under each decision",
               state.enabled)


def _cmd_quit(state: ReplState, arg: str) -> str:
    """Flip the running flag so the loop exits after this command."""
    state.running = False
    return gradient(SIGN_OFF, state.enabled)


_COMMANDS = {"/help": _cmd_help, "/agents": _cmd_agents, "/deep": _cmd_deep,
             "/browser": _cmd_browser, "/model": _cmd_model,
             "/voice": _cmd_voice, "/debug": _cmd_debug,
             "/setup": _cmd_setup, "/quit": _cmd_quit}


def handle_command(state: ReplState, line: str) -> str:
    """Dispatch a ``/command`` line; return the styled reply string.

    Unknown commands report themselves; handlers mutate ``state`` in place.
    """
    verb, _, arg = line.strip().partition(" ")
    handler = _COMMANDS.get(verb.lower())
    if handler is None:
        return fg(AMBER, f"unknown command: {verb} (try /help)", state.enabled)
    return handler(state, arg)


def _collect_result(outcome: dict, decisions: int, tokens: int,
                    seconds: float) -> dict:
    """Merge episode accounting into the agent outcome for the panel."""
    merged = dict(outcome)
    merged.setdefault("decisions", decisions)
    merged.setdefault("tokens", tokens)
    merged.setdefault("seconds", seconds)
    return merged


class _TickerCounter:
    """Streams ticker lines and tallies decisions/tokens as events arrive."""

    def __init__(self, state: ReplState, sink):
        """Store the REPL state (for color) and an output callable."""
        self.state = state
        self.sink = sink
        self.decisions = 0
        self.tokens = 0
        self.model_seconds = 0.0
        self.retries = 0

    def __call__(self, event: dict) -> None:
        """Trace ``on_event`` hook: print one ticker line, update tallies.

        With ``/debug`` on, each ticker line is followed by the raw
        completion it was parsed from, and a token-level dump of the
        full prompt and result for every model call.
        """
        if event.get("debug"):
            if self.state.debug:
                self.sink(build_debug_tokens_line(event, self.state.enabled))
            return
        self.decisions += 1
        self.tokens += int(event.get("out_tokens", 0) or 0)
        self.model_seconds += _num(event.get("wall_s"), float)
        if _num(event.get("attempt"), int) > 0:
            self.retries += 1
        self.sink(build_ticker_line(event, self.state.enabled))
        if self.state.debug:
            self.sink(build_debug_line(event, self.state.enabled))

    def stats(self, seconds: float) -> dict:
        """Tallies plus the task wall time, ready for the task report."""
        return {"seconds": seconds, "calls": self.decisions,
                "tokens": self.tokens, "model_seconds": self.model_seconds,
                "retries": self.retries}


def _run_task(state: ReplState, task: str, route, sink) -> None:
    """Route, recall context, run the agent, remember, render the panel."""
    from threetoks.agents.router import route as route_fn
    router = route or route_fn
    counter = _TickerCounter(state, sink)
    state.policy.tracer.on_event = counter
    start = time.time()  # the task includes routing and recall
    spec = router(task, state.specs, state.policy)
    sink(dim(f"  {ZAP} routed {CHEVRON} ", state.enabled)
         + fg(VIOLET, spec.name, state.enabled))
    _recall_context(state, task, sink)
    outcome = spec.run(task, state.services, state.policy)
    elapsed = time.time() - start
    _remember(state, spec.name, task, outcome)
    result = _collect_result(outcome, counter.decisions, counter.tokens,
                             elapsed)
    sink(render_result(result, state.enabled))
    sink(render_task_report(counter.stats(elapsed), state.enabled))
    if state.voice is not None:  # speak AFTER the panel so text never lags
        state.voice.speak(str(result.get("answer") or ""))


def _recall_context(state: ReplState, task: str, sink) -> None:
    """Selector step: pick relevant past memories into ``services.recalled``."""
    from threetoks.memory import select_memories
    if state.memory is None:
        return
    recalled = select_memories(task, state.memory, state.policy)
    state.services.recalled = recalled
    if recalled:
        preview = "; ".join(entry["query"][:40] for entry in recalled)
        sink(dim(f"  {ZAP} recalled {len(recalled)} {CHEVRON} {preview}",
                 state.enabled))


def _remember(state: ReplState, agent: str, task: str, outcome: dict) -> None:
    """Save this turn's (query, final answer) under the agent that handled it.

    Junk never enters memory: empty, "(no answer)", and judge-rejected
    answers are refused by the deterministic ``worth_remembering`` gate.
    """
    from threetoks.memory import worth_remembering
    if state.memory is not None and worth_remembering(outcome):
        state.memory.remember(agent, task, str(outcome.get("answer") or ""))


def _read_line(read_input, prompt: str):
    """Read one prompt line; return None on Ctrl-C / Ctrl-D at the prompt."""
    try:
        return read_input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def _debug_sink(state: ReplState, sink):
    """A dim per-event writer while voice debug is on, else None."""
    if not state.voice.debug:
        return None
    return lambda line: sink(dim("  · voice: " + line, state.enabled))


def _listen_line(state: ReplState, read_input, prompt: str, sink):
    """Block on the microphone until a pertinent transcript arrives.

    Gated-out transcripts show as a dim ignored line; an accepted one is
    echoed like typed input. In debug mode every raw transcript, gate
    drop, and pertinence verdict is traced too. Ctrl-C while listening
    turns voice mode off and falls back to one keyboard read instead of
    quitting; any other failure (backend down mid-gate, audio device
    gone) renders as an error panel and does the same — the session must
    always survive.
    """
    from threetoks.voice.session import HEARD
    sink(dim(f"  {ZAP} listening {CHEVRON} speak now (Ctrl-C for keyboard)",
             state.enabled))
    debug_sink = _debug_sink(state, sink)
    while state.running:
        try:
            gated = state.voice.listen(state.policy, debug_sink)
        except KeyboardInterrupt:
            _close_voice(state)
            sink(dim("voice off — keyboard input", state.enabled))
            return _read_line(read_input, prompt)
        except Exception as error:  # noqa: BLE001 - never unwind the REPL
            sink(render_error(error, state.enabled))
            _close_voice(state)
            sink(dim("voice off — keyboard input", state.enabled))
            return _read_line(read_input, prompt)
        if gated is None:
            continue
        kind, text = gated
        if kind == HEARD:
            sink(dim(f"  {ZAP} heard {CHEVRON} ", state.enabled)
                 + fg(CYAN, text, state.enabled))
            return text
        sink(dim(f"  · ignored: {text}", state.enabled))
    return None


def _next_line(state: ReplState, read_input, prompt: str, sink):
    """One user input line: keyboard, or the microphone in voice mode."""
    if state.voice is None or not state.voice.hears:
        return _read_line(read_input, prompt)
    return _listen_line(state, read_input, prompt, sink)


def _dispatch(state: ReplState, line: str, sink) -> None:
    """Handle one non-empty input line: slash command or research task."""
    if line.startswith("/"):
        sink(handle_command(state, line))
        return
    try:
        _run_task(state, line, None, sink)
    except KeyboardInterrupt:
        sink(fg(AMBER, "  ✖ cancelled — back to prompt",
                state.enabled))
    except Exception as error:  # noqa: BLE001 - never show a traceback
        sink(render_error(error, state.enabled))


def _error_hint(error: Exception) -> str | None:
    """One actionable line for well-known failure shapes, else None."""
    import urllib.error
    if isinstance(error, FileNotFoundError):
        return ("a data file is missing — reinstall: uv tool install "
                "--force '3toks[web,browser] @ <repo root>'")
    if isinstance(error, ImportError):
        return ("missing optional dependency — install the extras: "
                "pip install '3toks[web,browser]'")
    if isinstance(error, (ConnectionError, urllib.error.URLError)):
        return ("is Ollama running? start it and pull the model: "
                "ollama pull qwen3.5:2b")
    if "API_KEY" in str(error):
        return ("export the key in your shell (and your shell profile, so "
                "it survives a new terminal), or set [llm] provider = "
                "ollama in config.ini")
    return None


def render_error(error: Exception, enabled: bool = True) -> str:
    """A styled ✖ panel for a run failure (ollama down, network, ...).

    Shows the exception type with its message plus, for well-known
    environment failures, a dim one-line fix hint. The session always
    survives — errors render, they never unwind the REPL.
    """
    kind = type(error).__name__
    rows = [fg(AMBER, "ERROR", enabled),
            fg(AMBER, f"{CROSS} {kind}: {error}", enabled)]
    hint = _error_hint(error)
    if hint:
        rows.append(dim(hint, enabled))
    return "\n".join(_spined(rows, enabled, color=AMBER))


def repl(state: ReplState, read_input=input, sink=print) -> None:
    """The interactive loop: banner already shown; read, dispatch, repeat."""
    prompt = prompt_text(state.enabled)
    while state.running:
        line = _next_line(state, read_input, prompt, sink)
        if line is None:
            sink("\n" + gradient(SIGN_OFF, state.enabled))
            return
        if line.strip():
            _dispatch(state, line.strip(), sink)


def _make_provider(config, sink):
    """Search provider, or None (with a notice) on a bare core install.

    The web extras (requests, bs4, markdownify) are optional; a missing
    import must degrade to a working TUI without the web agent, never a
    startup crash.
    """
    try:
        import threetoks.web.search as search_mod  # lazy: web extras
        from threetoks.web.search import auto_provider
    except ImportError:
        sink(dim("  web extras missing — web agent disabled "
                 "(install: uv tool install '3toks[web]')",
                 _color_enabled()))
        return None
    search_mod.MIN_SEARCH_INTERVAL_S = config.search.min_interval_s
    return auto_provider()


def _initial_fetcher(config, services, sink=print):
    """Set ``services.fetch_page`` from ``config.browser.mode`` at startup.

    Returns the live browser fetcher (to close at shutdown) for the Chrome
    modes, or ``None`` for plain HTTP. A failed browser start degrades to
    HTTP rather than aborting the session — but says so: a silent
    fallback reads as "the config is ignored" to the user watching for a
    window that never comes.
    """
    from threetoks.cli import fetch_page
    mode = config.browser.mode
    if mode == "http":
        services.fetch_page = fetch_page
        return None
    try:
        fetcher = _make_browser_fetcher(mode, config.browser.visible)
    except Exception as error:  # noqa: BLE001 - optional module; use HTTP
        services.fetch_page = fetch_page
        sink(fg(AMBER, f"  browser {mode} unavailable: {error} — "
                "falling back to http fetch", _color_enabled()))
        return None
    services.fetch_page = fetcher.fetch_page
    return fetcher


def unknown_family_notice(model: str) -> str | None:
    """A warning when the model's template family had to be guessed.

    A wrong raw-mode template fails silently — menus keep answering (one
    digit survives junk tokens) while every multi-line generation derails
    — so the guess must be surfaced, never assumed.
    """
    from threetoks.backend.base import detect_family

    _, known = detect_family(model)
    if known:
        return None
    return (f"⚠ unknown template family for '{model}' — driving it as "
            "ChatML, which may silently degrade generations. Known "
            "families: qwen/smollm (chatml), gemma, llama3, "
            "mistral/mixtral/llama2, phi3, deepseek-r1. The measured "
            "policy model is qwen3.5:2b.")


def _make_retriever(config, provider):
    """The opt-in code retrieval hook, or None (the default).

    Requires both the ``[code] retrieval`` config flag AND a working
    search provider (a bare install has neither and gets None).
    """
    if provider is None or not config.code.retrieval:
        return None
    from threetoks.code.retrieve import Retriever
    return Retriever(provider, cache_path=config.code.snippet_cache)


def _warn_bad_config(sink) -> None:
    """Say so when the config file was ignored and defaults took over.

    A file that fails to parse degrades to the built-in defaults rather
    than raising, which is otherwise invisible: the session simply runs
    on a config the user did not write (see config.config_problem).
    """
    from threetoks.config import config_problem
    problem = config_problem()
    if problem:
        sink(fg(AMBER, problem, _color_enabled()))


def _make_relay(config, sink):
    """A live Relay on a Raspberry Pi with the extra installed, else None."""
    from threetoks.relay import relay_available
    if not relay_available():
        return None
    from threetoks.relay import Relay
    try:
        return Relay(pin=config.relay.pin)
    except Exception as error:  # noqa: BLE001 - hardware; degrade, don't die
        sink(fg(AMBER, f"  relay unavailable: {error}", _color_enabled()))
        return None


def _make_capture(config):
    """A camera-frame callable when opencv is installed, else None."""
    from threetoks.camera import camera_available, capture_jpeg_b64
    if not camera_available():
        return None
    return lambda: capture_jpeg_b64(config.camera.index)


def build_state(model: str = None, sink=print):
    """Assemble live Services/specs/policy from the resolved config file.

    Imports of the backend and the agent registry stay inside this function
    so the module (and its tests) load without ollama or the registry. The
    ``model`` argument, when given, overrides the configured model.
    """
    from pathlib import Path

    from threetoks.agents import default_agents, optional_agents
    from threetoks.backend.providers import make_backend
    from threetoks.cli import make_policy_config
    from threetoks.config import load_config
    from threetoks.memory import MemoryStore
    from threetoks.policy import Policy
    from threetoks.services import Services
    from threetoks.trace import Tracer

    config = load_config()
    _warn_bad_config(sink)
    provider = _make_provider(config, sink)
    services = Services(provider=provider,
                        files_root=Path(config.files.root),
                        max_research_rounds=config.research.max_rounds,
                        retriever=_make_retriever(config, provider),
                        relay=_make_relay(config, sink),
                        capture_frame=_make_capture(config))
    fetcher = _initial_fetcher(config, services, sink)
    chosen_model = model or config.llm.model
    specs = default_agents(services) + optional_agents(services, chosen_model)
    if provider is None:  # bare install: never route to the web agent
        specs = [spec for spec in specs if spec.name != "web"]
    factory = lambda name: Policy(make_backend(config.llm),
                                  make_policy_config(name), Tracer(None),
                                  debug=False)
    if config.llm.provider == "ollama":  # families only exist in raw mode
        notice = unknown_family_notice(chosen_model)
        if notice:
            sink(notice)
    policy = factory(chosen_model)
    memory = MemoryStore.load(config.memory.path) if config.memory.enabled \
        else None
    state = ReplState(services, specs, policy, chosen_model, factory,
                     enabled=_color_enabled(), fetcher=fetcher,
                     memory=memory, memory_path=config.memory.path,
                     browser_visible=config.browser.visible,
                     voice_config=config.voice,
                     llm_provider=config.llm.provider)
    # Re-bind the factory so future model swaps inherit the current debug flag
    state.policy_factory = lambda name: Policy(make_backend(config.llm),
                                                make_policy_config(name),
                                                Tracer(None),
                                                debug=state.debug)
    return state


def main(model: str = None) -> None:
    """Onboard on a configless first run, print the banner, run the REPL."""
    from threetoks.config import find_config_path
    if find_config_path() is None:
        from threetoks.onboard import run_onboarding
        try:
            run_onboarding()
        except EOFError:  # piped/closed stdin: run on defaults, save nothing
            print(dim("setup skipped (no interactive input) — using "
                      "built-in defaults", _color_enabled()))
        except KeyboardInterrupt:
            print()
            print(dim("setup cancelled — nothing saved", _color_enabled()))
            return
    try:
        state = build_state(model)
    except RuntimeError as error:  # e.g. a provider without its API key
        print(render_error(error, _color_enabled()))
        return
    print(build_banner(state.model, len(state.specs), state.enabled))
    if state.voice_config is not None and state.voice_config.enabled:
        print(handle_command(state, "/voice on"))
    try:
        repl(state)
    finally:
        _close_voice(state)
        _close_active_fetcher(state)
        if state.memory is not None:
            state.memory.dump(state.memory_path)


# Demo ticker events; the last one is a decision truncated by the node's
# token cap, the failure /debug exists to explain.
_DEMO_EVENTS = (
    {"node": "menu", "value": "open result 2: Anthropic pricing",
     "valid": True, "out_tokens": 2, "wall_s": 0.31},
    {"node": "pick_many", "value": [3, 7], "valid": True, "resolved":
     ["Opus pricing was updated in June.", "Input tokens cost $15/M."],
     "out_tokens": 5, "wall_s": 0.62},
    {"tag": "curate", "node": "menu", "value": "keep note 3",
     "valid": True, "out_tokens": 1, "wall_s": 0.18},
    {"node": "short_text", "value": "anthropic api price per token",
     "valid": True, "out_tokens": 8, "wall_s": 1.55},
    {"node": "menu", "valid": False, "out_tokens": 3, "wall_s": 0.22,
     "attempt": 2, "done_reason": "length",
     "raw_text": "Okay, so the user is asking me to"},
)


def _demo() -> None:
    """Non-interactive render of every TUI piece for eyeballing."""
    enabled = _color_enabled()
    print(build_banner("qwen3.5:2b", 2, enabled))
    print()
    for event in _DEMO_EVENTS:
        print(build_ticker_line(event, enabled))
    print(build_debug_line(_DEMO_EVENTS[-1], enabled))  # as /debug on shows it
    print(dim(f"  {ZAP} routed {CHEVRON} ", enabled)
          + fg(VIOLET, "web", enabled))
    print()
    result = {"agent": "web", "answer": "Claude Opus costs $15 per million "
              "input tokens.", "notes": "[1] Opus pricing page\n[2] docs",
              "decisions": 9, "tokens": 64, "seconds": 11.2, "rounds": 2}
    print(render_result(result, enabled))
    print(render_task_report({"seconds": 11.2, "calls": 9, "tokens": 64,
                              "model_seconds": 4.3, "retries": 1}, enabled))
    print()
    print(render_error(RuntimeError("ollama backend unreachable"), enabled))
    print()
    print(prompt_text(enabled) + "…")
    print(gradient(SIGN_OFF, enabled))


if __name__ == "__main__":
    _demo()
