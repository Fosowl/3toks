"""Neon-dawn cyberpunk terminal UI for the ThreeToks agent mesh.

The killer view: the agent makes ~1-token menu decisions several times a
second, and this TUI streams each one live as a row of a progress tree
(:mod:`threetoks.progress`), so the user watches the model think: which
part of the task runs, which action it chose, and what each part cost.

One visual identity holds it together: a cyan → lavender → hot-pink
gradient ramp on hairline rules and the vertical spine of result panels,
while the startup banner (:mod:`threetoks.banner`) draws a blue ramp per
launch. All structural chrome sits in dim gray so only the model's
decisions and answers carry color.

Rendering is stdlib-only raw ANSI; every escape sequence hides behind a
small helper (:func:`fg`, :func:`dim`, :func:`gradient`) so there is no
inline escape soup. ``NO_COLOR`` and non-tty output fall back to plain
text. Pure formatting/dispatch functions (:func:`render_result`,
:func:`handle_command`) are kept free of the input loop so they can be
unit-tested offline without an LLM or a network.
"""
import textwrap
import time
from typing import TYPE_CHECKING

from threetoks.banner import build_banner
from threetoks.live_config import sync_config
from threetoks.progress import DEMO_EVENTS, ProgressTree, as_number
from threetoks.session import (
    ReplState,
    build_state,
    close_fetcher,
    start_voice,
    stop_voice,
    swap_model,
    switch_fetcher,
)
from threetoks.theme import (
    AMBER,
    CHEVRON,
    CROSS,
    CYAN,
    GRAY_LIGHT,
    GREEN,
    NOTE_BULLET,
    PANEL_WIDTH,
    PINK,
    SPINE,
    VIOLET,
    WHITE,
    ZAP,
    color_enabled,
    dim,
    fg,
    gradient,
    ramp_at,
)
from threetoks.trace import span

if TYPE_CHECKING:  # the agent registry loads lazily at run time
    from threetoks.agents.base import AgentSpec

SIGN_OFF = "⌁ mesh offline — tokens saved, see you"


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    """Wrap plain text to the panel width; never truncates content."""
    return textwrap.wrap(text, width=width,
                         subsequent_indent=indent) or [""]


def _spined(rows: list[str], enabled: bool, color: int | None = None) -> list[str]:
    """Prefix each row with a ▍ spine, gradient top→bottom unless fixed.

    The spine is the panel's only frame: a thin neon edge stepping through
    the ramp, or a single ``color`` for uniform panels (e.g. errors).
    """
    out = []
    for index, row in enumerate(rows):
        code = color if color is not None else ramp_at(index, len(rows))
        out.append((fg(code, " " + SPINE, enabled) + " " + row).rstrip())
    return out


def _stats_line(result: dict, enabled: bool) -> str:
    """Dim ``N decisions · T tok · S.s`` line, rounds appended if present."""
    parts = [f"{result.get('decisions', 0)} decisions",
             f"{result.get('tokens', 0)} tok",
             f"{as_number(result.get('seconds'), float):.1f}s"]
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
    """Show the script's captured stdout under the code it came from.

    The result panel uses these rows so a working script does not look
    inert. _panel_rows calls it with the agent result.
    It clips long output with a "more lines" count and shows nothing when
    the output is empty or is already the answer (code compute mode).

    Args:
        result (dict): the agent result, from _panel_rows.
        enabled (bool): color flag, from _panel_rows.

    Returns:
        list[str]: panel rows, empty when there is nothing new to show.
    """
    output = str(result.get("output") or "").rstrip()
    if not output or output.strip() == str(result.get("answer") or "").strip():
        return []  # compute: the output IS the answer
    from threetoks.code.edit.nav import clip_lines
    shown, hidden = clip_lines(output.splitlines(), OUTPUT_MAX_LINES)
    rows = ["", dim("output when run:", enabled)]
    for line in shown:
        rows.extend(fg(GREEN, seg, enabled)
                    for seg in _wrap(line, PANEL_WIDTH - 3))
    if hidden:
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
    seconds = as_number(stats.get("seconds"), float)
    calls = as_number(stats.get("calls"), int)
    model_s = as_number(stats.get("model_seconds"), float)
    head = f"  {ZAP} task report {CHEVRON} {seconds:.1f}s"
    if model_s and seconds:
        share = min(100, round(100 * model_s / seconds))
        head += f" · model {model_s:.1f}s ({share}%)"
    if not calls:
        return dim(head + " · no model calls", enabled)
    tokens = as_number(stats.get("tokens"), int)
    parts = [f"{calls} calls", f"{tokens} tok",
             f"avg {tokens / calls:.1f} tok/call",
             f"{(model_s or seconds) / calls:.2f}s/call"]
    retries = as_number(stats.get("retries"), int)
    if retries:
        parts.append(f"{retries} retries")
    return "\n".join((dim(head, enabled),
                      dim("    " + " · ".join(parts), enabled)))


def prompt_text(enabled: bool = True) -> str:
    """The two-line input prompt ``╭╴you`` / ``╰╴❯ ``."""
    return (dim("╭╴", enabled) + fg(GRAY_LIGHT, "you", enabled) + "\n"
            + dim("╰╴", enabled) + fg(CYAN, CHEVRON + " ", enabled))


# --- slash-command dispatch ----------------------------------------------


def _cmd_help(state: ReplState, arg: str) -> str:
    """Styled list of the available slash commands."""
    rows = [
        ("/help", "show this command list"),
        ("/agents", "list registered agents"),
        ("/deep N", "set research rounds to N"),
        ("/browser MODE [on-screen]", "fetch via http|plain"),
        ("/model NAME", "swap the deciding model"),
        ("/voice [on|off|debug]", "talk instead of typing (voice extras)"),
        ("/debug [on|off|full]",
         "raw model answer per decision; full: whole prompts"),
        ("/config", "open the config screen (alias /setup)"),
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


def _cmd_model(state: ReplState, arg: str) -> str:
    """Rebuild the policy for a new model name via the policy factory.

    The optional agents are re-derived too: swapping to/from a
    vision-capable model registers/unregisters the ``look`` agent.
    """
    name = arg.strip()
    if not name:
        return fg(AMBER, "usage: /model NAME", state.enabled)
    try:
        notice = swap_model(state, name)
    except RuntimeError as error:  # e.g. a provider without its API key
        return fg(AMBER, f"model not changed: {error}", state.enabled)
    if notice:
        return fg(AMBER, notice, state.enabled)
    return dim(f"model = {name}", state.enabled)


_BROWSER_MODES = ("http", "plain")
_WINDOW_WORDS = {"on-screen": True, "headless": False}


def _cmd_browser(state: ReplState, arg: str) -> str:
    """Swap ``services.fetch_page`` between the http and plain fetchers.

    ``/browser http`` restores the plain-HTTP fetch. ``plain`` drives a
    real Chrome; visibility defaults to the config's ``[browser]
    visible`` and an explicit ``on-screen`` / ``headless`` argument
    overrides it. A visible window lets the user solve a bot check. The
    browser module is optional, so every failure surfaces as a styled
    message, not a traceback.
    """
    parts = arg.lower().split()
    mode = parts[0] if parts else ""
    if mode not in _BROWSER_MODES:
        return fg(AMBER,
                  "usage: /browser http|plain [on-screen|headless]",
                  state.enabled)
    window = parts[1] if len(parts) > 1 else ""
    visible = _WINDOW_WORDS.get(window, state.config.browser.visible)
    try:
        switch_fetcher(state, mode, visible)
    except Exception as error:  # noqa: BLE001 - optional module, any failure
        return fg(AMBER, f"browser unavailable: {error} — reverted to http",
                  state.enabled)
    if mode == "http":
        return dim("browser http — plain HTTP fetch", state.enabled)
    shown = "on-screen" if visible else "headless"
    return dim(f"browser {mode} — Chrome ({shown})", state.enabled)


def _cmd_setup(state: ReplState, arg: str) -> str:
    """Open the config screen; on a dumb terminal run the line wizard.

    A real terminal gets the arrow-key screen, which saves into the
    active file. Piped or dumb terminals get the classic wizard. Either
    way the saved file is applied to this session at once. Ctrl-C or a
    closed stdin cancels without writing, and a shadowing
    ``./config.ini`` is called out so the save is never silently
    ineffective.
    """
    from threetoks.keys import is_interactive
    if is_interactive():
        from threetoks.config_ui import config_command
        return config_command(state)
    from threetoks.config import find_config_path
    from threetoks.onboard import run_onboarding
    try:
        path = run_onboarding()
    except (EOFError, KeyboardInterrupt):
        return fg(AMBER, "setup cancelled — nothing saved", state.enabled)
    if find_config_path() != path:
        return fg(AMBER, f"saved — {path}, but {find_config_path()} takes "
                  "precedence here and will be read instead", state.enabled)
    applied = sync_config(state)
    saved = dim(f"saved — {path}", state.enabled)
    return saved + ("\n" + applied if applied else "")


_VOICE_ARGS = ("", "on", "off", "debug")  # accepted /voice arguments


def _toggle_voice_debug(state: ReplState) -> str:
    """Flip per-transcript tracing on the live session; report the state."""
    if state.voice is None:
        return dim("voice is off — /voice on first", state.enabled)
    state.voice.debug = not state.voice.debug
    status = "on" if state.voice.debug else "off"
    return dim(f"voice debug {status}", state.enabled)


def _voice_on(state: ReplState) -> str:
    """Build a session from the current ``[voice]`` config; report notes.

    The REPL applies a saved config before it runs each line, so
    ``state.config.voice`` already holds the last saved section here.
    """
    if state.voice is not None:
        return dim("voice is already on", state.enabled)
    try:
        notes = start_voice(state, state.config.voice)
    except Exception as error:  # noqa: BLE001 - optional extras, any failure
        return fg(AMBER, f"voice unavailable: {error}", state.enabled)
    rows = [dim("voice on — speak; Ctrl-C returns to the keyboard",
                state.enabled)]
    rows.extend(fg(AMBER, note, state.enabled) for note in notes)
    return "\n".join(rows)


def _cmd_voice(state: ReplState, arg: str) -> str:
    """Toggle voice mode: microphone in and/or spoken answers out.

    ``/voice on`` builds a session from the current ``[voice]`` config and
    whichever voice extras are installed (stt and tts degrade
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
        stop_voice(state)
        return dim("voice off — keyboard input", state.enabled)
    if want == "debug":
        return _toggle_voice_debug(state)
    return _voice_on(state)


_DEBUG_ARGS = ("", "on", "off", "full")  # accepted /debug arguments


def _cmd_debug(state: ReplState, arg: str) -> str:
    """Toggle the raw-answer and token-dump lines under every tree decision.

    Distinct from ``/voice debug``, which traces microphone transcripts:
    this one shows what the model actually completed, so a decision that
    renders ``—`` names its own cause instead of just looking broken.
    ``full`` prints the complete prompt of every model call — the exact
    request, menu options, and prefill the decision saw — instead of the
    head+tail summary. Bare ``/debug`` reports the current state.
    """
    want = arg.strip().lower()
    if want not in _DEBUG_ARGS:
        return fg(AMBER, "usage: /debug [on|off|full]", state.enabled)
    if want == "":
        status = ("off" if not state.debug
                  else "full" if state.debug_full else "on")
        return dim(f"debug is {status}", state.enabled)
    state.debug = want != "off"
    state.debug_full = want == "full"
    if state.policy is not None and hasattr(state.policy, "debug"):
        state.policy.debug = state.debug
    if not state.debug:
        return dim("debug off", state.enabled)
    if state.debug_full:
        return dim("debug full — complete prompt of every model call",
                   state.enabled)
    return dim("debug on — raw model answer + token dump under each decision",
               state.enabled)


def _cmd_quit(state: ReplState, arg: str) -> str:
    """Flip the running flag so the loop exits after this command."""
    state.running = False
    return gradient(SIGN_OFF, state.enabled)


_COMMANDS = {"/help": _cmd_help, "/agents": _cmd_agents, "/deep": _cmd_deep,
             "/browser": _cmd_browser, "/model": _cmd_model,
             "/voice": _cmd_voice, "/debug": _cmd_debug,
             "/setup": _cmd_setup, "/config": _cmd_setup, "/quit": _cmd_quit}


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


def _run_task(state: ReplState, task: str, route, sink) -> None:
    """Route, recall context, run the agent, remember, render the panel.

    The progress tree gets every trace event and agent progress line, so
    the task streams as a tree; the panel and task report follow it.
    """
    tree = ProgressTree(state, sink)
    state.policy.tracer.on_event = tree
    state.services.on_progress = tree.note
    start = time.time()  # the task includes routing and recall
    spec, outcome = _route_and_run(state, task, route)
    elapsed = time.time() - start
    _remember(state, spec.name, task, outcome)
    stats = tree.stats(elapsed)
    result = _collect_result(outcome, stats["calls"], stats["tokens"],
                             elapsed)
    sink(render_result(result, state.enabled))
    sink(render_task_report(stats, state.enabled))
    if state.voice is not None:  # speak AFTER the panel so text never lags
        state.voice.speak(str(result.get("answer") or ""))


def _route_and_run(state: ReplState, task: str,
                   route) -> "tuple[AgentSpec, dict]":
    """Pick the agent, recall memories and run the agent, each in a span.

    _run_task calls this with the REPL state and the user request.
    Each part is one top-level branch of the progress tree; the branch
    end row shows the chosen agent, the memory count, or the status.

    Args:
        state (ReplState): the REPL state; its policy tracer gets spans.
        task (str): the user request, from the input line.
        route (callable | None): a router override from tests; None
            uses threetoks.agents.router.route.

    Returns:
        tuple[AgentSpec, dict]: the chosen agent and its result.
    """
    from threetoks.agents.base import outcome_status
    from threetoks.agents.router import route as route_fn
    tracer = state.policy.tracer
    with span(tracer, "pick an agent") as picked:
        spec = (route or route_fn)(task, state.specs, state.policy)
        picked["result"] = spec.name
    _recall_context(state, task)
    with span(tracer, f"{spec.name} agent") as finished:
        outcome = spec.run(task, state.services, state.policy)
        finished["result"] = outcome_status(outcome)
    return spec, outcome


def _recall_context(state: ReplState, task: str) -> None:
    """Put the past answers that fit this request into the services.

    _route_and_run calls this after routing, before the agent runs.
    It does nothing without a memory store. Otherwise the model picks
    the relevant memories inside a "recall past answers" span, and the
    picks go to ``services.recalled`` for the agent.

    Args:
        state (ReplState): the REPL state, with memory, policy, services.
        task (str): the user request, from the input line.
    """
    from threetoks.memory import select_memories
    if state.memory is None:
        return
    with span(state.policy.tracer, "recall past answers") as found:
        recalled = select_memories(task, state.memory, state.policy)
        found["result"] = (f"{len(recalled)} relevant" if recalled
                           else "none relevant")
    state.services.recalled = recalled


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


CONFIRM_DIFF_MAX_LINES = 200  # one function's diff fits whole
CONFIRM_YES = ("y", "yes")


def make_confirm_edit(state: ReplState, read_input, sink):
    """Build the callback that asks the user to accept one code edit.

    The code agent calls the callback before it writes a file.
    repl calls this builder with its state, input reader and output sink.
    The callback prints the clipped diff, asks a y/N question and gives
    True only for "y" or "yes"; EOF and Ctrl-C give False.

    Args:
        state (ReplState): the REPL state (color flag), from repl.
        read_input (callable): prompt -> line reader, from repl.
        sink (callable): line printer, from repl.

    Returns:
        callable: (rel_path: str, diff: str) -> bool.
    """
    def confirm_edit(rel_path: str, diff: str) -> bool:
        sink(_diff_preview(diff, state.enabled))
        reply = _read_line(read_input,
                           f"  apply this edit to {rel_path}? [y/N] ")
        return (reply or "").strip().lower() in CONFIRM_YES
    return confirm_edit


def make_confirm_command(state: ReplState, read_input, sink):
    """Build the callback that asks the user to run one shell command.

    The code agent's shell mode calls the callback before it runs a
    command. repl calls this builder with its state, input reader and
    output sink. The callback prints the command and the folder, asks a
    y/N question and gives True only for "y" or "yes"; EOF and Ctrl-C
    give False.

    Args:
        state (ReplState): the REPL state (color flag, files root), from
            repl.
        read_input (callable): prompt -> line reader, from repl.
        sink (callable): line printer, from repl.

    Returns:
        callable: (command: str) -> bool.
    """
    def confirm_command(command: str) -> bool:
        sink(fg(AMBER, f"  $ {command}", state.enabled))
        reply = _read_line(read_input, f"  run this command in "
                           f"{state.services.files_root}? [y/N] ")
        return (reply or "").strip().lower() in CONFIRM_YES
    return confirm_command


def _diff_preview(diff: str, enabled: bool) -> str:
    """Color a unified diff and clip it for the confirm question.

    The user reads this preview before saying yes or no to an edit.
    The confirm callback calls it with the diff from the code agent.
    It shows added lines green, removed lines pink, other lines dim, and
    adds an "N more lines" note after CONFIRM_DIFF_MAX_LINES lines.

    Args:
        diff (str): unified diff text, from the code agent.
        enabled (bool): color flag, from the REPL state.

    Returns:
        str: the printable preview.
    """
    from threetoks.code.edit.nav import clip_lines
    lines, hidden = clip_lines(diff.splitlines(), CONFIRM_DIFF_MAX_LINES)
    shown = [_diff_line(line, enabled) for line in lines]
    if hidden:
        shown.append(dim(f"  … {hidden} more lines", enabled))
    return "\n".join(shown)


def _diff_line(line: str, enabled: bool) -> str:
    """Color one unified diff line by its kind.

    The diff preview uses this to make changes easy to see.
    _diff_preview calls it once per shown line.
    It colors "+" lines green and "-" lines pink; headers and context
    lines are dim.

    Args:
        line (str): one diff line, from _diff_preview.
        enabled (bool): color flag, from _diff_preview.

    Returns:
        str: the indented, colored line.
    """
    is_header = line.startswith(("+++", "---"))
    if line.startswith("+") and not is_header:
        return fg(GREEN, "  " + line, enabled)
    if line.startswith("-") and not is_header:
        return fg(PINK, "  " + line, enabled)
    return dim("  " + line, enabled)


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
            stop_voice(state)
            sink(dim("voice off — keyboard input", state.enabled))
            return _read_line(read_input, prompt)
        except Exception as error:  # noqa: BLE001 - never unwind the REPL
            sink(render_error(error, state.enabled))
            stop_voice(state)
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
    """Run the interactive loop: read a line, dispatch it, repeat.

    main uses this after the banner is shown; tests call it with a
    scripted reader and a list sink.
    It also gives the code agent an edit-confirm callback that uses the
    same reader and sink as the prompt.

    Args:
        state (ReplState): the live REPL state, from main.
        read_input (callable): prompt -> line reader, default input.
        sink (callable): line printer, default print.

    Returns:
        None.
    """
    state.services.confirm_edit = make_confirm_edit(state, read_input, sink)
    state.services.confirm_command = make_confirm_command(state, read_input,
                                                          sink)
    prompt = prompt_text(state.enabled)
    while state.running:
        line = _next_line(state, read_input, prompt, sink)
        if line is None:
            sink("\n" + gradient(SIGN_OFF, state.enabled))
            return
        if line.strip():
            applied = sync_config(state)  # a save from anywhere applies now
            if applied:
                sink(applied)
            _dispatch(state, line.strip(), sink)


def main(model: str | None = None) -> None:
    """Onboard on a configless first run, print the banner, run the REPL."""
    from threetoks.config import find_config_path
    from threetoks.config_ui import first_start
    from threetoks.line_input import terminal_reader
    if find_config_path() is None and not first_start():
        return
    try:
        state = build_state(model)
    except RuntimeError as error:  # e.g. a provider without its API key
        print(render_error(error, color_enabled()))
        return
    print(build_banner(state.model, len(state.specs), state.enabled))
    if state.config.voice.enabled:
        print(handle_command(state, "/voice on"))
    try:
        repl(state, read_input=terminal_reader(_COMMANDS))
    finally:
        stop_voice(state)
        close_fetcher(state)
        if state.memory is not None:
            state.memory.dump(state.config.memory.path)


def _demo() -> None:
    """Non-interactive render of every TUI piece for eyeballing."""
    from types import SimpleNamespace
    enabled = color_enabled()
    print(build_banner("qwen3.5:2b", 2, enabled))
    print()
    flags = SimpleNamespace(enabled=enabled, debug=False, debug_full=False)
    tree = ProgressTree(flags, print)
    for event in DEMO_EVENTS:
        tree(event)
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
    print(_diff_preview("--- a/mod.py\n+++ b/mod.py\n@@ -1,2 +1,2 @@\n"
                        " def add(a, b):\n-    return a - b\n"
                        "+    return a + b", enabled))
    print("  apply this edit to mod.py? [y/N] …")
    print()
    print(prompt_text(enabled) + "…")
    print(gradient(SIGN_OFF, enabled))


if __name__ == "__main__":
    _demo()
