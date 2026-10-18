"""Apply a saved config to the running chat session, without a restart.

The REPL calls :func:`sync_config` before each line, and ``/config`` and
``/setup`` call it after a save. When the config file changed since the
session last read it, the new file is loaded and every changed section
is applied to the live parts: policy, browser, search provider, research
budgets, file root, memory, code retrieval, voice, camera and relay.

Each applier reads the running value of its section from
``state.config`` and raises when the saved value cannot start (a chat
provider whose API key is missing, a Chrome that will not launch, voice
models that will not load). Such a section keeps its old value in
``state.config`` and says why; the next save tries it again, and the
rest of the save still applies.
"""
from dataclasses import replace
from pathlib import Path

from threetoks.config import config_stamp, load_config, parse_error, resolve_path
from threetoks.fields import changed_fields, config_warnings, format_value, get_value
from threetoks.session import (
    load_memory,
    make_capture,
    make_provider,
    make_relay,
    make_retriever,
    refresh_optional_agents,
    start_voice,
    stop_voice,
    swap_model,
    switch_fetcher,
)
from threetoks.theme import AMBER, dim, fg


def _apply_llm(state, new) -> list:
    """Rebuild the policy; a session-chosen model stays unless saved."""
    saved = new.llm.model != state.config.llm.model
    notice = swap_model(state, new.llm.model if saved else state.model,
                        new.llm)
    return [fg(AMBER, notice, state.enabled)] if notice else []


def _apply_browser(state, new) -> list:
    """Switch the fetcher like ``/browser`` does."""
    switch_fetcher(state, new.browser.mode, new.browser.visible)
    return []


def _apply_search(state, new) -> list:
    """Probe the new SearXNG URL and rebuild what uses the provider."""
    lines = []
    provider = make_provider(new, lines.append)
    state.services.provider = provider
    state.services.retriever = make_retriever(new, provider)
    return lines


def _apply_research(state, new) -> list:
    """Set the changed budgets; a ``/deep`` choice stays otherwise."""
    old = state.config.research
    if new.research.max_rounds != old.max_rounds:
        state.services.max_research_rounds = new.research.max_rounds
    if new.research.max_steps != old.max_steps:
        state.services.max_research_steps = new.research.max_steps
    return []


def _apply_files(state, new) -> list:
    """Point the file and code agents at the new root."""
    root = Path(new.files.root)
    state.services.files_root = root
    if root.is_dir():
        return []
    return [fg(AMBER, f"[files] root {root} is not a directory",
               state.enabled)]


def _apply_memory(state, new) -> list:
    """Save the open memory to its old file, then open the new one."""
    if state.memory is not None:
        state.memory.dump(state.config.memory.path)
    state.memory = load_memory(new)
    return []


def _apply_code(state, new) -> list:
    """Turn code retrieval on or off, or move its cache."""
    state.services.retriever = make_retriever(new, state.services.provider)
    return []


def _apply_voice(state, new) -> list:
    """Restart a live voice session; follow a changed ``enabled``."""
    was_on = state.voice is not None
    switched = new.voice.enabled != state.config.voice.enabled
    stop_voice(state)
    if not (new.voice.enabled if switched else was_on):
        return []
    notes = start_voice(state, new.voice)  # raises: keep the old [voice]
    return [fg(AMBER, note, state.enabled) for note in notes]


def _apply_camera(state, new) -> list:
    """Capture from the new camera; the look agent follows."""
    state.services.capture_frame = make_capture(new)
    refresh_optional_agents(state, state.model)
    return []


def _apply_relay(state, new) -> list:
    """Free the old GPIO pin and drive the new one."""
    lines = []
    if state.services.relay is not None:
        state.services.relay.close()
    state.services.relay = make_relay(new, lines.append)
    refresh_optional_agents(state, state.model)
    return lines


# Order matters: code reads the provider [search] builds, and [llm]
# goes last so its model swap sees the camera and relay agents.
APPLIERS = (("search", _apply_search), ("code", _apply_code),
            ("research", _apply_research), ("files", _apply_files),
            ("memory", _apply_memory), ("browser", _apply_browser),
            ("camera", _apply_camera), ("relay", _apply_relay),
            ("voice", _apply_voice), ("llm", _apply_llm))


def apply_config(state, new) -> list:
    """Apply each changed section of ``new`` to the live session.

    :func:`sync_config` calls this with a freshly loaded config. Each
    section is recorded in ``state.config`` as soon as it runs, so a
    Ctrl-C keeps what was done; a section whose applier raises keeps
    its old value, so the next save tries it again.

    Args:
        state (ReplState): the live session.
        new (ThreetoksConfig): the saved config to apply.

    Returns:
        list of str: styled notices from the appliers and failures.
    """
    lines = []
    for name, applier in APPLIERS:
        if getattr(state.config, name) == getattr(new, name):
            continue
        try:
            lines.extend(applier(state, new))
        except Exception as error:  # noqa: BLE001 - keep the old section
            lines.append(fg(AMBER, f"[{name}] not applied: {error}",
                            state.enabled))
            continue
        state.config = replace(state.config, **{name: getattr(new, name)})
    return lines


def _change_lines(changes: tuple, state) -> list:
    """One dim line per setting that now runs with its saved value."""
    return [dim(f"  {item.dotted} = "
                f"{format_value(item, get_value(state.config, item))}",
                state.enabled)
            for item in changes]


def _reload(state) -> str:
    """Load the saved file and apply what changed; return the report."""
    error = parse_error()
    if error:
        return fg(AMBER, f"⚠ {resolve_path()} has an error, so this "
                  f"session keeps its settings: {error}", state.enabled)
    new = load_config()
    warnings = [fg(AMBER, line, state.enabled) for line in config_warnings()]
    changes = changed_fields(state.config, new)
    if not changes:
        return "\n".join(warnings)
    notices = apply_config(state, new)
    failed = changed_fields(new, state.config)
    applied = tuple(item for item in changes if item not in failed)
    title = dim(f"config applied ({len(applied)} of {len(changes)})",
                state.enabled)
    return "\n".join([title] + _change_lines(applied, state) + notices
                     + warnings)


def sync_config(state) -> str:
    """Apply the config file when it changed since the session read it.

    The REPL calls this before each line; ``/config`` and ``/setup``
    call it right after a save. A state built by hand (``config_stamp``
    None) is left alone. Nothing here may end the chat session: Ctrl-C
    during a slow apply (Chrome start, voice models) keeps the sections
    done so far and leaves the rest for the next line.

    Args:
        state (ReplState): the live session.

    Returns:
        str: the styled report, or "" when nothing changed.
    """
    if state.config_stamp is None:
        return ""
    previous, stamp = state.config_stamp, config_stamp()
    if stamp == previous:
        return ""
    state.config_stamp = stamp
    try:
        return _reload(state)
    except KeyboardInterrupt:
        state.config_stamp = previous  # the next line applies the rest
        return fg(AMBER, "config apply stopped — the rest applies at the "
                  "next line", state.enabled)
    except Exception as error:  # noqa: BLE001 - never unwind the REPL
        return fg(AMBER, f"config reload failed: {error}", state.enabled)


if __name__ == "__main__":
    from types import SimpleNamespace

    from threetoks.config import ResearchConfig, ThreetoksConfig
    services = SimpleNamespace(max_research_rounds=3, max_research_steps=25)
    demo = SimpleNamespace(config=ThreetoksConfig(), services=services,
                           enabled=False)
    deeper = replace(ThreetoksConfig(), research=ResearchConfig(7, 40))
    assert apply_config(demo, deeper) == []
    assert (services.max_research_rounds, services.max_research_steps) == \
        (7, 40) and demo.config == deeper
    assert sync_config(SimpleNamespace(config_stamp=None)) == ""
    print("smoke OK")
