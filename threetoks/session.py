"""The chat session's state, and how a config section becomes live parts.

The TUI builds one :class:`ReplState` at start with :func:`build_state`.
The ``make_*`` helpers turn one config section into the live object it
controls (search provider, browser fetcher, policy factory, relay,
camera), and ``switch_fetcher``, ``swap_model``, ``start_voice`` and
``stop_voice`` change a running session. The slash commands and
:mod:`threetoks.live_config` call the same functions, so a start, a
command and a saved setting change things the same way. The switch
functions raise on failure; their callers turn errors into messages.
"""
from dataclasses import dataclass, field
from pathlib import Path

from threetoks.config import ThreetoksConfig
from threetoks.theme import AMBER, color_enabled, dim, fg


@dataclass
class ReplState:
    """Mutable REPL state passed to command handlers.

    ``policy_factory`` rebuilds the decision policy from a model name so
    ``/model`` can swap models without importing a backend at test time.
    ``config`` is the ThreetoksConfig the session runs on now; ``model``
    may differ from ``config.llm.model`` after ``-m`` or ``/model``.
    ``config_stamp`` names the file version ``config`` came from; it
    stays None in states built by hand, which turns live reload off.
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
    voice: object = None    # live VoiceSession, or None for keyboard input
    debug: bool = False     # /debug: show each decision's raw model answer
    # (NOT voice.debug, which traces microphone transcripts)
    debug_full: bool = False  # /debug full: dump whole prompts per call
    config: ThreetoksConfig = field(default_factory=ThreetoksConfig)
    config_stamp: tuple | None = None  # see config.config_stamp


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


def refresh_optional_agents(state: ReplState, model: str) -> None:
    """Re-derive the capability agents for ``model`` (look needs vision).

    ``/model`` and a live camera or relay change call this. The planner
    is rebuilt last, so its plans only use agents that exist.

    Args:
        state (ReplState): the live session; ``state.specs`` is replaced.
        model (str): the model the session now runs.
    """
    from threetoks.agents import (
        OPTIONAL_AGENT_NAMES,
        PLANNER_NAME,
        optional_agents,
        with_planner,
    )
    derived = OPTIONAL_AGENT_NAMES + (PLANNER_NAME,)
    kept = [spec for spec in state.specs if spec.name not in derived]
    state.specs = with_planner(kept + optional_agents(state.services, model))


def make_policy_factory(llm, state: ReplState = None):
    """Build the ``name -> Policy`` factory for one ``[llm]`` section.

    The start and a live ``[llm]`` change call this. The factory reads
    ``state.debug`` on each call, so a model swap keeps ``/debug``.

    Args:
        llm (LlmConfig): provider, host, api_base, vote_k, send_seed and
            disable_think, from the config.
        state (ReplState or None): the session whose debug flag to use.

    Returns:
        callable: model name (str) -> Policy. It raises RuntimeError when
        the provider cannot start (for example a missing API key).
    """
    from threetoks.backend.providers import make_backend
    from threetoks.cli import make_policy_config
    from threetoks.policy import Policy
    from threetoks.trace import Tracer

    def build(name: str):
        debug = state is not None and state.debug
        return Policy(make_backend(llm), make_policy_config(name, llm),
                      Tracer(None), debug=debug)
    return build


def make_provider(config, sink):
    """Search provider, or None (with a notice) on a bare core install.

    The web extras (curl_cffi, bs4, markdownify) are optional; a missing
    import must degrade to a working TUI without the web agent, never a
    startup crash. The start and a live ``[search]`` change call this.

    Args:
        config (ThreetoksConfig): reads ``search.searxng_url`` and
            ``search.min_interval_s``.
        sink (callable): takes one notice line (str).

    Returns:
        SearchProvider or None.
    """
    try:
        import threetoks.web.search as search_mod  # lazy: web extras
    except ImportError:
        sink(dim("  web extras missing — web agent disabled "
                 "(install: uv tool install '3toks[web]')",
                 color_enabled()))
        return None
    search_mod.MIN_SEARCH_INTERVAL_S = config.search.min_interval_s
    return search_mod.auto_provider(config.search.searxng_url)


def _searxng_gate(config, sink) -> None:
    """Browser modes pair with the local SearXNG: exit clearly without it.

    ``plain`` exists for JS-heavy pages found through search;
    starting Chrome when no search engine answers can only produce a
    session that opens nothing, so refuse early and say how to fix it.
    """
    try:
        from threetoks.web import search as search_mod  # lazy: web extras
    except ImportError:
        return  # no web extras: the provider is already disabled anyway
    base_url = search_mod.searxng_base(config.search.searxng_url)
    if not search_mod._searxng_answers(base_url):
        sink(fg(AMBER, f"  browser mode '{config.browser.mode}' needs a "
                f"local SearXNG, but none answers at {base_url} — run "
                "./start_search_engine.sh first", color_enabled()))
        raise SystemExit(1)


def make_browser_fetcher(visible: bool):
    """Lazily build the Chrome fetcher for ``plain`` (imports deferred)."""
    from threetoks.web.browser import make_fetcher
    return make_fetcher(visible=visible)


def initial_fetcher(config, services, sink=print):
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
    _searxng_gate(config, sink)
    try:
        fetcher = make_browser_fetcher(config.browser.visible)
    except Exception as error:  # noqa: BLE001 - optional module; use HTTP
        services.fetch_page = fetch_page
        sink(fg(AMBER, f"  browser {mode} unavailable: {error} — "
                "falling back to http fetch", color_enabled()))
        return None
    services.fetch_page = fetcher.fetch_page
    return fetcher


def make_retriever(config, provider):
    """The opt-in code retrieval hook, or None (the default).

    Requires both the ``[code] retrieval`` config flag AND a working
    search provider (a bare install has neither and gets None).
    """
    if provider is None or not config.code.retrieval:
        return None
    from threetoks.code.retrieve import Retriever
    return Retriever(provider, cache_path=config.code.snippet_cache)


def warn_bad_config(sink) -> None:
    """Say so when the config file, or a value in it, was ignored.

    A file that fails to parse degrades to the built-in defaults rather
    than raising, and a bad value degrades to its own default; both are
    otherwise invisible: the session simply runs on a config the user did
    not write (see fields.config_warnings).
    """
    from threetoks.fields import config_warnings
    for line in config_warnings():
        sink(fg(AMBER, line, color_enabled()))


def make_relay(config, sink):
    """A live Relay on a Raspberry Pi with the extra installed, else None."""
    from threetoks.relay import relay_available
    if not relay_available():
        return None
    from threetoks.relay import Relay
    try:
        return Relay(pin=config.relay.pin)
    except Exception as error:  # noqa: BLE001 - hardware; degrade, don't die
        sink(fg(AMBER, f"  relay unavailable: {error}", color_enabled()))
        return None


def make_capture(config):
    """A camera-frame callable when opencv is installed, else None."""
    from threetoks.camera import camera_available, capture_jpeg_b64
    if not camera_available():
        return None
    index = config.camera.index
    return lambda: capture_jpeg_b64(index)


def load_memory(config):
    """The MemoryStore for ``[memory]``, or None when memory is off.

    Args:
        config (ThreetoksConfig): reads ``memory.enabled`` and
            ``memory.path``.

    Returns:
        MemoryStore or None.
    """
    from threetoks.memory import MemoryStore
    return MemoryStore.load(config.memory.path) if config.memory.enabled \
        else None


def close_fetcher(state: ReplState) -> None:
    """Quit the browser fetcher of an earlier switch, if there is one.

    A fetcher switch and the end of the chat call this.

    Args:
        state (ReplState): the live session; ``state.fetcher`` is cleared.
    """
    if state.fetcher is None:
        return
    try:
        state.fetcher.close()
    except Exception:  # noqa: BLE001 - teardown must never crash the REPL
        pass
    state.fetcher = None


def switch_fetcher(state: ReplState, mode: str, visible: bool) -> None:
    """Point ``services.fetch_page`` at plain HTTP or a new Chrome.

    ``/browser`` and a saved ``[browser]`` change call this. A new
    fetcher gets a fresh chance at every site, so hosts blocked earlier
    in the session are forgotten.

    Args:
        state (ReplState): the live session.
        mode (str): ``http`` or ``plain``.
        visible (bool): show the Chrome window (``plain`` only).

    Raises:
        Exception: the browser start failed; HTTP fetch is bound then.
    """
    from threetoks.cli import fetch_page
    close_fetcher(state)
    state.services.blocked_hosts.clear()
    state.services.fetch_page = fetch_page  # never leave a dead driver bound
    if mode == "http":
        return
    state.fetcher = make_browser_fetcher(visible)
    state.services.fetch_page = state.fetcher.fetch_page


def swap_model(state: ReplState, name: str, llm=None) -> str | None:
    """Build the policy for ``name`` first, then switch the session to it.

    ``/model`` calls this with the session's factory; a saved ``[llm]``
    change calls it with the new section. Optional agents follow the
    model (``look`` needs vision).

    Args:
        state (ReplState): the live session.
        name (str): the model tag.
        llm (LlmConfig or None): a new ``[llm]`` section to build from;
            None keeps ``state.policy_factory``.

    Returns:
        str or None: the unknown-template warning, if any.

    Raises:
        RuntimeError: the provider cannot start; nothing was changed.
    """
    factory = state.policy_factory if llm is None \
        else make_policy_factory(llm, state)
    policy = factory(name)
    state.policy_factory, state.policy, state.model = factory, policy, name
    refresh_optional_agents(state, name)
    provider = (llm or state.config.llm).provider
    return unknown_family_notice(name) if provider == "ollama" else None


def start_voice(state: ReplState, voice) -> list:
    """Start listening and speaking with one ``[voice]`` section.

    ``/voice on``, the start and a saved ``[voice]`` change call this.

    Args:
        state (ReplState): the live session; ``state.voice`` is set.
        voice (VoiceConfig): language, model paths, speaker, delay.

    Returns:
        list of str: notes about a side (stt or tts) that is missing.

    Raises:
        Exception: no voice side can start.
    """
    from threetoks.voice.session import make_voice_session
    state.voice, notes = make_voice_session(voice)
    return notes


def stop_voice(state: ReplState) -> None:
    """Stop any live voice session and return to keyboard input."""
    if state.voice is not None:
        state.voice.close()
        state.voice = None


def _make_services(config, sink):
    """Build the Services every agent shares, from the whole config."""
    from threetoks.services import Services
    provider = make_provider(config, sink)
    return Services(provider=provider,
                    files_root=Path(config.files.root),
                    max_research_rounds=config.research.max_rounds,
                    max_research_steps=config.research.max_steps,
                    retriever=make_retriever(config, provider),
                    relay=make_relay(config, sink),
                    capture_frame=make_capture(config))


def _agent_specs(services, model: str) -> list:
    """Every agent the session can route to, planner last."""
    from threetoks.agents import default_agents, optional_agents, with_planner
    specs = default_agents(services) + optional_agents(services, model)
    if services.provider is None:  # bare install: never route to the web agent
        specs = [spec for spec in specs if spec.name != "web"]
    return with_planner(specs)


def build_state(model: str = None, sink=print) -> ReplState:
    """Assemble live Services/specs/policy from the resolved config file.

    ``tui.main`` calls this once at start. Imports of the backend and the
    agent registry stay lazy so the module loads without them.

    Args:
        model (str or None): the ``-m`` flag; it overrides ``[llm] model``.
        sink (callable): takes one startup notice line (str).

    Returns:
        ReplState: the ready session, with ``config`` and ``config_stamp``
        set so later saves apply live.
    """
    from threetoks.config import config_stamp, load_config
    stamp, config = config_stamp(), load_config()
    warn_bad_config(sink)
    services = _make_services(config, sink)
    fetcher = initial_fetcher(config, services, sink)
    chosen = model or config.llm.model
    state = ReplState(services, _agent_specs(services, chosen), None, chosen,
                      None, enabled=color_enabled(), fetcher=fetcher,
                      memory=load_memory(config), config=config,
                      config_stamp=stamp)
    state.policy_factory = make_policy_factory(config.llm, state)
    _family_notice(config, chosen, sink)
    state.policy = state.policy_factory(chosen)
    return state


def _family_notice(config, model: str, sink) -> None:
    """Warn at start when a raw-mode model's template is a guess."""
    notice = unknown_family_notice(model) \
        if config.llm.provider == "ollama" else None  # raw mode only
    if notice:
        sink(notice)


if __name__ == "__main__":
    assert unknown_family_notice("qwen3.5:2b") is None
    assert "granite" in unknown_family_notice("granite4:tiny")
    built = make_policy_factory(ThreetoksConfig().llm)("qwen3.5:2b")
    assert built.config.vote_k == 1 and not built.debug, built.config
    assert make_retriever(ThreetoksConfig(), object()) is None
    print("smoke OK")
