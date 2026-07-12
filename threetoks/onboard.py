"""First-run onboarding: a step-by-step console wizard for the user config.

The wizard walks the handful of settings a fresh install cares about —
model, Ollama host, browser mode, search URL, research depth, files root,
memory — one question at a time. Enter keeps the shown default, an invalid
answer re-asks, and the result is saved as a commented INI at the
onboarding target (``$THREETOKS_CONFIG`` if set, else the per-user config
file such as ``~/.config/threetoks/config.ini``).

``ask`` and ``sink`` are injectable so tests script the console offline.
"""
import os

from threetoks.config import (BROWSER_MODES, DEFAULT_MEMORY_PATH,
                             _FALSE_WORDS, _TRUE_WORDS, BrowserConfig,
                             FilesConfig, LlmConfig, MemoryConfig,
                             ResearchConfig, SearchConfig, ThreetoksConfig,
                             load_config, save_config, user_config_dir)

_YES_WORDS = ("y",) + _TRUE_WORDS
_NO_WORDS = ("n",) + _FALSE_WORDS


def _ask_text(ask, prompt: str, default: str) -> str:
    """One free-text question; Enter keeps ``default``."""
    answer = ask(f"{prompt} [{default}]: ").strip()
    return answer or default


def _ask_choice(ask, prompt: str, choices: tuple, default: str) -> str:
    """One constrained question; re-asks until the answer is a choice."""
    menu = "/".join(choices)
    while True:
        answer = ask(f"{prompt} ({menu}) [{default}]: ").strip().lower()
        if not answer:
            return default
        if answer in choices:
            return answer


def _ask_bool(ask, prompt: str, default: bool) -> bool:
    """One yes/no question; re-asks until the answer parses as a bool word."""
    shown = "Y/n" if default else "y/N"
    while True:
        answer = ask(f"{prompt} [{shown}]: ").strip().lower()
        if not answer:
            return default
        if answer in _YES_WORDS:
            return True
        if answer in _NO_WORDS:
            return False


def _ask_int(ask, prompt: str, default: int) -> int:
    """One positive-integer question; re-asks until the answer parses."""
    while True:
        answer = ask(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        if answer.isdigit() and int(answer) > 0:
            return int(answer)


def _ask_llm(base: ThreetoksConfig, ask) -> LlmConfig:
    """Model and host questions; vote_k stays as configured."""
    model = _ask_text(ask, "Ollama model", base.llm.model)
    host = _ask_text(ask, "Ollama host", base.llm.host)
    return LlmConfig(model=model, host=host, vote_k=base.llm.vote_k)


def _ask_browser(base: ThreetoksConfig, ask) -> BrowserConfig:
    """Fetch-mode question; window visibility only matters off ``http``."""
    mode = _ask_choice(ask, "Browser fetch mode", BROWSER_MODES,
                       base.browser.mode)
    visible = base.browser.visible
    if mode != "http":
        visible = _ask_bool(ask, "Show the browser window", visible)
    return BrowserConfig(mode=mode, visible=visible)


def _ask_search(base: ThreetoksConfig, ask) -> SearchConfig:
    """SearXNG URL question; query pacing stays as configured."""
    url = _ask_text(ask, "SearXNG base URL", base.search.searxng_url)
    return SearchConfig(searxng_url=url,
                        min_interval_s=base.search.min_interval_s)


def _ask_research(base: ThreetoksConfig, ask) -> ResearchConfig:
    """Research-depth question; the per-episode step cap stays as configured."""
    rounds = _ask_int(ask, "Max research rounds per question",
                      base.research.max_rounds)
    return ResearchConfig(max_rounds=rounds, max_steps=base.research.max_steps)


def _ask_files(base: ThreetoksConfig, ask) -> FilesConfig:
    """Sandbox-root question for the read-only file-explorer agent."""
    root = _ask_text(ask, "File-agent root directory", base.files.root)
    return FilesConfig(root=root)


def _suggest_memory_path(base: ThreetoksConfig) -> str:
    """A stable per-user location instead of the cwd-relative default."""
    if base.memory.path != DEFAULT_MEMORY_PATH:
        return base.memory.path
    return os.path.join(user_config_dir(), "memory.json")


def _ask_memory(base: ThreetoksConfig, ask) -> MemoryConfig:
    """Memory on/off question; the file location is only asked when on."""
    enabled = _ask_bool(ask, "Enable cross-session memory",
                        base.memory.enabled)
    path = _suggest_memory_path(base)
    if enabled:
        path = _ask_text(ask, "Memory file", path)
    return MemoryConfig(path=path, enabled=enabled)


def run_wizard(base: ThreetoksConfig = None, ask=input,
               sink=print) -> ThreetoksConfig:
    """Ask the setup questions one at a time and return the answers.

    ``base`` supplies the shown defaults (built-ins when ``None``); Enter
    keeps a default and invalid answers re-ask. Settings the wizard does
    not ask about (vote_k, pacing, step cap) carry over from ``base``.
    """
    base = base or ThreetoksConfig()
    sink("ThreeToks setup — Enter keeps the [default].")
    return ThreetoksConfig(
        llm=_ask_llm(base, ask),
        browser=_ask_browser(base, ask),
        search=_ask_search(base, ask),
        research=_ask_research(base, ask),
        files=_ask_files(base, ask),
        memory=_ask_memory(base, ask))


def run_onboarding(ask=input, sink=print) -> str:
    """Run the wizard seeded from any existing config, save, report the path.

    Returns the path written: ``$THREETOKS_CONFIG`` when set, otherwise the
    per-user config file.
    """
    config = run_wizard(load_config(), ask, sink)
    path = save_config(config)
    sink(f"config saved — {path}")
    return path


if __name__ == "__main__":
    answers = iter(["", "", "stealth", "y", "", "5", "", "n"])
    quiet = lambda line: None
    wizard = run_wizard(ask=lambda prompt: next(answers), sink=quiet)
    assert wizard.browser.mode == "stealth", wizard.browser
    assert wizard.browser.visible is True, wizard.browser
    assert wizard.research.max_rounds == 5, wizard.research
    assert wizard.memory.enabled is False, wizard.memory
    assert wizard.llm.model == ThreetoksConfig().llm.model, wizard.llm
    assert "threetoks" in wizard.memory.path, wizard.memory
    retries = iter(["", "", "teleport", "plain", "no", "", "zero", "3", "",
                    "off"])
    retried = run_wizard(ask=lambda prompt: next(retries), sink=quiet)
    assert retried.browser.mode == "plain", retried.browser
    assert retried.research.max_rounds == 3, retried.research
    print("smoke OK")
