"""Central configuration for ThreeToks (stdlib ``configparser`` only).

One editable ``config.ini`` holds every runtime knob so the CLI and TUI stop
hard-coding defaults. Resolution order for the file:

    1. ``$THREETOKS_CONFIG`` if set (explicit override),
    2. ``./config.ini`` in the current working directory (dev convenience),
    3. the per-user config file (``~/.config/threetoks/config.ini`` on
       Linux/macOS honouring ``$XDG_CONFIG_HOME``, ``%APPDATA%\\threetoks``
       on Windows) — written by the first-run onboarding wizard,
    4. no file — every field falls back to the built-in default.

Parsing is defensive: a missing file, missing section, missing key, or an
unparseable value never raises; each field independently falls back to its
default. This keeps a hand-edited config from ever crashing the agent.
"""
import configparser
import os
from dataclasses import dataclass

CONFIG_ENV_VAR = "THREETOKS_CONFIG"
DEFAULT_CONFIG_NAME = "config.ini"

# --- built-in defaults (mirrored, with comments, in config.ini) ----------
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_LLM_HOST = "http://localhost:11434"
DEFAULT_VOTE_K = 1
DEFAULT_BROWSER_MODE = "http"
DEFAULT_BROWSER_VISIBLE = False
DEFAULT_SEARXNG_URL = "http://localhost:8080"
DEFAULT_MIN_INTERVAL_S = 5.0
DEFAULT_MAX_ROUNDS = 3
DEFAULT_MAX_STEPS = 25
DEFAULT_FILES_ROOT = "."
DEFAULT_MEMORY_PATH = "threetoks_memory.json"
DEFAULT_MEMORY_ENABLED = True
DEFAULT_CODE_RETRIEVAL = False
DEFAULT_SNIPPET_CACHE = "threetoks_snippets.json"

BROWSER_MODES = ("http", "plain", "stealth")
_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")


@dataclass(frozen=True)
class LlmConfig:
    """Language-model settings: which model, where it lives, vote count."""

    model: str = DEFAULT_MODEL
    host: str = DEFAULT_LLM_HOST
    vote_k: int = DEFAULT_VOTE_K


@dataclass(frozen=True)
class BrowserConfig:
    """Fetch backend: ``http`` (requests), ``plain`` / ``stealth`` (Chrome)."""

    mode: str = DEFAULT_BROWSER_MODE
    visible: bool = DEFAULT_BROWSER_VISIBLE


@dataclass(frozen=True)
class SearchConfig:
    """Search backend URL and the minimum spacing between outbound queries."""

    searxng_url: str = DEFAULT_SEARXNG_URL
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S


@dataclass(frozen=True)
class ResearchConfig:
    """Episode budgets: judged research rounds and per-episode step cap."""

    max_rounds: int = DEFAULT_MAX_ROUNDS
    max_steps: int = DEFAULT_MAX_STEPS


@dataclass(frozen=True)
class FilesConfig:
    """Root directory the read-only file-explorer agent is sandboxed to."""

    root: str = DEFAULT_FILES_ROOT


@dataclass(frozen=True)
class MemoryConfig:
    """Where cross-session memory is stored, and whether it is used at all."""

    path: str = DEFAULT_MEMORY_PATH
    enabled: bool = DEFAULT_MEMORY_ENABLED


@dataclass(frozen=True)
class CodeConfig:
    """Coding-agent knobs: retrieval-as-repair (experimental, off by
    default — it executes and embeds code fetched from the internet)."""

    retrieval: bool = DEFAULT_CODE_RETRIEVAL
    snippet_cache: str = DEFAULT_SNIPPET_CACHE


@dataclass(frozen=True)
class ThreetoksConfig:
    """The whole configuration: one immutable section object per ``[section]``."""

    llm: LlmConfig = LlmConfig()
    browser: BrowserConfig = BrowserConfig()
    search: SearchConfig = SearchConfig()
    research: ResearchConfig = ResearchConfig()
    files: FilesConfig = FilesConfig()
    memory: MemoryConfig = MemoryConfig()
    code: CodeConfig = CodeConfig()


def _get_str(parser: configparser.ConfigParser, section: str,
             key: str, default: str) -> str:
    """Read a string, falling back to ``default`` when absent or blank."""
    value = parser.get(section, key, fallback=default)
    return value.strip() or default


def _get_int(parser: configparser.ConfigParser, section: str,
             key: str, default: int) -> int:
    """Read an int, falling back to ``default`` on absence or bad value."""
    try:
        return parser.getint(section, key, fallback=default)
    except ValueError:
        return default


def _get_float(parser: configparser.ConfigParser, section: str,
               key: str, default: float) -> float:
    """Read a float, falling back to ``default`` on absence or bad value."""
    try:
        return parser.getfloat(section, key, fallback=default)
    except ValueError:
        return default


def _get_bool(parser: configparser.ConfigParser, section: str,
              key: str, default: bool) -> bool:
    """Read a boolean by word list, tolerant of any casing or stray text."""
    raw = parser.get(section, key, fallback=None)
    if raw is None:
        return default
    word = raw.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return default


def _get_choice(parser: configparser.ConfigParser, section: str, key: str,
                default: str, choices: tuple) -> str:
    """Read a string constrained to ``choices``; bad values fall back."""
    value = _get_str(parser, section, key, default).lower()
    return value if value in choices else default


def _build_config(parser: configparser.ConfigParser) -> ThreetoksConfig:
    """Assemble a ``ThreetoksConfig`` from a (possibly empty) parser."""
    return ThreetoksConfig(
        llm=LlmConfig(
            model=_get_str(parser, "llm", "model", DEFAULT_MODEL),
            host=_get_str(parser, "llm", "host", DEFAULT_LLM_HOST),
            vote_k=_get_int(parser, "llm", "vote_k", DEFAULT_VOTE_K)),
        browser=BrowserConfig(
            mode=_get_choice(parser, "browser", "mode",
                             DEFAULT_BROWSER_MODE, BROWSER_MODES),
            visible=_get_bool(parser, "browser", "visible",
                              DEFAULT_BROWSER_VISIBLE)),
        search=SearchConfig(
            searxng_url=_get_str(parser, "search", "searxng_url",
                                 DEFAULT_SEARXNG_URL),
            min_interval_s=_get_float(parser, "search", "min_interval_s",
                                      DEFAULT_MIN_INTERVAL_S)),
        research=ResearchConfig(
            max_rounds=_get_int(parser, "research", "max_rounds",
                                DEFAULT_MAX_ROUNDS),
            max_steps=_get_int(parser, "research", "max_steps",
                               DEFAULT_MAX_STEPS)),
        files=FilesConfig(
            root=_get_str(parser, "files", "root", DEFAULT_FILES_ROOT)),
        memory=MemoryConfig(
            path=_get_str(parser, "memory", "path", DEFAULT_MEMORY_PATH),
            enabled=_get_bool(parser, "memory", "enabled",
                              DEFAULT_MEMORY_ENABLED)),
        code=CodeConfig(
            retrieval=_get_bool(parser, "code", "retrieval",
                                DEFAULT_CODE_RETRIEVAL),
            snippet_cache=_get_str(parser, "code", "snippet_cache",
                                   DEFAULT_SNIPPET_CACHE)))


def user_config_dir(platform: str = None) -> str:
    """Per-user threetoks config directory for this OS (never created here).

    Windows (``platform == "nt"``) uses ``%APPDATA%``; everything else
    follows XDG: ``$XDG_CONFIG_HOME``, defaulting to ``~/.config``.
    """
    system = platform or os.name
    if system == "nt":
        base = os.environ.get("APPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
            os.path.expanduser("~"), ".config")
    return os.path.join(base, "threetoks")


def user_config_path(platform: str = None) -> str:
    """Full path of the per-user config file."""
    return os.path.join(user_config_dir(platform), DEFAULT_CONFIG_NAME)


def find_config_path() -> str:
    """The config file that would actually be read, or ``None`` if none exists.

    Mirrors the load order ($THREETOKS_CONFIG, ./config.ini, user file); the
    TUI uses ``None`` to decide that first-run onboarding should run. A set
    but missing ``$THREETOKS_CONFIG`` returns ``None`` without falling
    through — the wizard then saves to that explicit path.
    """
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return env_path if os.path.isfile(env_path) else None
    for candidate in (DEFAULT_CONFIG_NAME, user_config_path()):
        if os.path.isfile(candidate):
            return candidate
    return None


def onboarding_target() -> str:
    """Where a wizard-produced config is saved: ``$THREETOKS_CONFIG`` or user file."""
    return os.environ.get(CONFIG_ENV_VAR) or user_config_path()


def _resolve_path(path: str = None) -> str:
    """Pick the config path: explicit arg, else the found file, else where
    one would be written (a missing file reads as pure defaults)."""
    if path:
        return path
    return find_config_path() or onboarding_target()


_INI_TEMPLATE = """\
# ThreeToks configuration — written by the onboarding wizard, safe to edit.
# Load order: $THREETOKS_CONFIG > ./config.ini > this file > built-in defaults.
# Bad or missing values silently fall back to the built-in defaults.

[llm]
# Ollama model used for every decision node, and the server that hosts it.
model = {model}
host = {host}
vote_k = {vote_k}

[browser]
# Page fetching: http (fast, bot-detectable) | plain | stealth (real Chrome).
mode = {mode}
visible = {visible}

[search]
# Local SearXNG base URL; preferred over the Bing/DDG HTML fallbacks.
searxng_url = {searxng_url}
min_interval_s = {min_interval_s}

[research]
# Judged research rounds per question / decision steps per episode.
max_rounds = {max_rounds}
max_steps = {max_steps}

[files]
# Root directory the read-only file-explorer agent is sandboxed to.
root = {root}

[memory]
# Cross-session (query, answer) memory store.
path = {memory_path}
enabled = {enabled}

[code]
# EXPERIMENTAL: when a planned function with trusted anchor examples
# exhausts generation attempts, fetch a classic implementation from the
# public web instead of stubbing it. Executes internet code in a
# subprocess and embeds it (provenance-stamped, license unreviewed).
retrieval = {retrieval}
snippet_cache = {snippet_cache}
"""


def render_ini(config: ThreetoksConfig) -> str:
    """Render ``config`` as a commented INI string that parses back equal."""
    return _INI_TEMPLATE.format(
        model=config.llm.model, host=config.llm.host, vote_k=config.llm.vote_k,
        mode=config.browser.mode, visible=str(config.browser.visible).lower(),
        searxng_url=config.search.searxng_url,
        min_interval_s=config.search.min_interval_s,
        max_rounds=config.research.max_rounds,
        max_steps=config.research.max_steps,
        root=config.files.root, memory_path=config.memory.path,
        enabled=str(config.memory.enabled).lower(),
        retrieval=str(config.code.retrieval).lower(),
        snippet_cache=config.code.snippet_cache)


def save_config(config: ThreetoksConfig, path: str = None) -> str:
    """Write ``config`` as commented INI to ``path`` and return that path.

    ``path`` defaults to the onboarding target ($THREETOKS_CONFIG if set,
    else the per-user config file). Parent directories are created.
    """
    target = path or onboarding_target()
    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(render_ini(config))
    return target


def load_config(path: str = None) -> ThreetoksConfig:
    """Load configuration from ``path`` / env / cwd, else pure defaults.

    A missing file is not an error: it yields the built-in defaults. An
    unreadable or malformed file also degrades to defaults rather than
    raising, so a broken edit never takes the agent down. Interpolation is
    off: a literal ``%`` in a value (URL escapes, ``%APPDATA%`` paths) is
    data, not a template — with it on, ``.get()`` would raise at build
    time, outside this guard.
    """
    parser = configparser.ConfigParser(interpolation=None)
    resolved = _resolve_path(path)
    try:
        parser.read(resolved, encoding="utf-8")
    except (OSError, configparser.Error):
        parser = configparser.ConfigParser(interpolation=None)
    return _build_config(parser)


if __name__ == "__main__":
    defaults = load_config("/nonexistent/threetoks.ini")
    assert defaults.llm.model == DEFAULT_MODEL, defaults.llm
    assert defaults.browser.mode == "http", defaults.browser
    assert defaults.research.max_rounds == DEFAULT_MAX_ROUNDS
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False) as handle:
        handle.write("[browser]\nmode = stealth\nvisible = yes\n"
                     "[llm]\nvote_k = not-a-number\n")
        temp_path = handle.name
    parsed = load_config(temp_path)
    assert parsed.browser.mode == "stealth", parsed.browser
    assert parsed.browser.visible is True, parsed.browser
    assert parsed.llm.vote_k == DEFAULT_VOTE_K, parsed.llm  # bad -> default
    assert defaults.memory.path == DEFAULT_MEMORY_PATH, defaults.memory
    assert defaults.memory.enabled is True, defaults.memory
    assert defaults.code.retrieval is False, defaults.code   # off by default
    assert defaults.code.snippet_cache == DEFAULT_SNIPPET_CACHE
    os.unlink(temp_path)
    assert user_config_path("nt").endswith(
        os.path.join("threetoks", "config.ini")), user_config_path("nt")
    assert user_config_path("posix").endswith(
        os.path.join("threetoks", "config.ini")), user_config_path("posix")
    with tempfile.TemporaryDirectory() as tmp:
        saved = save_config(parsed, os.path.join(tmp, "nested", "config.ini"))
        roundtrip = load_config(saved)
        assert roundtrip == parsed, roundtrip
    print("smoke OK")
