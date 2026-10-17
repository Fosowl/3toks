"""Health hints: a short verdict shown beside a setting.

Each check reads the current config and the probe answers and returns one
:class:`Hint`, or None when there is nothing to say. The config screen
and ``threetoks doctor`` share these checks, so they never disagree.
"""
import os
from dataclasses import dataclass

from threetoks.config import ThreetoksConfig
from threetoks.fields import Field
from threetoks.onboard import piper_voice_problem, vosk_folder_problem
from threetoks.probes import Probes

LEVEL_OK = "ok"
LEVEL_WARN = "warn"
LEVEL_BAD = "bad"
_WARNING_PREFIX = "warning: "


@dataclass(frozen=True)
class Hint:
    """One verdict: a level (ok, warn, bad) and a short text."""

    level: str
    text: str


def _provider(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Say whether the chat provider's API key is in the environment."""
    if probes.key is None:
        return None
    name, found = probes.key
    if found:
        return Hint(LEVEL_OK, f"{name} found")
    return Hint(LEVEL_WARN, f"export {name}")


def _model_listed(model: str, models: tuple) -> bool:
    """True when ``model`` is pulled; a bare name means ``:latest``."""
    return model in models or f"{model}:latest" in models


def _model(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Say whether the Ollama model is pulled and its template is known."""
    if config.llm.provider != "ollama":
        return None
    if not probes.ollama_up:
        return Hint(LEVEL_WARN, "ollama offline")
    if not _model_listed(config.llm.model, probes.models or ()):
        return Hint(LEVEL_BAD, f"ollama pull {config.llm.model}")
    from threetoks.backend.base import detect_family
    if not detect_family(config.llm.model)[1]:
        return Hint(LEVEL_WARN, "unknown template family")
    return Hint(LEVEL_OK, "pulled")


def _host(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Say whether Ollama answers at the configured host."""
    if config.llm.provider != "ollama":
        return None
    if probes.ollama_up:
        return Hint(LEVEL_OK, "ollama answers")
    return Hint(LEVEL_BAD, "no answer")


def _api_base(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Warn that the custom provider needs an API base."""
    if config.llm.provider == "custom" and not config.llm.api_base:
        return Hint(LEVEL_BAD, "required for custom")
    return None


def _browser_mode(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Warn that plain mode refuses to start without a search engine."""
    if config.browser.mode == "plain" and not probes.searxng_up:
        return Hint(LEVEL_WARN, "plain needs SearXNG")
    return None


def _searxng(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Say whether SearXNG answers at its URL."""
    if probes.searxng_up:
        return Hint(LEVEL_OK, "answers")
    return Hint(LEVEL_WARN, "no answer")


def _files_root(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Say whether the file root is a real directory."""
    if os.path.isdir(config.files.root):
        return Hint(LEVEL_OK, "directory found")
    return Hint(LEVEL_BAD, "not a directory")


def _voice_path(problem_of, path: str) -> Hint | None:
    """Turn an onboard path check into a hint; an empty path says nothing."""
    if not path:
        return None
    problem = problem_of(path)
    if problem is None:
        return Hint(LEVEL_OK, "found")
    return Hint(LEVEL_BAD, problem.removeprefix(_WARNING_PREFIX))


def _stt_path(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Check the vosk model folder."""
    return _voice_path(vosk_folder_problem, config.voice.stt_model_path)


def _tts_path(config: ThreetoksConfig, probes: Probes) -> Hint | None:
    """Check the piper voice file and its sidecar."""
    return _voice_path(piper_voice_problem, config.voice.tts_model_path)


_CHECKS = {
    "llm.provider": _provider, "llm.model": _model, "llm.host": _host,
    "llm.api_base": _api_base, "browser.mode": _browser_mode,
    "search.searxng_url": _searxng, "files.root": _files_root,
    "voice.stt_model_path": _stt_path, "voice.tts_model_path": _tts_path,
}


def hint_for(field: Field, config: ThreetoksConfig,
             probes: Probes) -> Hint | None:
    """Return the health hint for one setting, or None.

    Args: field (Field): the setting; config (ThreetoksConfig): the
    current config, including unsaved edits; probes (Probes): one sweep.
    Returns: a Hint, or None when nothing is worth saying.
    """
    check = _CHECKS.get(field.dotted)
    return check(config, probes) if check else None


if __name__ == "__main__":
    from threetoks.fields import find_field
    cfg = ThreetoksConfig()
    down = Probes()
    assert hint_for(find_field("llm.model"), cfg, down).level == LEVEL_WARN
    up = Probes(ollama_up=True, models=("qwen3.5:2b",))
    assert hint_for(find_field("llm.model"), cfg, up).level == LEVEL_OK
    assert hint_for(find_field("llm.vote_k"), cfg, up) is None
    print("smoke OK")
