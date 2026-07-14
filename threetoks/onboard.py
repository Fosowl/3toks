"""First-run onboarding: a step-by-step console wizard for the user config.

The wizard walks the handful of settings a fresh install cares about —
model, Ollama host, browser mode, search URL, research depth, files root,
memory, and an optional voice setup — one question at a time. Enter keeps
the shown default, an invalid answer re-asks, and the result is saved as a
commented INI at the onboarding target (``$THREETOKS_CONFIG`` if set, else
the per-user config file such as ``~/.config/threetoks/config.ini``).

``ask``, ``sink``, and ``open_url`` are injectable so tests script the
console offline and never launch a real browser.
"""
import importlib.util
import os
import webbrowser
from dataclasses import replace

from threetoks.config import (BROWSER_MODES, DEFAULT_MEMORY_PATH,
                             _FALSE_WORDS, _TRUE_WORDS, BrowserConfig,
                             FilesConfig, LlmConfig, MemoryConfig,
                             ResearchConfig, SearchConfig, ThreetoksConfig,
                             VoiceConfig, load_config, save_config,
                             user_config_dir)

_YES_WORDS = ("y",) + _TRUE_WORDS
_NO_WORDS = ("n",) + _FALSE_WORDS

# Model-download pages shown during the optional voice step.
VOSK_MODELS_URL = "https://alphacephei.com/vosk/models"
PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices"
_VOICE_EXTRA_HINT = "install with: uv pip install '3toks[voice]'"
_VOICE_PACKAGES = ("vosk", "sounddevice", "piper")
_PIPER_SIDECAR_SUFFIX = ".json"  # piper needs voice.onnx.json beside .onnx


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
    """Model and host questions; provider, api_base, vote_k stay as configured."""
    model = _ask_text(ask, "Ollama model", base.llm.model)
    host = _ask_text(ask, "Ollama host", base.llm.host)
    return LlmConfig(model=model, host=host, provider=base.llm.provider,
                     api_base=base.llm.api_base, vote_k=base.llm.vote_k)


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


def _open_url_default(url: str) -> bool:
    """Open ``url`` in the default browser; never raise.

    Inputs: a URL string. Output: True when handed to a browser, False
    when opening fails or the environment is headless.
    """
    try:
        return webbrowser.open(url)
    except Exception:  # a browser hiccup must not crash setup
        return False


def _show_model_pages(sink, open_url) -> None:
    """Open the vosk and piper model pages and always print both links.

    Inputs: ``sink`` for output and ``open_url`` to try a browser open.
    Output: none; the printed links are the headless/failure fallback,
    so both URLs are shown whether or not the browser opened.
    """
    open_url(VOSK_MODELS_URL)
    sink(f"vosk STT models: {VOSK_MODELS_URL}")
    open_url(PIPER_VOICES_URL)
    sink(f"piper TTS voices: {PIPER_VOICES_URL}")


def _verify_tts_path(tts_path: str, sink) -> None:
    """Warn about a missing piper voice, its sidecar, or an empty path.

    Inputs: the configured ``.onnx`` path and a ``sink``. Output: none.
    An empty path notes that voice output stays off; a set path is
    checked for the file and its required ``.onnx.json`` sidecar.
    """
    if not tts_path:
        sink("note: voice output stays off until a piper .onnx is set")
        return
    if not os.path.isfile(tts_path):
        sink(f"warning: piper voice file not found: {tts_path}")
        return
    sidecar = tts_path + _PIPER_SIDECAR_SUFFIX
    if not os.path.isfile(sidecar):
        sink(f"warning: piper sidecar missing: {sidecar}")


def _verify_packages(sink) -> None:
    """Hint the install command when a voice package is not importable.

    Inputs: a ``sink``. Output: none. Uses ``find_spec`` (no import side
    effects) for vosk/sounddevice/piper; one hint covers any missing.
    """
    missing = [name for name in _VOICE_PACKAGES
               if importlib.util.find_spec(name) is None]
    if missing:
        sink(_VOICE_EXTRA_HINT)


def _verify_voice(voice: VoiceConfig, sink) -> None:
    """Print one actionable warning per voice-setup problem, never blocking.

    Inputs: the assembled ``VoiceConfig`` and a ``sink``. Output: none.
    Sanity-checks the vosk folder, the piper voice and its sidecar, and
    the importability of the voice packages — warnings only, no re-asks.
    """
    stt_path = voice.stt_model_path
    if stt_path and not os.path.exists(stt_path):
        sink(f"warning: vosk model folder not found: {stt_path}")
    _verify_tts_path(voice.tts_model_path, sink)
    _verify_packages(sink)


def _ask_voice(base: ThreetoksConfig, ask, sink, open_url) -> VoiceConfig:
    """Optional last step: enable voice, pick language, set model paths.

    ``base`` gives defaults; declining returns ``base.voice`` with
    ``enabled=False``. On yes: asks language and both model paths, opens
    the model pages via ``open_url``, verifies via :func:`_verify_voice`,
    and returns the enabled ``VoiceConfig`` (``tts_speaker`` and
    ``post_speak_delay`` carried from base).
    """
    if not _ask_bool(ask, "Enable voice mode (microphone in, spoken "
                     "answers out)", base.voice.enabled):
        return replace(base.voice, enabled=False)
    lang = _ask_text(ask, "Voice input language", base.voice.lang)
    _show_model_pages(sink, open_url)
    stt = _ask_text(ask, f"Vosk model folder (Enter = auto-download for "
                    f"'{lang}')", base.voice.stt_model_path)
    tts = _ask_text(ask, "Piper voice .onnx file (Enter = spoken answers "
                    "stay off)", base.voice.tts_model_path)
    voice = replace(base.voice, enabled=True, lang=lang,
                    stt_model_path=stt, tts_model_path=tts)
    _verify_voice(voice, sink)
    return voice


def run_wizard(base: ThreetoksConfig = None, ask=input, sink=print,
               open_url=_open_url_default) -> ThreetoksConfig:
    """Ask the setup questions one at a time and return the answers.

    ``base`` supplies the shown defaults (built-ins when ``None``); Enter
    keeps a default and invalid answers re-ask. ``open_url`` launches the
    voice-model pages (injected in tests). Everything the wizard does not
    ask about — provider/api_base/vote_k, pacing, step cap, and the whole
    code/camera/relay sections — carries over from ``base`` untouched, so
    re-running ``/setup`` never wipes hand-edits.
    """
    base = base or ThreetoksConfig()
    sink("ThreeToks setup — Enter keeps the [default].")
    return ThreetoksConfig(
        llm=_ask_llm(base, ask),
        browser=_ask_browser(base, ask),
        search=_ask_search(base, ask),
        research=_ask_research(base, ask),
        files=_ask_files(base, ask),
        memory=_ask_memory(base, ask),
        voice=_ask_voice(base, ask, sink, open_url),
        code=base.code, camera=base.camera, relay=base.relay)


def run_onboarding(ask=input, sink=print,
                   open_url=_open_url_default) -> str:
    """Run the wizard seeded from any existing config, save, report the path.

    Returns the path written: ``$THREETOKS_CONFIG`` when set, otherwise the
    per-user config file. ``open_url`` is passed through to the voice step.
    """
    config = run_wizard(load_config(), ask, sink, open_url)
    path = save_config(config)
    sink(f"config saved — {path}")
    return path


if __name__ == "__main__":
    opened = []
    record = opened.append  # fake open_url: records, never opens a browser
    quiet = lambda line: None
    answers = iter(["", "", "stealth", "y", "", "5", "", "n", "n"])
    wizard = run_wizard(ask=lambda prompt: next(answers), sink=quiet,
                        open_url=record)
    assert wizard.browser.mode == "stealth", wizard.browser
    assert wizard.browser.visible is True, wizard.browser
    assert wizard.research.max_rounds == 5, wizard.research
    assert wizard.memory.enabled is False, wizard.memory
    assert wizard.llm.model == ThreetoksConfig().llm.model, wizard.llm
    assert "threetoks" in wizard.memory.path, wizard.memory
    assert wizard.voice.enabled is False, wizard.voice  # voice declined
    assert opened == [], opened  # declining opens no pages
    retries = iter(["", "", "teleport", "plain", "no", "", "zero", "3", "",
                    "off", "n"])
    retried = run_wizard(ask=lambda prompt: next(retries), sink=quiet,
                        open_url=record)
    assert retried.browser.mode == "plain", retried.browser
    assert retried.research.max_rounds == 3, retried.research

    from threetoks.config import (CameraConfig, CodeConfig, RelayConfig,
                                 VoiceConfig)
    edited = ThreetoksConfig(
        llm=LlmConfig(provider="anthropic", api_base="http://p"),
        code=CodeConfig(retrieval=True),
        voice=VoiceConfig(enabled=True, tts_model_path="/v.onnx"),
        camera=CameraConfig(index=2), relay=RelayConfig(pin=27))
    keeps = iter(["", "", "http", "", "", "", "n", "n"])
    rerun = run_wizard(edited, ask=lambda prompt: next(keeps), sink=quiet,
                      open_url=record)
    assert rerun.llm.provider == "anthropic", rerun.llm  # never wiped
    assert rerun.llm.api_base == "http://p", rerun.llm
    assert rerun.code.retrieval is True, rerun.code
    assert rerun.voice.tts_model_path == "/v.onnx", rerun.voice  # carried
    assert rerun.voice.enabled is False, rerun.voice  # declined this run
    assert rerun.camera.index == 2 and rerun.relay.pin == 27

    opened.clear()
    accept = iter(["", "", "http", "", "", "", "n", "y", "de", "/stt",
                   "/voice.onnx"])
    voiced = run_wizard(ask=lambda prompt: next(accept), sink=quiet,
                       open_url=record)
    assert voiced.voice.enabled is True, voiced.voice
    assert voiced.voice.lang == "de", voiced.voice
    assert voiced.voice.stt_model_path == "/stt", voiced.voice
    assert voiced.voice.tts_model_path == "/voice.onnx", voiced.voice
    assert opened == [VOSK_MODELS_URL, PIPER_VOICES_URL], opened
    print("smoke OK")
