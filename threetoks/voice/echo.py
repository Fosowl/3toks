"""Echo filter: drop the assistant's own TTS speech re-heard by the mic.

Single responsibility: decide whether a transcript is the STT hearing the
TTS, by matching normalized consecutive-word runs in both directions. Zero
third-party imports so the voice package still imports on a bare install.
"""
import re
from typing import Iterable, Union

_APOSTROPHE_RE = re.compile(r"[’ʼ']")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_DEFAULT_WINDOW = 3  # consecutive words that constitute an echo

SpokenHistory = Union[str, Iterable[str]]


def _normalize(text: str) -> str:
    """Lowercase, drop apostrophes, collapse punctuation to single spaces.

    Input: raw transcript or TTS text. Output: a normalized token string
    ("It's ON!" -> "its on") so case and punctuation can't slip past.
    """
    if not text:
        return ""
    lowered = _APOSTROPHE_RE.sub("", text.lower())
    return " ".join(_NON_WORD_RE.sub(" ", lowered).split())


def _flatten_spoken(spoken: SpokenHistory) -> str:
    """Normalize a string, or an iterable of strings joined by spaces."""
    if isinstance(spoken, str):
        return _normalize(spoken)
    return " ".join(_normalize(part) for part in spoken if part)


def _window_hit(words: list[str], haystack: str, window: int) -> bool:
    """True if any `window` consecutive `words` occur in `haystack`."""
    for start in range(len(words) - window + 1):
        if " ".join(words[start:start + window]) in haystack:
            return True
    return False


def is_echo(text: str, spoken: SpokenHistory,
            min_consecutive_words: int = _DEFAULT_WINDOW) -> bool:
    """Return True if `text` looks like the assistant hearing its own TTS.

    `text`: candidate transcript. `spoken`: recent TTS as a string or an
    iterable of strings (flattened by joining with spaces). Both sides are
    normalized first. Short transcripts (fewer than min_consecutive_words
    words) use a whole-token substring check; longer ones slide a word
    window in BOTH directions so overlap is caught whichever side is longer.
    """
    if not text or not spoken:
        return False
    text_norm = _normalize(text)
    spoken_norm = _flatten_spoken(spoken)
    if not text_norm or not spoken_norm:
        return False
    text_words = text_norm.split()
    if len(text_words) < min_consecutive_words:
        return f" {text_norm} " in f" {spoken_norm} "
    window = min_consecutive_words
    return (_window_hit(text_words, spoken_norm, window)
            or _window_hit(spoken_norm.split(), text_norm, window))


if __name__ == "__main__":
    assert _normalize("It's ON!") == "its on"
    assert _normalize("light,  on!!!") == "light on"
    assert is_echo("please turn the light on", "sure, turn the light on now")
    assert is_echo("sure turn the light on now", "turn the light")  # both dirs
    assert is_echo("hi", ["well hi there"])  # short-token substring path
    assert not is_echo("hi", "goodbye everyone")
    assert not is_echo("what is the capital of France", "turn the light on")
    assert is_echo("it's on the desk", "its on the desk")  # apostrophes
    print("smoke OK")
