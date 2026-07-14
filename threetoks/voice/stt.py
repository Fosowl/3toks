"""Speech-to-text intake: deterministic gates plus a Vosk mic pipeline.

Single responsibility: turn microphone audio into transcripts the assistant
should act on. Two cheap stdlib gates run first — a hallucination/echo
filter (``gate_transcript``, or ``accept_transcript`` when the drop reason
is not wanted) then a model-backed pertinence check (``is_pertinent``) —
before the expensive assistant is ever consulted.
``SpeechListener`` wraps Vosk + sounddevice; those extras are imported
lazily so this module (and its gates) import on a bare install.
"""
import json
import queue
import sys
import threading
import time

from threetoks.nodes import MenuNode
from threetoks.policy import Policy
from threetoks.render import Episode
from threetoks.voice.echo import SpokenHistory, is_echo

# Tokens Vosk reliably invents from silence or ambient noise; "he" is
# added because the reference main loop skipped it too.
KNOWN_HALLUCINATIONS = ("hi", "hi.", "hey", "hey.", "hum", "huh", "he")

# Why the deterministic gate let a transcript through, or dropped it.
GATE_OK = "ok"
GATE_EMPTY = "empty"
GATE_HALLUCINATION = "hallucination"
GATE_ECHO = "echo"

# Pertinence gate: two semantically described options, never bare yes/no.
DIRECTED_OPTION = "a task, question, or command meant for the assistant"
NOISE_OPTION = "background noise, other people talking, or fragmented speech"
_GATE_PREFIX = (
    "You gate an always-listening microphone that constantly picks up TV, "
    "music, and side conversations. Let through only speech aimed at the "
    "assistant; when unsure, treat it as noise.")
_GATE_TASK = "classify one overheard transcript"

# Audio stream shape (Vosk expects 16-bit mono PCM).
BLOCK_SIZE = 8000
_CHANNELS = 1
_DTYPE = "int16"
_LOG_SILENT = -1  # Vosk level: mute model chatter


def gate_transcript(text: str,
                    spoken_history: SpokenHistory) -> tuple[str | None, str]:
    """Deterministic pre-model gate reporting why it decided.

    `text`: raw STT output. `spoken_history`: recent TTS (string or
    iterable of strings) for echo rejection. Returns ``(accepted,
    reason)``: the stripped transcript with GATE_OK, else None with
    GATE_EMPTY, GATE_HALLUCINATION, or GATE_ECHO. The reason exists so
    voice debug mode can show what vanished. House rule: these cheap
    checks run before any model call.
    """
    stripped = text.strip()
    if not stripped:
        return None, GATE_EMPTY
    if stripped.lower() in KNOWN_HALLUCINATIONS:
        return None, GATE_HALLUCINATION
    if is_echo(stripped, spoken_history):
        return None, GATE_ECHO
    return stripped, GATE_OK


def accept_transcript(text: str, spoken_history: SpokenHistory) -> str | None:
    """Deterministic pre-model gate on one raw transcript.

    `text`: raw STT output. `spoken_history`: recent TTS (string or
    iterable of strings) for echo rejection. Returns the stripped
    transcript, or None when it is empty, a known silence hallucination,
    or an echo of the assistant's own speech — :func:`gate_transcript`
    without the reason, for callers that only want the text.
    """
    return gate_transcript(text, spoken_history)[0]


def is_pertinent(text: str, policy: Policy) -> bool:
    """True only if the transcript is plausibly addressed to the assistant.

    Replaces the reference COHERENT/GIBBERISH prompt with the house
    mechanism: a two-option menu (no escape) decided by `policy`. `text`:
    an accepted transcript. Returns True only when the decision is valid
    AND picks the directed-at-assistant option; any invalid parse or the
    noise option returns False — fail closed.
    """
    episode = Episode(_GATE_PREFIX, _GATE_TASK)
    question = f'Which best describes this transcript? "{text}"'
    node = MenuNode(question, [DIRECTED_OPTION, NOISE_OPTION], escape=False)
    decision = policy.decide(episode, node)
    return decision.valid and decision.value == DIRECTED_OPTION


class SpeechListener:
    """Vosk microphone pipeline with a hard mute gate for anti-echo.

    A sounddevice RawInputStream feeds raw PCM blocks into a queue that a
    Vosk recognizer drains one FINAL utterance at a time. vosk/sounddevice
    are imported lazily, so importing this module needs no audio extras.
    While muted (wired to TTS start/stop) incoming audio is dropped and the
    recognizer reset, so the assistant never transcribes its own voice.
    """

    def __init__(self, lang: str = "en-us", model_path: str | None = None,
                 device: int | None = None, block_size: int = BLOCK_SIZE,
                 sample_rate: int | None = None):
        """Load the model and prepare the recognizer (no stream opened yet).

        lang: Vosk model language when model_path is None (auto-downloads).
        model_path: local model folder; overrides lang when given. device:
        input device id (system default when None). block_size: samples per
        read. sample_rate: forced rate, else the device default is used.
        """
        from vosk import KaldiRecognizer, Model, SetLogLevel  # lazy extra
        import sounddevice as sd  # lazy extra
        SetLogLevel(_LOG_SILENT)
        self._device = device
        self._block_size = block_size
        self._sample_rate = sample_rate or _default_sample_rate(sd, device)
        model = Model(model_path) if model_path else Model(lang=lang)
        self._recognizer = KaldiRecognizer(model, self._sample_rate)
        self._queue: queue.Queue = queue.Queue()
        self._stream = None
        self._muted = False
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status) -> None:
        """Stream callback: drop audio while muted, else enqueue raw bytes."""
        if status:
            print(f"audio status: {status}", file=sys.stderr)
        if self._muted:
            return
        self._queue.put(bytes(indata))

    def start(self) -> "SpeechListener":
        """Open the microphone stream (idempotent). Returns self."""
        if self._stream is not None:
            return self
        import sounddevice as sd  # lazy extra
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate, blocksize=self._block_size,
            device=self._device, dtype=_DTYPE, channels=_CHANNELS,
            callback=self._callback)
        self._stream.start()
        return self

    def stop(self) -> None:
        """Close the microphone stream and release the device."""
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _drain(self) -> None:
        """Discard every queued audio block."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def mute(self) -> None:
        """Drop audio and reset the recognizer while the TTS is speaking."""
        with self._lock:
            self._muted = True
            self._drain()
            self._recognizer.Reset()

    def unmute(self) -> None:
        """Resume listening; drop residual samples so TTS tails don't leak."""
        with self._lock:
            self._drain()
            self._recognizer.Reset()
            self._muted = False

    def _next_block(self, deadline: float | None) -> bytes | None:
        """Next queued block honoring the absolute `deadline`, else None."""
        wait = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            return self._queue.get(timeout=wait)
        except queue.Empty:
            return None

    def listen(self, timeout_s: float | None = None) -> str | None:
        """Block for one FINAL Vosk transcript.

        timeout_s: max seconds to wait (None waits forever). Returns the
        transcript text, or None on timeout or while muted. Feeds queued
        audio to the recognizer until it reports a complete utterance.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while not self._muted:
            block = self._next_block(deadline)
            if block is None:
                return None
            if self._recognizer.AcceptWaveform(block):
                return json.loads(self._recognizer.Result()).get("text") or None
        return None

    def __enter__(self) -> "SpeechListener":
        """Start listening on context entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop listening on context exit."""
        self.stop()


def _default_sample_rate(sd, device: int | None) -> int:
    """Default input sample rate for `device`, queried via sounddevice."""
    info = sd.query_devices(device, "input")
    return int(info["default_samplerate"])


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import PolicyConfig

    assert accept_transcript("   ", "") is None
    assert accept_transcript("hi", "") is None            # hallucination
    assert accept_transcript("he", "") is None            # skipped like ref
    assert accept_transcript("turn the light on",
                             "turn the light on") is None  # echo
    assert accept_transcript(" hello there ", "") == "hello there"

    assert gate_transcript("   ", "") == (None, GATE_EMPTY)
    assert gate_transcript("hi", "") == (None, GATE_HALLUCINATION)
    assert gate_transcript("turn the light on",
                           "turn the light on") == (None, GATE_ECHO)
    assert gate_transcript(" hello there ", "") == ("hello there", GATE_OK)

    class _PickDirected:
        """Fake backend: reply with the digit that shows DIRECTED_OPTION."""

        def complete(self, model, prompt, opts):
            for line in prompt.splitlines():
                if line.endswith(DIRECTED_OPTION):
                    return GenResult(f" {line.split(' = ')[0]}", 5, 2, 0.0, "s")
            return GenResult(" 9", 5, 2, 0.0, "s")  # unreachable escape

    policy = Policy(_PickDirected(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    assert is_pertinent("turn on the desk light", policy) is True
    print("smoke OK")
