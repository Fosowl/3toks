"""Voice session: compose the STT gates, the TTS speaker, and echo state.

One VoiceSession owns whichever voice sides are installed (stt and tts
degrade independently), wires the speaker's start/stop callbacks to the
listener's hard mute, and keeps the rolling spoken history that feeds
the echo filter. The TUI talks only to this class.
"""
from collections import deque

from threetoks.voice.stt import accept_transcript, is_pertinent

LISTEN_TICK_S = 0.5     # short poll so Ctrl-C stays responsive
SPOKEN_HISTORY = 3      # answers remembered for echo rejection
HEARD = "heard"         # transcript aimed at the assistant
IGNORED = "ignored"     # transcript gated out by the pertinence check


class VoiceSession:
    """Microphone in, speaker out, with anti-echo state in between.

    Inputs: a started-or-startable SpeechListener (or None for typed
    input), a Speaker (or None for text-only output). When both exist
    the speaker mutes the microphone for the whole playback window.
    """

    def __init__(self, listener=None, speaker=None):
        self.listener = listener
        self.speaker = speaker
        self._spoken: deque = deque(maxlen=SPOKEN_HISTORY)
        if listener is not None and speaker is not None:
            speaker.on_start = listener.mute
            speaker.on_stop = listener.unmute
        if listener is not None:
            listener.start()

    @property
    def hears(self) -> bool:
        """Whether voice input is available."""
        return self.listener is not None

    def listen(self, policy) -> tuple | None:
        """One gated transcript as ``(HEARD | IGNORED, text)``, or None.

        Blocks up to LISTEN_TICK_S for a final transcript, then runs the
        deterministic gates (empty / hallucination / echo of our own
        speech — those yield None like a quiet tick) and the model
        pertinence gate via ``policy`` (a rejection yields IGNORED so
        the caller can show it). Callers loop on None.
        """
        raw = self.listener.listen(timeout_s=LISTEN_TICK_S)
        if raw is None:
            return None
        text = accept_transcript(raw, tuple(self._spoken))
        if text is None:
            return None
        if not is_pertinent(text, policy):
            return IGNORED, text
        return HEARD, text

    def speak(self, answer: str) -> None:
        """Speak an answer and remember it for echo rejection."""
        if not answer:
            return
        self._spoken.append(answer)
        if self.speaker is not None:
            self.speaker.speak(answer)

    def close(self) -> None:
        """Stop the microphone stream (idempotent)."""
        if self.listener is not None:
            self.listener.stop()


def _default_listener(voice_config):
    """A SpeechListener from config; ImportError when stt extra missing."""
    from threetoks.voice.stt import SpeechListener
    return SpeechListener(lang=voice_config.lang,
                          model_path=voice_config.stt_model_path or None)


def _default_speaker(voice_config):
    """A Speaker from config; raises when the tts side cannot start."""
    from threetoks.voice.tts import Speaker
    if not voice_config.tts_model_path:
        raise FileNotFoundError("set [voice] tts_model_path to a piper "
                                ".onnx voice (not auto-downloaded)")
    speaker_id = None if voice_config.tts_speaker < 0 \
        else voice_config.tts_speaker
    return Speaker(voice_config.tts_model_path, speaker_id=speaker_id,
                   post_speak_delay=voice_config.post_speak_delay)


def make_voice_session(voice_config, listener_factory=_default_listener,
                       speaker_factory=_default_speaker):
    """Build a VoiceSession from whichever voice extras are installed.

    Each side is tried independently; a side that fails (missing extra,
    missing voice model) is skipped with a note. Returns ``(session,
    notes)``; raises RuntimeError when NEITHER side is available. The
    factories are injectable for offline tests.
    """
    notes = []
    listener = _try_side(lambda: listener_factory(voice_config), "voice input",
                         "3toks[stt]", notes)
    speaker = _try_side(lambda: speaker_factory(voice_config), "voice output",
                        "3toks[tts]", notes)
    if listener is None and speaker is None:
        raise RuntimeError("no voice capability: " + "; ".join(notes))
    return VoiceSession(listener, speaker), notes


def _try_side(factory, label: str, extra: str, notes: list):
    """Run one side's factory; on failure append a note and return None."""
    try:
        return factory()
    except (ImportError, FileNotFoundError, OSError) as error:
        notes.append(f"{label} off — {error} (install: uv pip install "
                     f"'{extra}')")
        return None


if __name__ == "__main__":
    class _FakeListener:
        def __init__(self, transcripts=("turn on the light",)):
            self.started, self.muted = False, None
            self.transcripts = list(transcripts)

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def mute(self):
            self.muted = True

        def unmute(self):
            self.muted = False

        def listen(self, timeout_s=None):
            return self.transcripts.pop(0) if self.transcripts else None

    class _FakeSpeaker:
        def __init__(self):
            self.on_start, self.on_stop, self.spoken = None, None, []

        def speak(self, text):
            self.spoken.append(text)

    class _YesPolicy:
        def decide(self, episode, node):
            from threetoks.nodes import Decision
            return Decision("menu", node.options[0], "1", valid=True)

    listener = _FakeListener(["turn on the light", "the light is now on"])
    speaker = _FakeSpeaker()
    session = VoiceSession(listener, speaker)
    assert listener.started and speaker.on_start == listener.mute
    kind, text = session.listen(_YesPolicy())
    assert (kind, text) == (HEARD, "turn on the light")
    session.speak("The light is now on.")
    assert speaker.spoken == ["The light is now on."]
    assert session.listen(_YesPolicy()) is None  # echo of our own answer
    session.close()
    assert not listener.started

    def _broken(_):
        raise ImportError("vosk not installed")

    try:
        make_voice_session(None, _broken, _broken)
        raise AssertionError("both sides down should raise")
    except RuntimeError as error:
        assert "no voice capability" in str(error)
    half, notes = make_voice_session(None, lambda _: _FakeListener(), _broken)
    assert half.hears and half.speaker is None and len(notes) == 1
    print("smoke OK")
