"""Text-to-speech output: speak answers aloud via a local Piper voice.

Single responsibility: turn text into spoken audio. Sentence splitting
and PCM decoding are pure and offline-testable; piper, sounddevice and
numpy are imported lazily so a bare install still imports the package.
"""
import io
import re
import time
import wave

POST_SPEAK_DELAY_S = 0.8      # let room reverb decay before unmuting
VOLUME_SCALE = 0.2            # fixed playback attenuation

SAMPLE_WIDTH_8BIT = 1
SAMPLE_WIDTH_16BIT = 2
SAMPLE_WIDTH_32BIT = 4
UINT8_MIDPOINT = 128         # centres unsigned 8-bit around zero
INT16_PEAK = 32768           # 2 ** 15
INT32_PEAK = 2147483648      # 2 ** 31
MONO_CHANNELS = 1
STEREO_CHANNELS = 2

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation.

    Inputs: raw text. Splits after ``. ! ?`` followed by whitespace and
    drops empty or whitespace-only pieces. Returns a list of sentences
    (a single element when unpunctuated, empty for blank input).
    """
    pieces = SENTENCE_BOUNDARY.split(text.strip())
    return [piece for piece in pieces if piece.strip()]


def _decode_samples(raw: bytes, sample_width: int, numpy):
    """Normalize raw PCM bytes to float32 in [-1, 1] by bit depth.

    Inputs: interleaved PCM bytes, the sample width, and the numpy
    module. Returns a float32 array. Raises ValueError on odd widths.
    """
    if sample_width == SAMPLE_WIDTH_8BIT:
        data = numpy.frombuffer(raw, dtype=numpy.uint8).astype(numpy.float32)
        return (data - UINT8_MIDPOINT) / UINT8_MIDPOINT
    if sample_width == SAMPLE_WIDTH_16BIT:
        data = numpy.frombuffer(raw, dtype=numpy.int16).astype(numpy.float32)
        return data / INT16_PEAK
    if sample_width == SAMPLE_WIDTH_32BIT:
        data = numpy.frombuffer(raw, dtype=numpy.int32).astype(numpy.float32)
        return data / INT32_PEAK
    raise ValueError(f"unsupported sample width: {sample_width}")


def _downmix(samples, n_channels: int):
    """Collapse interleaved multi-channel samples to a mono array."""
    if n_channels == MONO_CHANNELS:
        return samples
    if n_channels == STEREO_CHANNELS:
        return samples.reshape(-1, STEREO_CHANNELS).mean(axis=1)
    return samples[::n_channels]  # >2 channels: take channel 0


def _pcm_to_float(wav_reader):
    """Decode a wave reader to a mono float32 array ready for playback.

    Inputs: an open ``wave`` read handle. Normalizes by bit depth,
    downmixes to mono, and applies fixed attenuation. Returns a numpy
    float32 array; numpy is imported lazily.
    """
    import numpy
    raw = wav_reader.readframes(wav_reader.getnframes())
    samples = _decode_samples(raw, wav_reader.getsampwidth(), numpy)
    samples = _downmix(samples, wav_reader.getnchannels())
    return (samples * VOLUME_SCALE).astype(numpy.float32)


class Speaker:
    """Piper text-to-speech with mic-muting callbacks around playback.

    Synthesizes text to in-memory WAV, fires ``on_start`` just before
    audio output (so the mic mutes only during playback), plays each
    sentence, waits for room reverb to decay, then fires ``on_stop`` in
    a ``finally``. Callbacks are instance-level callables (or None) and
    their exceptions propagate: a silently failed unmute would
    re-introduce echo into the microphone.

    Voice models (.onnx plus their .onnx.json sidecar) are NOT
    auto-downloaded; ``model_path`` must point at a local file, e.g. one
    from the rhasspy/piper-voices Hugging Face repo.
    """

    def __init__(self, model_path: str, speaker_id: int | None = None,
                 on_start=None, on_stop=None,
                 post_speak_delay: float = POST_SPEAK_DELAY_S):
        """Load a Piper voice and store playback callbacks.

        Inputs: path to a local .onnx voice model, optional
        multi-speaker id, optional on_start/on_stop callables fired
        around playback, and the post-speak delay in seconds. Importing
        piper here implies the tts extra is installed.
        """
        from piper import PiperVoice
        self._voice = PiperVoice.load(model_path, speaker_id)
        self.on_start = on_start
        self.on_stop = on_stop
        self.post_speak_delay = post_speak_delay

    def speak(self, text: str) -> None:
        """Synthesize every sentence, then play it while muting the mic.

        Inputs: text to speak. Synthesis happens up front (before the
        mic is muted); on_start fires once before playback, each buffer
        plays in order, the delay lets reverb decay, and on_stop fires
        in ``finally``. No-op when text yields no sentences.
        """
        buffers = self._synthesize_all(text)
        if not buffers:
            return
        self._fire(self.on_start)
        try:
            for buffer in buffers:
                self._play(buffer)
            if self.post_speak_delay > 0:
                time.sleep(self.post_speak_delay)
        finally:
            self._fire(self.on_stop)

    def _synthesize_all(self, text: str) -> list:
        """Render each sentence of text to a rewound WAV buffer."""
        return [self._synthesize(sentence)
                for sentence in split_sentences(text)]

    def _synthesize(self, sentence: str) -> io.BytesIO:
        """Render one sentence to an in-memory, rewound WAV buffer."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            self._voice.synthesize_wav(sentence, wav_file)
        buffer.seek(0)
        return buffer

    def _play(self, buffer: io.BytesIO) -> None:
        """Decode a WAV buffer to float32 and play it, blocking."""
        import sounddevice
        with wave.open(buffer, "rb") as reader:
            rate = reader.getframerate()
            samples = _pcm_to_float(reader)
        sounddevice.play(samples, samplerate=rate)
        sounddevice.wait()

    @staticmethod
    def _fire(callback) -> None:
        """Invoke a lifecycle callback if one was supplied."""
        if callback is not None:
            callback()


if __name__ == "__main__":
    import importlib.util

    assert split_sentences("A. B! C?") == ["A.", "B!", "C?"]
    assert split_sentences("") == []
    assert split_sentences("   \n\t") == []
    assert split_sentences("tail with no stop  ") == ["tail with no stop"]
    if importlib.util.find_spec("numpy"):
        import numpy

        smoke = io.BytesIO()
        with wave.open(smoke, "wb") as writer:
            writer.setnchannels(MONO_CHANNELS)
            writer.setsampwidth(SAMPLE_WIDTH_16BIT)
            writer.setframerate(22050)  # Piper default sample rate
            frames = numpy.array([0, INT16_PEAK // 2, -(INT16_PEAK // 2)],
                                 dtype=numpy.int16)
            writer.writeframes(frames.tobytes())
        smoke.seek(0)
        with wave.open(smoke, "rb") as reader:
            decoded = _pcm_to_float(reader)
        assert decoded.dtype == numpy.float32
        assert float(numpy.abs(decoded).max()) <= VOLUME_SCALE + 1e-6
    else:
        print("numpy absent: _pcm_to_float smoke skipped")
    print("smoke OK")
