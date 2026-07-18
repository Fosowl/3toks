"""Offline tests for the text-to-speech module (no audio hardware).

Never imports piper or sounddevice, and never instantiates ``Speaker``
with a real voice model. ``split_sentences`` and ``_pcm_to_float`` are
exercised directly; playback lifecycle is checked by stubbing the
synthesize/play steps on a bare ``Speaker`` instance built via
``__new__``. The PCM case is skipped when numpy is absent.
"""
import importlib.util
import io
import struct
import unittest
import wave

from threetoks.voice import tts
from threetoks.voice.tts import (INT16_PEAK, UINT8_MIDPOINT, VOLUME_SCALE,
                                 Speaker, split_sentences)

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class SplitSentencesTest(unittest.TestCase):
    def test_multiple_sentences(self):
        self.assertEqual(split_sentences("Hi there. How are you? Fine!"),
                         ["Hi there.", "How are you?", "Fine!"])

    def test_single_sentence(self):
        self.assertEqual(split_sentences("Just one."), ["Just one."])

    def test_trailing_whitespace_dropped(self):
        self.assertEqual(split_sentences("Done.   "), ["Done."])

    def test_empty_string(self):
        self.assertEqual(split_sentences(""), [])

    def test_whitespace_only(self):
        self.assertEqual(split_sentences("   \n\t"), [])

    def test_exclaim_and_question(self):
        self.assertEqual(split_sentences("Wow! Really?"), ["Wow!", "Really?"])

    def test_no_terminal_punctuation(self):
        self.assertEqual(split_sentences("no period here"),
                         ["no period here"])


def _make_wav(sample_width: int, channels: int, frames: bytes) -> io.BytesIO:
    """Write frames to a rewound in-memory WAV and return the buffer."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(22050)  # arbitrary; decode ignores it
        writer.writeframes(frames)
    buffer.seek(0)
    return buffer


def _decode(sample_width: int, channels: int, frames: bytes):
    """Run ``_pcm_to_float`` over a freshly built WAV buffer."""
    with wave.open(_make_wav(sample_width, channels, frames), "rb") as reader:
        return tts._pcm_to_float(reader)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class PcmToFloatTest(unittest.TestCase):
    def test_16bit_mono_dtype_and_attenuated_range(self):
        import numpy
        raw = struct.pack("<3h", 0, INT16_PEAK // 2, -(INT16_PEAK // 2))
        out = _decode(2, 1, raw)
        self.assertEqual(out.dtype, numpy.float32)
        self.assertEqual(out.shape, (3,))
        expected = numpy.array([0.0, 0.5, -0.5], numpy.float32) * VOLUME_SCALE
        numpy.testing.assert_allclose(out, expected, atol=1e-6)
        self.assertLessEqual(float(numpy.abs(out).max()), VOLUME_SCALE)

    def test_16bit_stereo_downmixed_by_channel_average(self):
        import numpy
        raw = struct.pack("<4h", 10000, 20000, -10000, -20000)
        out = _decode(2, 2, raw)
        self.assertEqual(out.shape, (2,))  # two frames, averaged to mono
        expected = (numpy.array([15000, -15000], numpy.float32)
                    / INT16_PEAK * VOLUME_SCALE)
        numpy.testing.assert_allclose(out, expected, atol=1e-6)

    def test_8bit_mono_offset_and_range(self):
        import numpy
        raw = struct.pack("<3B", 0, UINT8_MIDPOINT, 255)
        out = _decode(1, 1, raw)
        self.assertEqual(out.dtype, numpy.float32)
        top = (255 - UINT8_MIDPOINT) / UINT8_MIDPOINT
        expected = numpy.array([-1.0, 0.0, top], numpy.float32) * VOLUME_SCALE
        numpy.testing.assert_allclose(out, expected, atol=1e-6)
        self.assertGreaterEqual(float(out.min()), -VOLUME_SCALE - 1e-6)

    def test_unsupported_sample_width_raises(self):
        raw = b"\x00\x00\x00\x01\x02\x03"  # two 24-bit frames
        with self.assertRaises(ValueError):
            _decode(3, 1, raw)


def _bare_speaker(events, on_start=None, on_stop=None, buffers=("b",)):
    """Build a ``Speaker`` without a model and stub its playback steps."""
    speaker = Speaker.__new__(Speaker)
    speaker.on_start = on_start
    speaker.on_stop = on_stop
    speaker.post_speak_delay = 0  # never sleep in tests
    speaker._synthesize_all = lambda text: list(buffers)
    speaker._play = lambda buffer: events.append(f"play:{buffer}")
    return speaker


class CallbackOrderingTest(unittest.TestCase):
    def test_on_start_precedes_playback_and_on_stop_follows(self):
        events = []
        speaker = _bare_speaker(
            events,
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
            buffers=["x", "y"])
        speaker.speak("anything")
        self.assertEqual(events, ["start", "play:x", "play:y", "stop"])

    def test_on_stop_fires_even_when_playback_raises(self):
        events = []
        speaker = _bare_speaker(
            events,
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
            buffers=["x"])

        def boom(buffer):
            events.append("play")
            raise RuntimeError("audio device gone")

        speaker._play = boom
        with self.assertRaises(RuntimeError):
            speaker.speak("hi")
        self.assertEqual(events, ["start", "play", "stop"])

    def test_raising_on_stop_propagates(self):
        events = []

        def bad_stop():
            raise ValueError("unmute failed")

        speaker = _bare_speaker(events, on_stop=bad_stop, buffers=["x"])
        with self.assertRaises(ValueError):
            speaker.speak("hi")

    def test_empty_text_fires_no_callbacks(self):
        events = []
        speaker = _bare_speaker(
            events,
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
            buffers=[])
        speaker.speak("")
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
