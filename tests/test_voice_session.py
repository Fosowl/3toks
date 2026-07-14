"""Offline tests for the voice session and its TUI wiring."""
import unittest
from unittest import mock

from threetoks.nodes import Decision
from threetoks.trace import Tracer
from threetoks.tui import ReplState, _cmd_voice, _next_line, _run_task
from threetoks.voice.session import (HEARD, IGNORED, VoiceSession,
                                     make_voice_session)
from threetoks.voice.stt import DIRECTED_OPTION, NOISE_OPTION


class FakeListener:
    def __init__(self, transcripts=()):
        self.transcripts = list(transcripts)
        self.started = False
        self.muted = None

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


class FakeSpeaker:
    def __init__(self):
        self.on_start = None
        self.on_stop = None
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class FakePolicy:
    """decide() always picks the given option text."""

    def __init__(self, option):
        self.option = option
        self.tracer = Tracer(None)

    def decide(self, episode, node, think_budget=0):
        return Decision("menu", self.option, "1", valid=True)


class VoiceSessionTest(unittest.TestCase):
    def test_wires_mute_callbacks_and_starts_listener(self):
        listener, speaker = FakeListener(), FakeSpeaker()
        VoiceSession(listener, speaker)
        self.assertTrue(listener.started)
        self.assertEqual(speaker.on_start, listener.mute)  # bound methods
        self.assertEqual(speaker.on_stop, listener.unmute)

    def test_listen_returns_heard_for_directed_speech(self):
        session = VoiceSession(FakeListener(["what time is it"]), None)
        gated = session.listen(FakePolicy(DIRECTED_OPTION))
        self.assertEqual(gated, (HEARD, "what time is it"))

    def test_listen_flags_noise_as_ignored(self):
        session = VoiceSession(FakeListener(["random tv chatter"]), None)
        gated = session.listen(FakePolicy(NOISE_OPTION))
        self.assertEqual(gated, (IGNORED, "random tv chatter"))

    def test_listen_drops_echo_of_own_answer_silently(self):
        session = VoiceSession(FakeListener(["the light is now on"]), None)
        session.speaker = None
        session.speak("The light is now on.")
        self.assertIsNone(session.listen(FakePolicy(DIRECTED_OPTION)))

    def test_quiet_tick_and_hallucination_return_none(self):
        session = VoiceSession(FakeListener(["hi"]), None)
        self.assertIsNone(session.listen(FakePolicy(DIRECTED_OPTION)))
        self.assertIsNone(session.listen(FakePolicy(DIRECTED_OPTION)))

    def test_speak_records_history_and_delegates(self):
        speaker = FakeSpeaker()
        session = VoiceSession(None, speaker)
        session.speak("Hello there.")
        session.speak("")
        self.assertEqual(speaker.spoken, ["Hello there."])
        self.assertFalse(session.hears)

    def test_close_stops_the_listener(self):
        listener = FakeListener()
        session = VoiceSession(listener, None)
        session.close()
        self.assertFalse(listener.started)


class MakeVoiceSessionTest(unittest.TestCase):
    @staticmethod
    def _broken(_config):
        raise ImportError("extra not installed")

    def test_both_sides_down_raises_with_notes(self):
        with self.assertRaisesRegex(RuntimeError, "no voice capability"):
            make_voice_session(None, self._broken, self._broken)

    def test_one_side_up_returns_session_and_notes(self):
        session, notes = make_voice_session(None,
                                            lambda _: FakeListener(),
                                            self._broken)
        self.assertTrue(session.hears)
        self.assertIsNone(session.speaker)
        self.assertEqual(len(notes), 1)
        self.assertIn("3toks[tts]", notes[0])


def make_state(**overrides):
    fields = dict(services=object(), specs=[], policy=FakePolicy("x"),
                  model="m", policy_factory=None, enabled=False)
    fields.update(overrides)
    return ReplState(**fields)


class CmdVoiceTest(unittest.TestCase):
    def test_bad_argument_shows_usage(self):
        self.assertIn("usage", _cmd_voice(make_state(), "loud"))

    def test_bare_reports_status_and_off_is_safe_when_absent(self):
        state = make_state()
        self.assertIn("voice is off", _cmd_voice(state, ""))
        self.assertIn("keyboard", _cmd_voice(state, "off"))

    def test_on_builds_a_session_and_off_closes_it(self):
        state = make_state()
        session = VoiceSession(FakeListener(), None)
        with mock.patch("threetoks.voice.session.make_voice_session",
                        return_value=(session, ["note"])) as factory:
            reply = _cmd_voice(state, "on")
        factory.assert_called_once_with(state.voice_config)
        self.assertIs(state.voice, session)
        self.assertIn("note", reply)
        _cmd_voice(state, "off")
        self.assertIsNone(state.voice)
        self.assertFalse(session.listener.started)

    def test_on_failure_surfaces_a_styled_message(self):
        state = make_state()
        with mock.patch("threetoks.voice.session.make_voice_session",
                        side_effect=RuntimeError("no voice capability: x")):
            reply = _cmd_voice(state, "on")
        self.assertIsNone(state.voice)
        self.assertIn("voice unavailable", reply)


class VoiceInputLoopTest(unittest.TestCase):
    def test_next_line_uses_keyboard_without_voice(self):
        state = make_state()
        line = _next_line(state, lambda prompt: "typed", "> ", lambda _: None)
        self.assertEqual(line, "typed")

    def test_next_line_skips_gated_ticks_then_returns_the_transcript(self):
        state = make_state(policy=FakePolicy(DIRECTED_OPTION))
        state.voice = VoiceSession(
            FakeListener(["hi", "what time is it"]), None)  # "hi" is junk
        shown = []
        line = _next_line(state, None, "> ", shown.append)
        self.assertEqual(line, "what time is it")
        self.assertTrue(any("what time is it" in row for row in shown))

    def test_ignored_transcript_is_shown_then_listening_continues(self):
        state = make_state(policy=FakePolicy(NOISE_OPTION))
        state.voice = VoiceSession(FakeListener(["blah blah blah"]), None)
        state.running = True

        def stop_soon(row):
            shown.append(row)
            state.running = "ignored" not in row  # exit after the note

        shown = []
        line = _next_line(state, None, "> ", stop_soon)
        self.assertIsNone(line)
        self.assertTrue(any("ignored" in row for row in shown))

    def test_ctrl_c_while_listening_falls_back_to_keyboard(self):
        class InterruptingListener(FakeListener):
            def listen(self, timeout_s=None):
                raise KeyboardInterrupt

        state = make_state()
        state.voice = VoiceSession(InterruptingListener(), None)
        line = _next_line(state, lambda prompt: "typed", "> ",
                          lambda _: None)
        self.assertEqual(line, "typed")
        self.assertIsNone(state.voice)

    def test_backend_error_while_listening_never_unwinds_the_repl(self):
        class DyingListener(FakeListener):
            def listen(self, timeout_s=None):
                raise RuntimeError("cannot reach http://llm: refused")

        state = make_state()
        state.voice = VoiceSession(DyingListener(), None)
        shown = []
        line = _next_line(state, lambda prompt: "typed", "> ", shown.append)
        self.assertEqual(line, "typed")  # session survived, keyboard next
        self.assertIsNone(state.voice)
        self.assertTrue(any("cannot reach" in row for row in shown))


class SpeakHookTest(unittest.TestCase):
    def test_run_task_speaks_the_answer_in_voice_mode(self):
        from threetoks.agents.base import AgentSpec
        spec = AgentSpec("stub", "stub agent",
                         lambda task, services, policy: {"answer": "42"})
        state = make_state(specs=[spec])
        speaker = FakeSpeaker()
        state.voice = VoiceSession(None, speaker)
        _run_task(state, "meaning of life",
                  lambda task, specs, policy: spec, lambda _: None)
        self.assertEqual(speaker.spoken, ["42"])


if __name__ == "__main__":
    unittest.main()
