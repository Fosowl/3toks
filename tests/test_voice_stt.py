"""STT gate tests: deterministic acceptance + policy-backed pertinence.

Fully offline: never imports vosk/sounddevice, never builds SpeechListener.
"""
import unittest

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.nodes import MenuNode
from threetoks.policy import Policy, PolicyConfig
from threetoks.voice import stt


class ScriptedBackend:
    """Backend that returns pre-scripted completions in order."""

    def __init__(self, texts):
        self.texts = list(texts)

    def complete(self, model, raw_prompt, opts):
        return GenResult(self.texts.pop(0), 5, 2, 0.0, "stop")


def make_policy(texts):
    return Policy(ScriptedBackend(texts),
                  PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


def reply_for(option):
    """Digit selecting `option` under the seed-0 permutation at step 0.

    is_pertinent uses a fresh episode (step 0) and attempt 0, so the same
    permutation is reproduced here — expectations come from node geometry,
    not hardcoded digits.
    """
    options = [stt.DIRECTED_OPTION, stt.NOISE_OPTION]
    node = MenuNode("q", options, escape=False)
    perm = make_policy([])._permutation(node, 0, 0)
    return f" {perm.index(options.index(option)) + 1}"


class AcceptTranscriptTest(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(stt.accept_transcript("   ", ""))

    def test_every_known_hallucination_rejected(self):
        for junk in stt.KNOWN_HALLUCINATIONS:
            self.assertIsNone(stt.accept_transcript(junk, ""), junk)

    def test_he_is_a_known_hallucination(self):
        self.assertIn("he", stt.KNOWN_HALLUCINATIONS)
        self.assertIsNone(stt.accept_transcript("He", ""))

    def test_echo_of_tts_rejected(self):
        self.assertIsNone(stt.accept_transcript("turn the light on",
                                                "turn the light on"))

    def test_clean_transcript_passes_through_stripped(self):
        self.assertEqual(stt.accept_transcript("  what time is it  ", ""),
                         "what time is it")


class IsPertinentTest(unittest.TestCase):
    def test_directed_option_is_pertinent(self):
        policy = make_policy([reply_for(stt.DIRECTED_OPTION)])
        self.assertTrue(stt.is_pertinent("turn on the desk light", policy))

    def test_noise_option_not_pertinent(self):
        policy = make_policy([reply_for(stt.NOISE_OPTION)])
        self.assertFalse(stt.is_pertinent("and then he left the room", policy))

    def test_opposite_picks_give_opposite_verdicts(self):
        # Same transcript, so any difference is the picked option alone.
        yes = stt.is_pertinent("hello assistant",
                               make_policy([reply_for(stt.DIRECTED_OPTION)]))
        no = stt.is_pertinent("hello assistant",
                              make_policy([reply_for(stt.NOISE_OPTION)]))
        self.assertTrue(yes)
        self.assertFalse(no)

    def test_garbage_completion_fails_closed(self):
        # Unparseable output exhausts the 3-attempt ladder -> False.
        policy = make_policy(["x", "x", "x"])
        self.assertFalse(stt.is_pertinent("???", policy))


if __name__ == "__main__":
    unittest.main()
