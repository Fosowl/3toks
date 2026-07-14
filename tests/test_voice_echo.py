"""Echo-filter tests: pure, offline, no audio extras imported."""
import unittest

from threetoks.voice.echo import _normalize, is_echo


class NormalizeTest(unittest.TestCase):
    def test_lowercases_and_drops_apostrophes(self):
        self.assertEqual(_normalize("It's ON"), "its on")

    def test_collapses_punctuation_to_single_spaces(self):
        self.assertEqual(_normalize("light,  on!!!"), "light on")

    def test_empty_text_normalizes_to_empty(self):
        self.assertEqual(_normalize(""), "")


class IsEchoTest(unittest.TestCase):
    def test_exact_echo_rejected(self):
        self.assertTrue(is_echo("turn the light on", "turn the light on"))

    def test_overlap_with_longer_spoken_rejected(self):
        # 3-word run of text sits inside a longer spoken line.
        self.assertTrue(is_echo("please turn the light",
                                "sure, turn the light on now"))

    def test_overlap_with_longer_text_rejected(self):
        # Symmetric direction: the shared run is caught when text is longer.
        self.assertTrue(is_echo("sure turn the light on now please",
                                "turn the light"))

    def test_short_utterance_contained_in_spoken_rejected(self):
        self.assertTrue(is_echo("hi", "well hi there friend"))

    def test_short_utterance_not_contained_passes(self):
        self.assertFalse(is_echo("hi", "goodbye everyone"))

    def test_unrelated_text_passes(self):
        self.assertFalse(is_echo("what is the capital of France",
                                 "turn the light on"))

    def test_apostrophe_normalization_matches_across_sides(self):
        self.assertTrue(is_echo("it's on the desk", "its on the desk"))

    def test_punctuation_does_not_hide_an_echo(self):
        self.assertTrue(is_echo("Turn the light, on.", "turn the light on"))

    def test_spoken_as_list_of_strings(self):
        self.assertTrue(is_echo("turn the light on",
                                ["hello there", "turn the light on"]))

    def test_empty_sides_are_never_echoes(self):
        self.assertFalse(is_echo("", "anything at all"))
        self.assertFalse(is_echo("something here", ""))


class ExactEchoTest(unittest.TestCase):
    def test_identical_text_is_an_echo(self):
        self.assertTrue(is_echo("turn the light on", "turn the light on"))

    def test_unrelated_text_is_not(self):
        self.assertFalse(is_echo("what time is it", "turn the light on"))


if __name__ == "__main__":
    unittest.main()
