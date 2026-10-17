"""Tests for the config field registry: coverage, parsing, stepping."""
import unittest

from threetoks import fields
from threetoks.config import ThreetoksConfig
from threetoks.fields import (changed_fields, cycle_value, find_field,
                              parse_value, set_value)


class CoverageTest(unittest.TestCase):
    def test_every_dataclass_field_has_a_registry_row(self):
        # A new config knob without a row would be invisible in /config.
        self.assertEqual(fields.covered_names(), fields.dataclass_names())

    def test_every_section_has_rows(self):
        for section in fields.SECTIONS:
            self.assertTrue(fields.fields_of(section.name), section.name)


class ParseTest(unittest.TestCase):
    def test_choice_is_case_insensitive_and_closed(self):
        mode = find_field("browser.mode")
        self.assertEqual(parse_value(mode, " PLAIN "), ("plain", None))
        self.assertIsNone(parse_value(mode, "stealth")[0])

    def test_bounds_and_types(self):
        rounds = find_field("research.max_rounds")
        self.assertEqual(parse_value(rounds, "0")[1], "at least 1")
        self.assertEqual(parse_value(rounds, "x")[1], "not a int")
        self.assertEqual(parse_value(rounds, "7"), (7, None))
        speaker = find_field("voice.tts_speaker")
        self.assertEqual(parse_value(speaker, "-1"), (-1, None))
        self.assertEqual(parse_value(find_field("search.min_interval_s"),
                                     "2.5"), (2.5, None))

    def test_bool_words(self):
        flag = find_field("llm.send_seed")
        self.assertEqual(parse_value(flag, "Off"), (False, None))
        self.assertEqual(parse_value(flag, "yes"), (True, None))
        self.assertIsNone(parse_value(flag, "maybe")[0])

    def test_text_cannot_blank_a_field_that_has_a_default(self):
        self.assertEqual(parse_value(find_field("llm.model"), "")[1],
                         "cannot be empty")
        self.assertEqual(parse_value(find_field("llm.api_base"), ""),
                         ("", None))  # default is empty: blank is fine


class StepTest(unittest.TestCase):
    def test_choice_wraps_both_ways(self):
        provider = find_field("llm.provider")
        last = provider.choices[-1]
        self.assertEqual(cycle_value(provider, last, 1), provider.choices[0])
        self.assertEqual(cycle_value(provider, provider.choices[0], -1), last)

    def test_numbers_stop_at_their_minimum(self):
        rounds = find_field("research.max_rounds")
        self.assertEqual(cycle_value(rounds, 1, -1), 1)
        self.assertEqual(cycle_value(rounds, 1, 1), 2)
        gap = find_field("search.min_interval_s")
        self.assertEqual(cycle_value(gap, 0.0, -1), 0.0)
        self.assertEqual(cycle_value(gap, 5.0, 1), 5.5)

    def test_text_does_not_move(self):
        self.assertEqual(cycle_value(find_field("llm.model"), "m", 1), "m")


class ChangeTest(unittest.TestCase):
    def test_set_value_is_a_copy_and_changed_fields_sees_it(self):
        base = ThreetoksConfig()
        item = find_field("voice.lang")
        edited = set_value(base, item, "fr")
        self.assertEqual(base.voice.lang, "en-us")
        self.assertEqual(changed_fields(base, edited), (item,))


if __name__ == "__main__":
    unittest.main()
