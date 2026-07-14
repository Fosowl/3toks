"""Offline tests for the web vertical's final-answer path.

(Separate from test_web_vertical.py, which currently fails to import on
this branch; these tests cover only the goal-aware answer fallback.)
"""
import unittest

from threetoks.nodes import Decision
from threetoks.web.notes import NoteStore
from threetoks.web.vertical import WebResearchVertical

FACT = "The Eiffel Tower is 330 meters tall."


class _NoResults:
    def search(self, query, max_results=8):
        return []


def _text(value):
    return Decision("short_text", value, value, valid=True)


def _make_vertical(note_texts, task="how tall is the tower?"):
    notes = NoteStore()
    for index, text in enumerate(note_texts, 1):
        notes.add(text, "https://example.com/", index)
    return WebResearchVertical(task, _NoResults(), None, notes)


class QuoteFallbackTest(unittest.TestCase):
    def test_fallback_quotes_task_closest_notes_first(self):
        vertical = _make_vertical(["Paris is in France.", FACT])
        self.assertTrue(vertical._quote_fallback().startswith(FACT))

    def test_fallback_without_notes_is_no_answer(self):
        vertical = _make_vertical([])
        self.assertEqual(vertical._quote_fallback(), "(no answer)")

    def test_double_bleed_answer_is_goal_aware(self):
        vertical = _make_vertical(["Paris is in France.", FACT])
        node = vertical._synthesis_node()
        vertical.apply(node, _text("3"))                    # bleed once
        vertical.apply(vertical.pending, _text("7"))        # bleed again
        self.assertTrue(vertical.answer.startswith(FACT), vertical.answer)


if __name__ == "__main__":
    unittest.main()
