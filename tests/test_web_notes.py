"""Unit tests for the provenance-tracked note store (offline)."""
import unittest

from threetoks.web.notes import Note, NoteStore, rank_by_overlap

WIKI = "https://en.wikipedia.org/wiki/Paris"


class NoteStoreTest(unittest.TestCase):
    def setUp(self):
        self.store = NoteStore()

    def test_add_and_count(self):
        self.store.add("Paris is the capital.", WIKI, 3)
        self.store.add("It has 2.1 million people.", WIKI, 7)
        self.assertEqual(self.store.count(), 2)

    def test_exact_duplicate_text_skipped(self):
        self.store.add("Same quote.", WIKI, 1)
        self.store.add("Same quote.", "https://other.com/x", 9)
        self.assertEqual(self.store.count(), 1)

    def test_near_duplicate_text_kept(self):
        self.store.add("Same quote.", WIKI, 1)
        self.store.add("Same quote", WIKI, 2)  # no period -> distinct
        self.assertEqual(self.store.count(), 2)

    def test_shared_prefix_near_duplicate_dropped(self):
        head = "The capital city of the French Republic is the city of Paris"
        self.store.add(head + ", a hub.", WIKI, 1)
        self.store.add(head + ", and big.", WIKI, 2)  # same first 60 chars
        self.assertEqual(self.store.count(), 1)

    def test_select_preserves_pick_order_and_provenance(self):
        for idx in range(4):
            self.store.add(f"quote number {idx}", WIKI, idx * 10)
        ranked = self.store.select([3, 1])
        texts = [note.text for note in ranked.entries()]
        self.assertEqual(texts, ["quote number 2", "quote number 0"])
        self.assertEqual(ranked.entries()[0].sentence_idx, 20)  # provenance

    def test_select_skips_out_of_range_indices(self):
        self.store.add("only note", WIKI, 1)
        self.assertEqual(self.store.select([1, 9]).count(), 1)

    def test_render_numbers_and_tags_host(self):
        self.store.add("Paris is the capital.", WIKI, 3)
        self.store.add("It has 2.1 million people.", WIKI, 7)
        rendered = self.store.render()
        self.assertEqual(
            rendered.splitlines()[0],
            "[1] Paris is the capital. (source: en.wikipedia.org)")
        self.assertIn("[2] It has 2.1 million people.", rendered)

    def test_render_caps_at_max_notes(self):
        for idx in range(5):
            self.store.add(f"quote {idx}", WIKI, idx)
        rendered = self.store.render(max_notes=2)
        self.assertEqual(len(rendered.splitlines()), 2)

    def test_notes_are_frozen_dataclasses(self):
        self.store.add("q", WIKI, 0)
        note = Note("q", WIKI, 0)
        with self.assertRaises(Exception):
            note.text = "mutated"


class RankByOverlapTest(unittest.TestCase):
    def test_task_closest_text_ranks_first(self):
        texts = ["Paris is in France.",
                 "The Eiffel Tower is 330 meters tall."]
        ranked = rank_by_overlap(texts, "how tall is the tower?", 2)
        self.assertEqual(ranked[0], "The Eiffel Tower is 330 meters tall.")

    def test_top_k_caps_the_result(self):
        texts = ["tower one", "tower two", "tower three"]
        self.assertEqual(len(rank_by_overlap(texts, "tower?", 2)), 2)

    def test_no_keywords_keeps_original_head(self):
        texts = ["first", "second"]
        self.assertEqual(rank_by_overlap(texts, "a b?", 1), ["first"])

    def test_ties_preserve_original_order(self):
        texts = ["alpha note", "beta note"]
        self.assertEqual(rank_by_overlap(texts, "note?", 2), texts)


if __name__ == "__main__":
    unittest.main()
