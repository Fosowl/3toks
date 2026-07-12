"""Unit tests for the append-only episode renderer."""
import unittest

from threetoks.render import Episode


class EpisodeTest(unittest.TestCase):
    def setUp(self):
        self.episode = Episode("SYSTEM prefix", "find the answer")

    def test_session_appends_preserve_prefix(self):
        self.episode.open_observation("[1] a sentence")
        before = self.episode.render_base()
        self.episode.log_session("> noted s1")
        after = self.episode.render_base()
        self.assertTrue(after.startswith(before))

    def test_observation_replacement_folds_session_into_history(self):
        self.episode.open_observation("[1] page one text")
        self.episode.log_session("> noted s1")
        self.episode.open_observation("[1] page two", "> left page one")
        rendered = self.episode.render_base()
        self.assertIn("> left page one", rendered)
        self.assertIn("> noted s1", rendered)
        self.assertNotIn("page one text", rendered)
        self.assertIn("page two", rendered)

    def test_history_precedes_observation(self):
        self.episode.open_observation("OBSERVATION text")
        rendered = self.episode.render_base()
        self.assertLess(rendered.index("SYSTEM prefix"),
                        rendered.index("OBSERVATION text"))

    def test_task_always_present(self):
        self.assertIn("TASK: find the answer", self.episode.render_base())


if __name__ == "__main__":
    unittest.main()
