"""Unit tests for HTML-to-sentences and link cleaning (offline)."""
import unittest
from pathlib import Path

from threetoks.web.textify import (PageLink, _is_link_valid, _is_sentence,
                                   html_to_page)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    """Read a fixture HTML file as text."""
    return (FIXTURES / name).read_text(encoding="utf-8")


class SentenceFilterTest(unittest.TestCase):
    def test_digit_lines_kept_with_word_context(self):
        self.assertTrue(_is_sentence("Error 404 occurred"))
        self.assertTrue(_is_sentence("The tower is 330m tall"))

    def test_bare_digit_fragment_dropped(self):
        self.assertFalse(_is_sentence("330m"))
        self.assertFalse(_is_sentence("320.jpg"))

    def test_short_nav_words_dropped(self):
        for junk in ("Home", "Login", "Read more", "Terms",
                     "Jump to ratings and reviews"):
            self.assertFalse(_is_sentence(junk), junk)

    def test_interrogative_lines_dropped(self):
        self.assertFalse(_is_sentence("What is the capital of Australia?"))
        self.assertFalse(_is_sentence("How mountainous is Japan?"))

    def test_clause_with_names_kept(self):
        self.assertTrue(
            _is_sentence("Gabriel Garcia Marquez, Gregory Rabassa (Translator)"))

    def test_long_sentence_with_punctuation_kept(self):
        self.assertTrue(_is_sentence("The tower stands proudly over Paris."))


class HtmlToPageTest(unittest.TestCase):
    def setUp(self):
        self.page = html_to_page(_load("article.html"), "https://ex.com/eiffel")

    def test_title_extracted(self):
        self.assertEqual(self.page.title, "The Eiffel Tower - Facts")

    def test_nav_junk_filtered_out(self):
        joined = " || ".join(self.page.sentences)
        for junk in ("Home", "About", "Contact", "Login", "Read more",
                     "Privacy", "Terms"):
            self.assertNotIn(f"|| {junk} ||", f"|| {joined} ||")

    def test_real_content_kept_and_numberable(self):
        self.assertTrue(any("wrought-iron lattice tower" in s
                            for s in self.page.sentences))
        self.assertTrue(any("330 metres tall" in s
                            for s in self.page.sentences))
        # Sentences are a list, so their positions are the note indices.
        self.assertEqual(self.page.sentences,
                         list(self.page.sentences))

    def test_max_sentences_caps_output(self):
        capped = html_to_page(_load("article.html"), "https://ex.com/",
                              max_sentences=2)
        self.assertEqual(len(capped.sentences), 2)


class LinkCleaningTest(unittest.TestCase):
    def test_is_link_valid_rejects_assets_and_junk(self):
        self.assertFalse(_is_link_valid("https://ex.com/a.png"))
        self.assertFalse(_is_link_valid("mailto:hi@ex.com"))
        self.assertFalse(_is_link_valid("https://ex.com/page/42"))
        self.assertFalse(_is_link_valid("https://ex.com/" + "x" * 80))
        self.assertTrue(_is_link_valid("https://ex.com/article"))

    def test_links_absolutized_deduped_and_filtered(self):
        page = html_to_page(_load("article.html"), "https://ex.com/eiffel")
        urls = [link.url for link in page.links]
        self.assertIn("https://history.example.com/eiffel-history", urls)
        self.assertIn("https://ex.com/about", urls)
        self.assertNotIn("https://ex.com/gallery/photo.jpg", urls)
        self.assertNotIn("mailto:hi@example.com", urls)
        self.assertEqual(len(urls), len(set(urls)), "links must be unique")

    def test_label_trimmed_to_limit(self):
        page = html_to_page(
            '<a href="https://ex.com/x">' + "word " * 40 + "</a>",
            "https://ex.com/")
        self.assertLessEqual(len(page.links[0].label), 60)

    def test_max_links_caps_output(self):
        page = html_to_page(_load("article.html"), "https://ex.com/",
                            max_links=1)
        self.assertEqual(len(page.links), 1)
        self.assertIsInstance(page.links[0], PageLink)


if __name__ == "__main__":
    unittest.main()


class JunkDenyListTest(unittest.TestCase):
    def test_forum_chrome_dropped(self):
        for junk in ("Please sign in to post your reply now.",
                     "09/23/16 07:41 PM",
                     "Subscribe to our newsletter for updates today."):
            self.assertFalse(_is_sentence(junk), junk)

    def test_real_content_with_dates_kept(self):
        self.assertTrue(_is_sentence(
            "The restaurant opened in March 2015 near the old port."))

    def test_github_notification_prompt_dropped(self):
        self.assertFalse(_is_sentence(
            "You must be signed in to change notification settings"))


class BoilerplateFilterTest(unittest.TestCase):
    def test_copyright_footers_dropped(self):
        for junk in ("© 2026 Google LLC",
                     "© 2026 GitHub, Inc.",
                     "Copyright 2025 Example Corp and its affiliates."):
            self.assertFalse(_is_sentence(junk), junk)

    def test_shell_command_lines_dropped(self):
        for junk in ("$ curl -fsSL https://openclaw.ai/install.sh | bash",
                     "wget -qO- https://get.example.sh | sh",
                     "Run curl -X POST to send the request now."):
            self.assertFalse(_is_sentence(junk), junk)

    def test_prose_mentioning_money_kept(self):
        self.assertTrue(_is_sentence(
            "The project raised $5 million in funding last year."))


class TablePipeTest(unittest.TestCase):
    def test_pipe_rows_cleaned_not_kept_raw(self):
        from threetoks.web.textify import _clean_line
        cleaned = _clean_line("| Cite as: | arXiv:2510.02669 [cs.AI] |")
        self.assertNotIn("|", cleaned)
        self.assertIn("Cite as:", cleaned)
