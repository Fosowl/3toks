"""Offline tests for bot-check page detection in ``blockpage``."""
import unittest

from threetoks.web.blockpage import block_reason, page_title


def _page(title: str, body: str = "<p>text</p>") -> str:
    """Build a small HTML page with ``title`` and ``body``."""
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


class BodyMarkerTest(unittest.TestCase):
    def test_cloudflare_challenge_script(self):
        page = _page("Example", "<script>window._cf_chl_opt={}</script>")
        self.assertEqual(block_reason(page), "cloudflare challenge")

    def test_datadome_captcha_frame(self):
        page = _page("x", "<iframe src='https://geo.captcha-delivery.com/c'>")
        self.assertEqual(block_reason(page), "datadome captcha")

    def test_perimeterx_captcha_div(self):
        page = _page("x", "<div id='px-captcha'></div>")
        self.assertEqual(block_reason(page), "perimeterx captcha")


class TitleMarkerTest(unittest.TestCase):
    def test_each_challenge_title_is_blocked(self):
        for title in ("Just a moment...",
                      "Attention Required! | Cloudflare",
                      "Reddit - Prove your humanity",
                      "Access to this page has been denied"):
            with self.subTest(title=title):
                reason = block_reason(_page(title))
                self.assertIn(title.lower(), reason)


class NormalPageTest(unittest.TestCase):
    def test_challenge_platform_script_is_not_a_block(self):
        page = _page("Ethanol - Wikipedia",
                     "<script src='/cdn-cgi/challenge-platform/x.js'></script>")
        self.assertIsNone(block_reason(page))

    def test_recaptcha_widget_is_not_a_block(self):
        page = _page("Sign up", "<div class='g-recaptcha'></div>")
        self.assertIsNone(block_reason(page))

    def test_word_captcha_is_not_a_block(self):
        page = _page("What is a CAPTCHA?", "<p>A captcha tests humans.</p>")
        self.assertIsNone(block_reason(page))


class PageTitleTest(unittest.TestCase):
    def test_missing_title_is_empty(self):
        self.assertEqual(page_title("<html><p>no title</p></html>"), "")

    def test_unclosed_title_is_empty(self):
        self.assertEqual(page_title("<html><title>never closed"), "")

    def test_title_with_attributes_is_read(self):
        self.assertEqual(page_title("<title lang='en'> hello </title>"), "hello")


if __name__ == "__main__":
    unittest.main()
