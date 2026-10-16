"""Offline tests for ``fetch_html``: status, type, size and bot-check rules.

No network: ``curl_requests.get`` is replaced by a fake streamed response.
"""
import unittest
from unittest import mock

from curl_cffi import requests as curl_requests

from threetoks.web import fetch
from threetoks.web.errors import BlockedPage, FetchError
from threetoks.web.fetch import fetch_html

URL = "https://ex.com/page"
HTML_TYPE = {"Content-Type": "text/html; charset=utf-8"}
CLOUDFLARE_403 = ("<html><head><title>Just a moment...</title></head><body>"
                  "<script>window._cf_chl_opt={cvId: '3'}</script></body></html>")
REDDIT_200 = ("<html><head><title>Reddit - Prove your humanity</title></head>"
              "<body><p>Complete the check to continue.</p></body></html>")
NORMAL_PAGE = "<html><head><title>Ethanol</title></head><body>C2H5OH</body></html>"


class _FakeStream:
    """Minimal stand-in for a streamed curl_cffi ``Response``."""

    def __init__(self, body: str, status_code: int = 200,
                 headers: dict | None = None, chunk_size: int = 64):
        data = body.encode("utf-8")
        self.chunks = [data[i:i + chunk_size]
                       for i in range(0, len(data), chunk_size)]
        self.status_code = status_code
        self.headers = HTML_TYPE if headers is None else headers
        self.charset_encoding = "utf-8"
        self.closed = False

    def iter_content(self):
        return iter(self.chunks)

    def close(self) -> None:
        self.closed = True


def _fetch_with(response, **kwargs) -> str:
    """Run ``fetch_html`` with ``response`` as the network answer."""
    with mock.patch.object(fetch.curl_requests, "get", return_value=response):
        return fetch_html(URL, **kwargs)


class FetchHtmlTest(unittest.TestCase):
    def test_normal_page_returns_html_and_closes(self):
        response = _FakeStream(NORMAL_PAGE)
        self.assertEqual(_fetch_with(response), NORMAL_PAGE)
        self.assertTrue(response.closed)

    def test_cloudflare_403_raises_blocked_page(self):
        with self.assertRaises(BlockedPage) as caught:
            _fetch_with(_FakeStream(CLOUDFLARE_403, status_code=403))
        self.assertIn("cloudflare", str(caught.exception))

    def test_reddit_humanity_check_with_200_raises_blocked_page(self):
        with self.assertRaises(BlockedPage):
            _fetch_with(_FakeStream(REDDIT_200))

    def test_plain_404_raises_fetch_error_not_blocked_page(self):
        with self.assertRaises(FetchError) as caught:
            _fetch_with(_FakeStream("<html>Not found</html>", status_code=404))
        self.assertNotIsInstance(caught.exception, BlockedPage)
        self.assertIn("HTTP 404", str(caught.exception))

    def test_oversize_body_raises_fetch_error(self):
        response = _FakeStream(NORMAL_PAGE * 10)
        with self.assertRaises(FetchError) as caught:
            _fetch_with(response, max_bytes=len(NORMAL_PAGE))
        self.assertNotIsInstance(caught.exception, BlockedPage)
        self.assertIn("exceeds", str(caught.exception))
        self.assertTrue(response.closed)

    def test_non_html_content_type_raises_fetch_error(self):
        pdf = _FakeStream("%PDF-1.7", headers={"Content-Type": "application/pdf"})
        with self.assertRaises(FetchError) as caught:
            _fetch_with(pdf)
        self.assertIn("not HTML", str(caught.exception))

    def test_network_error_becomes_fetch_error(self):
        refused = curl_requests.exceptions.ConnectionError("refused")
        with mock.patch.object(fetch.curl_requests, "get", side_effect=refused):
            with self.assertRaises(FetchError) as caught:
                fetch_html(URL)
        self.assertIn(URL, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
