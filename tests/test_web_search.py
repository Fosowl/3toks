"""Unit tests for search-result parsing and provider auto-detect (offline).

No network: parse helpers run on fixtures, and provider ``search`` methods
are exercised by monkeypatching ``requests``.
"""
import base64
import unittest
from pathlib import Path
from unittest import mock

from threetoks.web import search
from threetoks.web.search import (BingHtmlProvider, DdgHtmlProvider,
                                 FallbackProvider, SearchResult,
                                 SearchUnavailable, SearxngProvider,
                                 _decode_bing_href, _decode_ddg_href,
                                 _parse_bing_results, _parse_ddg_results,
                                 _parse_searxng_results, auto_provider)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    """Read a fixture HTML file as text."""
    return (FIXTURES / name).read_text(encoding="utf-8")


class _FakeResponse:
    """Minimal stand-in for a ``requests.Response``."""

    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """No-op; fixtures are always 200."""


class SearxngParseTest(unittest.TestCase):
    def test_parses_result_blocks(self):
        results = _parse_searxng_results(_load("searxng_results.html"), 8)
        self.assertEqual(results[0], SearchResult(
            "Paris - Wikipedia", "https://en.wikipedia.org/wiki/Paris",
            "Paris is the capital and most populous city of France."))

    def test_skips_blocks_without_url_header(self):
        results = _parse_searxng_results(_load("searxng_results.html"), 8)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.url.startswith("http") for r in results))

    def test_missing_snippet_is_empty(self):
        results = _parse_searxng_results(_load("searxng_results.html"), 8)
        self.assertEqual(results[2].snippet, "")

    def test_max_results_caps(self):
        results = _parse_searxng_results(_load("searxng_results.html"), 1)
        self.assertEqual(len(results), 1)

    def test_provider_posts_and_parses(self):
        provider = SearxngProvider("http://localhost:8080")
        fake = _FakeResponse(_load("searxng_results.html"))
        with mock.patch.object(search.requests, "post",
                               return_value=fake) as posted:
            results = provider.search("paris")
        self.assertEqual(results[0].title, "Paris - Wikipedia")
        args, kwargs = posted.call_args
        self.assertEqual(args[0], "http://localhost:8080/search")
        self.assertEqual(kwargs["data"]["categories"], "general")


class DdgParseTest(unittest.TestCase):
    def test_decodes_uddg_redirect(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fa&rut=z"
        self.assertEqual(_decode_ddg_href(href), "https://ex.com/a")

    def test_parses_title_url_and_snippet(self):
        results = _parse_ddg_results(_load("ddg_results.html"), 8)
        self.assertEqual(results[0], SearchResult(
            "Paris - Wikipedia", "https://en.wikipedia.org/wiki/Paris",
            "Paris is the capital and most populous city of France."))
        self.assertEqual(results[1].url, "https://www.britannica.com/place/Paris")

    def test_provider_gets_and_parses(self):
        provider = DdgHtmlProvider()
        fake = _FakeResponse(_load("ddg_results.html"))
        with mock.patch.object(search.requests, "get", return_value=fake):
            results = provider.search("paris")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Paris - Wikipedia")


class AutoProviderTest(unittest.TestCase):
    def test_picks_searxng_when_probe_succeeds(self):
        with mock.patch.object(search, "_searxng_answers", return_value=True):
            provider = auto_provider()
            self.assertIsInstance(provider, search.FallbackProvider)
            self.assertIsInstance(provider.providers[0], SearxngProvider)
            self.assertEqual(len(provider.providers), 3)

    def test_falls_back_to_ddg_when_probe_fails(self):
        with mock.patch.object(search, "_searxng_answers", return_value=False):
            provider = auto_provider()
            self.assertIsInstance(provider, search.FallbackProvider)
            self.assertIsInstance(provider.providers[0],
                                  search.BingHtmlProvider)
            self.assertIsInstance(provider.providers[1], DdgHtmlProvider)


def _wrap_bing(url: str) -> str:
    """A bing.com/ck/a redirect href for ``url``, unpadded like the real ones."""
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return f"https://www.bing.com/ck/a?!&&p=hash&u=a1{encoded}&ntb=1"


class BingDecodeTest(unittest.TestCase):
    def test_unwraps_ck_redirect(self):
        self.assertEqual(_decode_bing_href(_wrap_bing("https://ex.com/page")),
                         "https://ex.com/page")

    def test_direct_urls_pass_through(self):
        self.assertEqual(_decode_bing_href("https://ex.com/a"),
                         "https://ex.com/a")

    def test_undecodable_payload_passes_through(self):
        href = "https://www.bing.com/ck/a?u=a1x&ntb=1"
        self.assertEqual(_decode_bing_href(href), href)

    def test_parse_returns_unwrapped_urls(self):
        html = ('<li class="b_algo"><h2><a href="'
                + _wrap_bing("https://ex.com/c")
                + '">Title C</a></h2><p>snippet c</p></li>')
        results = _parse_bing_results(html, 8)
        self.assertEqual(results[0], SearchResult(
            "Title C", "https://ex.com/c", "snippet c"))


class ProviderChallengeTest(unittest.TestCase):
    """Zero results plus a bot-wall marker must raise, not return []."""

    def test_searxng_suspension_raises_unavailable(self):
        suspended = ('<html><div class="dialog-error">'
                     'Suspended: too many requests</div></html>')
        provider = SearxngProvider("http://localhost:8080")
        with mock.patch.object(search.requests, "post",
                               return_value=_FakeResponse(suspended)):
            with self.assertRaises(SearchUnavailable):
                provider.search("q")

    def test_bing_challenge_raises_unavailable(self):
        shell = ('<html><script>"verifyEndpoint":'
                 '"https://www.bing.com/challenge/verify?p=7"</script></html>')
        with mock.patch.object(search.requests, "get",
                               return_value=_FakeResponse(shell)):
            with self.assertRaises(SearchUnavailable):
                BingHtmlProvider().search("q")

    def test_ddg_challenge_raises_unavailable(self):
        shell = ('<div class="anomaly-modal__title">Unfortunately, '
                 'bots use DuckDuckGo too.</div>')
        with mock.patch.object(search.requests, "get",
                               return_value=_FakeResponse(shell)):
            with self.assertRaises(SearchUnavailable):
                DdgHtmlProvider().search("q")

    def test_genuine_empty_page_still_returns_empty(self):
        with mock.patch.object(search.requests, "get",
                               return_value=_FakeResponse("<html>0 hits</html>")):
            self.assertEqual(BingHtmlProvider().search("q"), [])


class _DeadHop:
    def search(self, query, max_results=8):
        raise SearchUnavailable("blocked")


class _EmptyHop:
    def search(self, query, max_results=8):
        return []


class _HitsHop:
    def search(self, query, max_results=8):
        return [SearchResult("T", "http://t", "s")]


def _unpaced(provider):
    """Disable the 5s pacing sleep for offline tests."""
    provider._pace = lambda: None
    return provider


class FallbackHealthTest(unittest.TestCase):
    def test_all_hops_failing_counts_consecutive_failures(self):
        provider = _unpaced(FallbackProvider(_DeadHop(), _DeadHop()))
        self.assertEqual(provider.search("a"), [])
        self.assertEqual(provider.search("b"), [])
        self.assertTrue(provider.last_call_failed)
        self.assertEqual(provider.consecutive_failures, 2)

    def test_alive_empty_hop_resets_failures(self):
        provider = _unpaced(FallbackProvider(_DeadHop(), _EmptyHop()))
        provider.consecutive_failures = 3
        self.assertEqual(provider.search("a"), [])
        self.assertFalse(provider.last_call_failed)
        self.assertEqual(provider.consecutive_failures, 0)

    def test_later_hop_results_still_served_after_dead_hop(self):
        provider = _unpaced(FallbackProvider(_DeadHop(), _HitsHop()))
        results = provider.search("a")
        self.assertEqual(results[0].title, "T")
        self.assertFalse(provider.last_call_failed)
        self.assertEqual(provider.consecutive_failures, 0)

    def test_cache_hit_clears_the_stale_failure_flag(self):
        provider = _unpaced(FallbackProvider(_HitsHop()))
        provider.search("a")  # cached
        provider.last_call_failed = True  # stale from another query
        provider.search("a")  # served from cache = not a failure
        self.assertFalse(provider.last_call_failed)


if __name__ == "__main__":
    unittest.main()
