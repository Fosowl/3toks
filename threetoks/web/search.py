"""Web search providers.

A ``SearchProvider`` turns a query into a short list of ``SearchResult``s.
Two backends: ``SearxngProvider`` (local docker, HTML scrape of
``article.result`` blocks) and ``DdgHtmlProvider`` (zero-infrastructure
fallback that scrapes the DuckDuckGo HTML endpoint). ``auto_provider``
picks searxng when it answers, else ddg.

HTML parsing lives in module-level ``_parse_*`` helpers so tests can feed
fixtures without any network access.
"""
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlparse

import time

import requests
from bs4 import BeautifulSoup

DEFAULT_MAX_RESULTS = 8
DEFAULT_SEARXNG_URL = "http://localhost:8080"
SEARXNG_URL_ENV = "THREETOKS_SEARXNG_URL"
PROBE_TIMEOUT_S = 2
SEARCH_TIMEOUT_S = 10
DDG_HTML_URL = "https://html.duckduckgo.com/html/"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class SearchResult:
    """One search hit: a title, its URL, and a short snippet."""

    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    """Anything that turns a query into a list of ``SearchResult``s."""

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS
               ) -> list[SearchResult]:
        """Return up to ``max_results`` hits for ``query``."""


def _parse_searxng_results(html: str, max_results: int) -> list[SearchResult]:
    """Parse a searxng HTML results page into ``SearchResult``s."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for article in soup.find_all("article", class_="result"):
        header = article.find("a", class_="url_header")
        if not header or not header.get("href"):
            continue
        title = article.find("h3")
        snippet = article.find("p", class_="content")
        results.append(SearchResult(
            title=title.get_text(" ", strip=True) if title else "",
            url=header["href"],
            snippet=snippet.get_text(" ", strip=True) if snippet else "",
        ))
        if len(results) >= max_results:
            break
    return results


class SearxngProvider:
    """Search a local SearXNG instance via its HTML results page."""

    def __init__(self, base_url: str = DEFAULT_SEARXNG_URL):
        """Store the instance base URL (no trailing slash needed)."""
        self.base_url = base_url.rstrip("/")
        self._rotation = 0

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS
               ) -> list[SearchResult]:
        """POST to ``/search``, rotating engine subsets between calls.

        SearXNG fans out to every enabled engine on each query, which
        gets them all rate-limited together under automated load; asking
        a different small subset per call spreads the traffic instead.
        """
        engines = ENGINE_ROTATION[self._rotation % len(ENGINE_ROTATION)]
        self._rotation += 1
        response = requests.post(
            f"{self.base_url}/search",
            data={"q": query, "categories": "general", "language": "auto",
                  "engines": ",".join(engines)},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=SEARCH_TIMEOUT_S,
        )
        response.raise_for_status()
        return _parse_searxng_results(response.text, max_results)


def _decode_ddg_href(href: str) -> str:
    """Turn a ``//duckduckgo.com/l/?uddg=...`` redirect into the real URL."""
    query = urlparse(href).query
    target = parse_qs(query).get("uddg")
    if target:
        return unquote(target[0])
    return href


def _parse_ddg_results(html: str, max_results: int) -> list[SearchResult]:
    """Parse a DuckDuckGo HTML results page into ``SearchResult``s."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for anchor in soup.find_all("a", class_="result__a"):
        href = anchor.get("href")
        if not href:
            continue
        snippet = _nearest_snippet(anchor)
        results.append(SearchResult(
            title=anchor.get_text(" ", strip=True),
            url=_decode_ddg_href(href),
            snippet=snippet,
        ))
        if len(results) >= max_results:
            break
    return results


def _nearest_snippet(anchor) -> str:
    """Find the ``result__snippet`` text belonging to a result anchor."""
    container = anchor.find_parent(class_="result")
    scope = container if container else anchor.parent
    snippet = scope.find(class_="result__snippet") if scope else None
    return snippet.get_text(" ", strip=True) if snippet else ""


class DdgHtmlProvider:
    """Search DuckDuckGo's no-JavaScript HTML endpoint."""

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS
               ) -> list[SearchResult]:
        """GET the DDG HTML page for ``query`` and parse its results."""
        response = requests.get(
            DDG_HTML_URL,
            params={"q": query},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=SEARCH_TIMEOUT_S,
        )
        response.raise_for_status()
        return _parse_ddg_results(response.text, max_results)


def _searxng_answers(base_url: str) -> bool:
    """Return True if a cheap GET of the searxng root succeeds quickly."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/",
                                timeout=PROBE_TIMEOUT_S)
        return response.status_code == 200
    except requests.RequestException:
        return False


MIN_SEARCH_INTERVAL_S = 5.0
SEARCH_CACHE_MAX = 200

ENGINE_ROTATION = (
    ("brave", "wikipedia"),
    ("google", "startpage"),
    ("brave", "google"),
    ("wikipedia", "startpage"),
    ("qwant", "brave"),
    ("duckduckgo", "qwant"),
)


class BingHtmlProvider:
    """Scrape bing.com/search — historically tolerant of plain HTTP."""

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS
               ) -> list[SearchResult]:
        """GET the Bing SERP and parse organic results."""
        response = requests.get(
            "https://www.bing.com/search", params={"q": query},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=SEARCH_TIMEOUT_S)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SearchResult] = []
        for item in soup.select("li.b_algo")[:max_results]:
            link = item.find("h2")
            anchor = link.find("a") if link else None
            if not anchor or not anchor.get("href"):
                continue
            snippet = item.find("p")
            results.append(SearchResult(
                title=anchor.get_text(" ", strip=True),
                url=anchor["href"],
                snippet=snippet.get_text(" ", strip=True) if snippet else ""))
        return results


class FallbackProvider:
    """Search the primary provider; fall back on error OR zero results."""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary
        self._cache: dict[str, list] = {}
        self._last_search = 0.0

    def search(self, query: str, max_results: int = 8):
        """Cached, paced search that degrades to [] instead of raising."""
        if query in self._cache:
            return self._cache[query]
        self._pace()
        results = self._try(self.primary, query, max_results) \
            or self._try(self.secondary, query, max_results)
        if results:  # empty = transient (rate limit); retry next time
            if len(self._cache) >= SEARCH_CACHE_MAX:
                self._cache.clear()
            self._cache[query] = results
        return results

    def _try(self, provider, query: str, max_results: int):
        """One provider attempt; any failure means empty results."""
        try:
            return provider.search(query, max_results)
        except Exception:
            return []

    def _pace(self) -> None:
        """Keep at least MIN_SEARCH_INTERVAL_S between outbound searches."""
        elapsed = time.monotonic() - self._last_search
        if elapsed < MIN_SEARCH_INTERVAL_S:
            time.sleep(MIN_SEARCH_INTERVAL_S - elapsed)
        self._last_search = time.monotonic()


def auto_provider() -> SearchProvider:
    """Prefer a reachable local searxng, else fall back to DDG HTML."""
    base_url = os.getenv(SEARXNG_URL_ENV, DEFAULT_SEARXNG_URL)
    if _searxng_answers(base_url):
        return FallbackProvider(SearxngProvider(base_url),
                                FallbackProvider(BingHtmlProvider(),
                                                 DdgHtmlProvider()))
    return FallbackProvider(BingHtmlProvider(), DdgHtmlProvider())


if __name__ == "__main__":
    _SEARX = """<article class="result">
      <a class="url_header" href="https://ex.com/a">x</a>
      <h3>Alpha Title</h3><p class="content">alpha snippet</p></article>"""
    parsed = _parse_searxng_results(_SEARX, DEFAULT_MAX_RESULTS)
    assert parsed == [SearchResult("Alpha Title", "https://ex.com/a",
                                   "alpha snippet")], parsed
    _DDG = ('<div class="result"><a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fb">B</a>'
            '<a class="result__snippet">bee snippet</a></div>')
    parsed = _parse_ddg_results(_DDG, DEFAULT_MAX_RESULTS)
    assert parsed[0].url == "https://ex.com/b", parsed
    assert parsed[0].snippet == "bee snippet", parsed
    print("smoke OK")
