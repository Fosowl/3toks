"""Web search providers.

A ``SearchProvider`` turns a query into a short list of ``SearchResult``s.
Backends: ``SearxngProvider`` (local docker, HTML scrape of
``article.result`` blocks), ``BingHtmlProvider`` and ``DdgHtmlProvider``
(zero-infrastructure HTML scrapes). A bot-wall/challenge page raises
``SearchUnavailable`` instead of returning [] — an empty answer and a
dead engine must not look alike, or the research loop cannot tell
"no hits" from "search layer down". ``auto_provider`` chains the
reachable backends inside a ``FallbackProvider`` that paces calls, caches
hits, and tracks ``last_call_failed`` / ``consecutive_failures`` so the
caller can see the whole layer go dark.

HTML parsing lives in module-level ``_parse_*`` helpers so tests can feed
fixtures without any network access.
"""
import base64
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

# Bot-wall markers: the page answered 200 but carries a challenge instead
# of results. Each provider checks its own signature before parsing.
SEARXNG_SUSPENDED_MARKERS = ("dialog-error", "Suspended")
BING_CHALLENGE_MARKER = "bing.com/challenge/verify"
DDG_ANOMALY_MARKER = "anomaly-modal"


class SearchUnavailable(Exception):
    """The engine served a bot wall / challenge instead of results."""


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
        A "Suspended" dialog-error means the instance itself is walled:
        raise so the fallback chain moves on.
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
        if all(marker in response.text
               for marker in SEARXNG_SUSPENDED_MARKERS):
            raise SearchUnavailable("searxng suspended the client")
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
        """GET the DDG HTML page for ``query`` and parse its results.

        The anomaly modal is a bot check, not an empty SERP: raise so the
        failure is visible to the fallback chain.
        """
        response = requests.get(
            DDG_HTML_URL,
            params={"q": query},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=SEARCH_TIMEOUT_S,
        )
        response.raise_for_status()
        if DDG_ANOMALY_MARKER in response.text:
            raise SearchUnavailable("duckduckgo served a bot check")
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


def _decode_bing_href(href: str) -> str:
    """Unwrap a ``bing.com/ck/a`` redirect; anything else passes through.

    Bing encodes the target URL as unpadded urlsafe base64 in the ``u``
    parameter, prefixed with ``a1``. Undecodable payloads (or links that
    were never wrapped) come back unchanged.
    """
    payload = parse_qs(urlparse(href).query).get("u", [""])[0]
    if not payload.startswith("a1"):
        return href
    encoded = payload[2:]
    encoded += "=" * (-len(encoded) % 4)
    try:
        return base64.urlsafe_b64decode(encoded).decode("utf-8")
    except Exception:
        return href


def _parse_bing_results(html: str, max_results: int) -> list[SearchResult]:
    """Parse a Bing SERP into ``SearchResult``s, unwrapping ck/a hrefs."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for item in soup.select("li.b_algo")[:max_results]:
        link = item.find("h2")
        anchor = link.find("a") if link else None
        if not anchor or not anchor.get("href"):
            continue
        snippet = item.find("p")
        results.append(SearchResult(
            title=anchor.get_text(" ", strip=True),
            url=_decode_bing_href(anchor["href"]),
            snippet=snippet.get_text(" ", strip=True) if snippet else ""))
    return results


class BingHtmlProvider:
    """Scrape bing.com/search — historically tolerant of plain HTTP."""

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS
               ) -> list[SearchResult]:
        """GET the Bing SERP and parse organic results.

        A challenge/verify shell is a bot wall, not an empty SERP: raise
        so the fallback chain treats the hop as down.
        """
        response = requests.get(
            "https://www.bing.com/search", params={"q": query},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=SEARCH_TIMEOUT_S)
        response.raise_for_status()
        if BING_CHALLENGE_MARKER in response.text:
            raise SearchUnavailable("bing served a challenge page")
        return _parse_bing_results(response.text, max_results)


def _parse_mojeek_results(html: str, max_results: int) -> list[SearchResult]:
    """Parse a Mojeek HTML results page (a.title anchors, li>p.s snippet)."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for anchor in soup.select("a.title"):
        href = anchor.get("href")
        if not href:
            continue
        li = anchor.find_parent("li")
        snippet_tag = li.find("p", class_="s") if li else None
        results.append(SearchResult(
            title=anchor.get_text(" ", strip=True),
            url=href,
            snippet=snippet_tag.get_text(" ", strip=True)
            if snippet_tag else ""))
        if len(results) >= max_results:
            break
    return results


class MojeekHtmlProvider:
    """Scrape mojeek.com/search — plain HTML, no JS challenge.

    E9 measured this live: while Bing and both DuckDuckGo HTML endpoints
    were bot-walled from a residential connection, Mojeek answered plain
    requests and carried most of the retrieval benchmark. It rate-limits
    too (403 after a fast burst, recovering in ~45s), so it stays
    available standalone even though the auto chain leads with Bing.
    """

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS
               ) -> list[SearchResult]:
        """GET the Mojeek SERP and parse organic results."""
        response = requests.get(
            "https://www.mojeek.com/search", params={"q": query},
            headers={"User-Agent": BROWSER_USER_AGENT},
            timeout=SEARCH_TIMEOUT_S)
        response.raise_for_status()
        return _parse_mojeek_results(response.text, max_results)


class FallbackProvider:
    """Try each provider in turn; a dead hop falls through to the next.

    Health tracking: a call where every hop raised (``SearchUnavailable``
    or a transport error) sets ``last_call_failed`` and increments
    ``consecutive_failures`` — the research loop reads those to abort
    honestly instead of requerying a dead layer. A hop that answered at
    all (even with zero results, a real "no hits") resets both, and a
    cache hit clears the stale flag.
    """

    def __init__(self, *providers):
        self.providers: list = []
        for provider in providers:  # flatten nested chains into one list
            if isinstance(provider, FallbackProvider):
                self.providers.extend(provider.providers)
            else:
                self.providers.append(provider)
        self.last_call_failed = False
        self.consecutive_failures = 0
        self._cache: dict[str, list] = {}
        self._last_search = 0.0

    def search(self, query: str, max_results: int = 8):
        """Cached, paced search that degrades to [] instead of raising."""
        if query in self._cache:
            self.last_call_failed = False
            return self._cache[query]
        self._pace()
        answered = False
        results: list = []
        for provider in self.providers:
            try:
                results = provider.search(query, max_results)
                answered = True  # the hop is alive, even at zero hits
            except Exception:
                continue
            if results:
                break
        if answered:
            self.last_call_failed = False
            self.consecutive_failures = 0
        else:
            self.last_call_failed = True
            self.consecutive_failures += 1
        if results:  # empty = transient (rate limit); retry next time
            if len(self._cache) >= SEARCH_CACHE_MAX:
                self._cache.clear()
            self._cache[query] = results
        return results

    def _pace(self) -> None:
        """Keep at least MIN_SEARCH_INTERVAL_S between outbound searches."""
        elapsed = time.monotonic() - self._last_search
        if elapsed < MIN_SEARCH_INTERVAL_S:
            time.sleep(MIN_SEARCH_INTERVAL_S - elapsed)
        self._last_search = time.monotonic()


def auto_provider() -> SearchProvider:
    """Prefer a reachable local searxng, then Bing, then DDG.

    The chain is one flat FallbackProvider exposing ``.providers`` so
    callers (and tests) can see which hops are live.
    """
    base_url = os.getenv(SEARXNG_URL_ENV, DEFAULT_SEARXNG_URL)
    if _searxng_answers(base_url):
        return FallbackProvider(SearxngProvider(base_url),
                                BingHtmlProvider(), DdgHtmlProvider())
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
    _wrapped = ("https://www.bing.com/ck/a?!&&p=hash&u=a1"
                + base64.urlsafe_b64encode(b"https://ex.com/c").decode()
                .rstrip("=") + "&ntb=1")
    assert _decode_bing_href(_wrapped) == "https://ex.com/c"
    _BING = ('<li class="b_algo"><h2><a href="' + _wrapped
             + '">Cee</a></h2><p>cee snippet</p></li>')
    parsed = _parse_bing_results(_BING, DEFAULT_MAX_RESULTS)
    assert parsed == [SearchResult("Cee", "https://ex.com/c",
                                   "cee snippet")], parsed
    _MOJEEK = ('<ul><li><a class="title" href="https://ex.com/d">Dee</a>'
               '<p class="s">dee snippet</p></li></ul>')
    parsed = _parse_mojeek_results(_MOJEEK, DEFAULT_MAX_RESULTS)
    assert parsed == [SearchResult("Dee", "https://ex.com/d",
                                   "dee snippet")], parsed
    assert issubclass(SearchUnavailable, Exception)
    print("smoke OK")
