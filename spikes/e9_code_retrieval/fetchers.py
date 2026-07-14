"""Search + fetch hops for E9, mirroring threetoks/web/search.py's
FallbackProvider pattern: fall back on EMPTY results, not just an
exception, and keep every hop honest about whether it actually answered.

Confirmed from THIS machine (see REPORT.md "Providers" section for the
full writeup):
  - localhost:8080 SearXNG: WORKS initially, but its outbound engines
    (brave/duckduckgo/qwant/startpage/wikipedia, then eventually even
    google) got rate-limited/CAPTCHA'd mid-session BY OUR OWN TESTING
    VOLUME — visible on the instance's own /stats page as
    "Suspended: too many requests" / "CAPTCHA". A fresh instance would
    not start in this state.
  - Mojeek (www.mojeek.com/search): WORKS, plain HTML, no challenge —
    the hop that ended up carrying most of the live benchmark once
    SearXNG's engines were exhausted.
  - Bing HTML (bing.com/search): returns a captcha/verify challenge page
    for every query tried, including plain queries with no site: filter
    — BLOCKED from this network, not just this query shape.
  - DuckDuckGo HTML, both html.duckduckgo.com and lite.duckduckgo.com:
    both return an "anomaly" challenge page — BLOCKED network-wide, not
    just one endpoint.
  - Startpage HTML: redirects into an interstitial captcha page — BLOCKED.
  - GitHub code search (both api.github.com/search/code and the
    github.com/search?type=code web UI): returns 401 / an empty
    logged_in:false payload without a token — UNUSABLE unauthenticated.
  - grep.app API: returns a Vercel bot-checkpoint HTML page to curl —
    UNUSABLE.
  - searchcode.com /api/codesearch_I/: 404, endpoint gone — UNUSABLE.
  - api.github.com REPO metadata / git trees / raw file content: WORKS
    unauthenticated (60 req/hour rate limit) — used only to turn a
    github.com/<owner>/<repo> hit (no path) into a shortlist of .py
    files, never for search itself.
  - raw.githubusercontent.com: WORKS, not bot-walled — every accepted
    GitHub snippet is fetched from here.

So the FallbackProvider chain has 4 hops now (SearXNG -> Mojeek -> Bing
-> DDG), mirroring search.py's shape of "keep trying the next hop on
empty results"; only SearXNG (partially, until it wasn't) and Mojeek
were actually alive on this network during this run.
"""
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from threetoks.web.search import (  # noqa: E402
    BingHtmlProvider, DdgHtmlProvider, FallbackProvider, SearxngProvider,
    _parse_searxng_results, _searxng_answers,
)

SEARXNG_TIMEOUT_S = 10

FETCH_TIMEOUT_S = 10
GITHUB_API_TIMEOUT_S = 10
SEARCH_TIMEOUT_S = 10
MAX_TREE_FILES = 3
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
_BLOB_RE = re.compile(r"^/([^/]+)/([^/]+)/blob/([^/]+)/(.+?)$")
_REPO_RE = re.compile(r"^/([^/]+)/([^/]+)/?$")
_SKIP_PY_NAMES = {"__init__.py", "setup.py", "conftest.py"}
# github.com/<x>/<y> paths where <x> is a site section, not a real owner
# (e.g. github.com/topics/caesar-cipher) — would otherwise burn an API
# call on a guaranteed-404 "repo".
_NON_OWNER_SEGMENTS = {"topics", "search", "marketplace", "sponsors",
                       "collections", "trending", "orgs", "about", "features"}


def _is_plain_github_com(url: str) -> bool:
    """True only for the real github.com host — NOT gist.github.com,
    raw.githubusercontent.com, github.io, etc. A substring match on
    'github.com' misrouted gist URLs to the repo-tree resolver (they look
    like github.com/<user>/<gist-id>), which then wasted GitHub API calls
    on a 404; this got caught live in the first benchmark dry run."""
    return urlparse(url).netloc == "github.com"


class PinnedEngineSearxng:
    """SearxngProvider that queries one fixed engine instead of rotating
    through threetoks.web.search.ENGINE_ROTATION's 6 pairs.

    Deviation from the shared provider, done deliberately: mid-session
    probing (documented in REPORT.md) tripped "Suspended: too many
    requests" / CAPTCHA on brave, duckduckgo, qwant, startpage AND
    wikipedia on this machine's local SearXNG instance — confirmed via
    its own /stats page — while `google` recovered fastest and answered
    a spot-check. Rotating through the stock 6 pairs would have meant
    ~4/6 of queries landing on an already-dead engine. Pinning to one
    engine avoids compounding that; google itself later also went quiet
    under continued load (see REPORT.md) — the honest reading is that
    this session's own test volume exhausted the shared local instance's
    goodwill with every one of its 6 upstream engines in turn.
    """

    def __init__(self, base_url: str, engines=("google",)):
        self.base_url = base_url.rstrip("/")
        self.engines = engines

    def search(self, query: str, max_results: int = 8) -> list:
        response = requests.post(
            f"{self.base_url}/search",
            data={"q": query, "categories": "general", "language": "auto",
                  "engines": ",".join(self.engines)},
            headers={"User-Agent": BROWSER_UA},
            timeout=SEARXNG_TIMEOUT_S,
        )
        response.raise_for_status()
        return _parse_searxng_results(response.text, max_results)


def _parse_mojeek_results(html: str, max_results: int) -> list:
    """Parse a Mojeek HTML results page (a.title anchors, li>p.s snippet)."""
    from threetoks.web.search import SearchResult
    soup = BeautifulSoup(html, "html.parser")
    results = []
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
            if snippet_tag else "",
        ))
        if len(results) >= max_results:
            break
    return results


class MojeekHtmlProvider:
    """Scrape mojeek.com/search — plain HTML, no JS challenge encountered
    on this machine, discovered as a working hop only after Bing, DDG
    (both endpoints), and Startpage all turned out to be bot-walled."""

    def search(self, query: str, max_results: int = 8) -> list:
        response = requests.get(
            "https://www.mojeek.com/search", params={"q": query},
            headers={"User-Agent": BROWSER_UA}, timeout=SEARCH_TIMEOUT_S)
        response.raise_for_status()
        return _parse_mojeek_results(response.text, max_results)


def build_provider_chain():
    """4-hop FallbackProvider, same "fall back on empty, not just
    exception" shape as threetoks.web.search.auto_provider(): SearXNG (if
    the local instance answers) -> Mojeek -> Bing -> DDG. Returns the
    provider plus a note on which hop actually answered the reachability
    probe (for the report to state plainly, per task instructions) — NOT
    a guarantee that hop stays alive for the whole run; see REPORT.md."""
    rest = FallbackProvider(MojeekHtmlProvider(),
                            FallbackProvider(BingHtmlProvider(),
                                             DdgHtmlProvider()))
    if _searxng_answers("http://localhost:8080"):
        chain = FallbackProvider(
            PinnedEngineSearxng("http://localhost:8080"), rest)
        note = "searxng (localhost:8080) reachable, used as primary hop"
    else:
        chain = rest
        note = "searxng NOT reachable, falling back to Mojeek/Bing/DDG HTML"
    return chain, note


class GithubApiRateLimited(Exception):
    """Raised when api.github.com answers with a rate-limit block."""


def _get(url: str, timeout: int, github: bool = False):
    headers = {"User-Agent": BROWSER_UA}
    if github:
        headers["Accept"] = "application/vnd.github+json"
    resp = requests.get(url, headers=headers, timeout=timeout)
    if github and resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise GithubApiRateLimited(url)
    resp.raise_for_status()
    return resp


def fetch_text(url: str) -> str | None:
    """Plain GET, browser UA, returns None on any failure (never raises)."""
    try:
        return _get(url, FETCH_TIMEOUT_S).text
    except Exception:
        return None


def resolve_github_blob(url: str) -> str | None:
    """github.com/<owner>/<repo>/blob/<branch>/<path> -> raw URL, else None.

    Host-checked (see _is_plain_github_com) so gist.github.com and other
    lookalikes never match.
    """
    if not _is_plain_github_com(url):
        return None
    m = _BLOB_RE.match(urlparse(url).path)
    if not m:
        return None
    owner, repo, branch, path = m.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def resolve_github_repo_root(url: str) -> tuple:
    """github.com/<owner>/<repo> (no path) -> (owner, repo), else None."""
    if not _is_plain_github_com(url):
        return None
    m = _REPO_RE.match(urlparse(url).path)
    if not m or m.group(1) in _NON_OWNER_SEGMENTS:
        return None
    return m.group(1), m.group(2)


def github_repo_py_files(owner: str, repo: str, cache: dict) -> list:
    """Return up to MAX_TREE_FILES raw.githubusercontent.com URLs for
    plausible .py files in a repo's default branch, via the unauthenticated
    REST API (2 calls: repo metadata for the default branch, then the
    recursive tree). Cached per (owner, repo) since a benchmark run may
    see the same repo from more than one search hit.

    Returns [] (not an exception) when the API is unreachable or rate
    limited — callers treat that exactly like "no files found", logged
    as a failed hop rather than crashing the whole task.
    """
    key = (owner, repo)
    if key in cache:
        return cache[key]
    urls = []
    try:
        meta = _get(f"https://api.github.com/repos/{owner}/{repo}",
                    GITHUB_API_TIMEOUT_S, github=True).json()
        branch = meta.get("default_branch", "main")
        tree = _get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/"
            f"{branch}?recursive=1", GITHUB_API_TIMEOUT_S, github=True).json()
        py_paths = [item["path"] for item in tree.get("tree", [])
                    if item.get("type") == "blob"
                    and item["path"].endswith(".py")
                    and Path(item["path"]).name not in _SKIP_PY_NAMES]
        for path in py_paths[:MAX_TREE_FILES]:
            urls.append(
                f"https://raw.githubusercontent.com/{owner}/{repo}/"
                f"{branch}/{path}")
    except Exception:
        urls = []
    cache[key] = urls
    return urls


def extract_code_blocks(html: str) -> list:
    """Pull each <pre> block's text out of an HTML page (rosettacode,
    Stack Overflow, tutorial sites, etc.) as an independent candidate
    "file". Non-GitHub sources are fetchable and worth trying (task
    instructions) even though we cannot resolve their license."""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    blocks = []
    for pre in soup.find_all("pre"):
        text = pre.get_text("\n")
        if "def " in text:  # cheap prefilter before we bother ast-parsing
            blocks.append(text)
    return blocks


def classify_and_fetch(url: str, github_cache: dict) -> list:
    """Turn one search-result URL into a list of (source_url, file_text)
    "files" ready for extract.find_matches. Handles three shapes:
      - a direct github blob URL -> one raw fetch
      - a bare github repo URL -> up to MAX_TREE_FILES raw fetches via
        the REST API tree listing
      - anything else -> fetch the HTML and split into <pre> blocks
    Returns [] on total failure (never raises) so the caller's funnel
    count simply doesn't grow for that URL.
    """
    raw = resolve_github_blob(url)
    if raw:
        text = fetch_text(raw)
        return [(raw, text)] if text else []

    repo = resolve_github_repo_root(url)
    if repo:
        owner, name = repo
        py_urls = github_repo_py_files(owner, name, github_cache)
        out = []
        for py_url in py_urls:
            text = fetch_text(py_url)
            if text:
                out.append((py_url, text))
        return out

    html = fetch_text(url)
    if not html:
        return []
    blocks = extract_code_blocks(html)
    return [(url, block) for block in blocks]


if __name__ == "__main__":
    assert resolve_github_blob(
        "https://github.com/o/r/blob/main/pkg/f.py") == \
        "https://raw.githubusercontent.com/o/r/main/pkg/f.py"
    assert resolve_github_blob("https://example.com/x") is None
    assert resolve_github_repo_root("https://github.com/o/r") == ("o", "r")
    assert resolve_github_repo_root("https://github.com/o/r/blob/m/f.py") \
        is None
    # gist.github.com and github.com/topics/... must NOT be mistaken for
    # a github.com/<owner>/<repo> root (the bug caught in the live dry run).
    assert resolve_github_repo_root(
        "https://gist.github.com/AO8/3a89ba7c8f032c7a1ff505baa3ce970e") \
        is None
    assert resolve_github_blob(
        "https://gist.github.com/AO8/3a89ba7c8f032c7a1ff505baa3ce970e") \
        is None
    assert resolve_github_repo_root(
        "https://github.com/topics/caesar-cipher?l=python") is None
    html = "<html><body><pre>def f(x):\n    return x\n</pre>" \
           "<pre>not code</pre></body></html>"
    blocks = extract_code_blocks(html)
    assert blocks == ["def f(x):\n    return x\n"], blocks

    mojeek_fixture = (
        '<ul><li class="r1"><a class="title" href="https://ex.com/a">'
        "Alpha result</a><p class=\"s\">alpha snippet text</p></li></ul>")
    parsed = _parse_mojeek_results(mojeek_fixture, 8)
    assert len(parsed) == 1 and parsed[0].url == "https://ex.com/a" \
        and parsed[0].snippet == "alpha snippet text", parsed
    print("smoke OK")
