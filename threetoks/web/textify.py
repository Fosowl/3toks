"""HTML to numbered sentences and clean links.

Ports agenticSeek's ``Browser.get_text()`` / ``is_sentence()`` /
``is_link_valid()`` pipeline (sources/browser.py) but drives it from an
HTML *string* instead of a live Selenium driver, so it needs no browser:

    strip script/style/noscript/meta/link
        -> markdownify (drop <a> tags)
        -> per-line noise filter (is_sentence)
        -> split lines into sentences
        -> cap at max_sentences

Links are absolute-ized against the base URL, filtered by ported
heuristics, deduplicated, and capped. The result feeds the numbered-
sentence renderer; the model chooses sentences to quote, never rewrites.
"""
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

DEFAULT_MAX_SENTENCES = 128
DEFAULT_MAX_LINKS = 32
MIN_SENTENCE_WORDS = 5
DIGIT_MIN_WORDS = 3
CLAUSE_MIN_WORDS = 4
PROSE_MIN_WORDS = 7
LABEL_MAX_CHARS = 60
LINK_MAX_CHARS = 72
STRIP_TAGS = ("script", "style", "noscript", "meta", "link")
SENTENCE_END_CHARS = ".!?。！？，,।۔"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
METADATA_EXTENSIONS = (".ico", ".xml", ".json", ".rss", ".atom")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TRAILING_INDEX_RE = re.compile(r"/\d+$")
_IMAGE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_JUNK_RE = re.compile(
    r"\b(sign in|log in|signed? up|signed? in|notification settings|"
    r"posted by|cookies?|newsletter|"
    r"subscribe|all rights reserved|error while loading|please reload|"
    r"perform that action|enable javascript|checking your browser|"
    r"verify you are human)\b|^\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.IGNORECASE)
_BOILERPLATE_RE = re.compile(
    r"©|\(c\)\s*(?:19|20)\d{2}|\bcopyright\s+(?:19|20)\d{2}|"
    r"^\$\s|\|\s*(?:ba|z)?sh\b|\bcurl\s+-", re.IGNORECASE)
_LINE_MARKUP_RE = re.compile(r"^[#>*+\-\s]+")
_GLUED_WORD_RE = re.compile(r"[a-z][A-Z]")
MAX_GLUED_TRANSITIONS = 3
MAX_SENTENCE_CHARS = 300


@dataclass(frozen=True)
class PageLink:
    """A navigable link: display label plus absolute URL."""

    label: str
    url: str


@dataclass
class PageText:
    """Extracted page: title, numbered-ready sentences, links, source URL."""

    title: str
    sentences: list[str] = field(default_factory=list)
    links: list[PageLink] = field(default_factory=list)
    url: str = ""


def _is_sentence(text: str) -> bool:
    """True if a cleaned line reads as content, not nav/menu noise.

    Tuned on real pages (Goodreads chrome, tables): digit lines need a
    few words of context, punctuation-terminated prose needs
    MIN_SENTENCE_WORDS, clauses (comma/paren) slip in at
    CLAUSE_MIN_WORDS, and long unpunctuated prose at PROSE_MIN_WORDS —
    which drops short nav fragments like "Jump to ratings and reviews".
    """
    if text.rstrip().endswith("?"):
        return False  # interrogatives echo the task; never answer-notes
    if _JUNK_RE.search(text):
        return False  # forum/UI chrome: sign-in prompts, timestamps, legal
    if _BOILERPLATE_RE.search(text):
        return False  # © footers, copyright years, shell-command lines
    if len(_GLUED_WORD_RE.findall(text)) > MAX_GLUED_TRANSITIONS:
        return False  # "SolGPT-5.6TerraGPT" soup: nav lists glued together
    if len(text) > MAX_SENTENCE_CHARS:
        return False  # single unsplittable mega-line is never clean prose
    word_count = len(_WORD_RE.findall(text))
    if any(char.isdigit() for char in text):
        return word_count >= DIGIT_MIN_WORDS
    if word_count >= MIN_SENTENCE_WORDS \
            and text.endswith(tuple(SENTENCE_END_CHARS)):
        return True
    if "," in text or "(" in text:
        return word_count >= CLAUSE_MIN_WORDS
    return word_count >= PROSE_MIN_WORDS


def _clean_line(line: str) -> str:
    """Drop markdown artifacts: images, tables, list markers, bold."""
    line = _IMAGE_MD_RE.sub("", line)
    if line.lstrip().startswith("|") or line.count(" | ") >= 2:
        line = " ".join(part.strip() for part in line.split("|")
                        if part.strip())
    line = _LINE_MARKUP_RE.sub("", line.replace("**", ""))
    return " ".join(line.split())


def _markdown_lines(html: str) -> list[str]:
    """Strip noise tags, convert to markdown, return cleaned lines."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(list(STRIP_TAGS)):
        element.decompose()
    converter = MarkdownConverter(heading_style="ATX", strip=["a"],
                                  autolinks=False)
    body = soup.body if soup.body else soup
    markdown = converter.convert(str(body))
    cleaned = (_clean_line(line) for line in markdown.splitlines())
    return [line for line in cleaned if line]


def _split_sentences(line: str) -> list[str]:
    """Split a content line into sentences, dropping interrogative parts."""
    parts = (part.strip() for part in _SENTENCE_SPLIT_RE.split(line))
    return [part for part in parts if part and not part.endswith("?")]


def _extract_sentences(html: str, max_sentences: int) -> list[str]:
    """Filter markdown lines to content and split them into sentences."""
    sentences: list[str] = []
    seen: set[str] = set()
    for line in _markdown_lines(html):
        if not _is_sentence(line):
            continue
        for sentence in _split_sentences(line):
            if sentence not in seen:
                seen.add(sentence)
                sentences.append(sentence)
        if len(sentences) >= max_sentences:
            break
    return sentences[:max_sentences]


def _is_link_valid(url: str) -> bool:
    """Cheap heuristics: keep real http(s) pages, drop asset/junk URLs."""
    if len(url) > LINK_MAX_CHARS:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    if _TRAILING_INDEX_RE.search(parsed.path):
        return False
    lowered = url.lower()
    return not lowered.endswith(IMAGE_EXTENSIONS + METADATA_EXTENSIONS)


def _extract_links(html: str, base_url: str,
                   max_links: int) -> list[PageLink]:
    """Absolute-ize, filter, dedupe, and cap anchor links."""
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[PageLink] = []
    for anchor in soup.find_all("a", href=True):
        url = urljoin(base_url, anchor["href"].strip())
        if url in seen or not _is_link_valid(url):
            continue
        seen.add(url)
        label = anchor.get_text(strip=True)[:LABEL_MAX_CHARS]
        links.append(PageLink(label=label or url, url=url))
        if len(links) >= max_links:
            break
    return links


def _extract_title(html: str) -> str:
    """Return the page ``<title>`` text, or empty string."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.title.get_text(strip=True) if soup.title else ""


def html_to_page(html: str, base_url: str,
                 max_sentences: int = DEFAULT_MAX_SENTENCES,
                 max_links: int = DEFAULT_MAX_LINKS) -> PageText:
    """Turn an HTML string into title, content sentences, and links."""
    return PageText(
        title=_extract_title(html),
        sentences=_extract_sentences(html, max_sentences),
        links=_extract_links(html, base_url, max_links),
        url=base_url,
    )


if __name__ == "__main__":
    _HTML = """<html><head><title>Demo</title></head><body>
      <nav><a href="/home">Home</a> <a href="/about">About us here</a></nav>
      <p>The Eiffel Tower is 330 metres tall. It stands in Paris.</p>
      <a href="https://ok.example.com/article">Read the full article now</a>
    </body></html>"""
    page = html_to_page(_HTML, "https://ok.example.com/")
    assert page.title == "Demo", page.title
    assert any("Eiffel Tower" in s for s in page.sentences), page.sentences
    assert any("330 metres tall" in s for s in page.sentences), page.sentences
    assert len(page.sentences) >= 2, page.sentences
    assert any(link.url == "https://ok.example.com/article"
               for link in page.links), page.links
    print("smoke OK")
