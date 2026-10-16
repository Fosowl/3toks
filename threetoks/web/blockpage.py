"""Recognise bot-check pages (CAPTCHA, "Just a moment...") in fetched HTML.

Every fetcher (plain HTTP and Chrome) hands its HTML here before text
extraction, so a challenge page becomes a ``BlockedPage`` failure instead
of "content" the model would read and quote.

The markers were measured live (2026-10-15) on block pages and on normal
pages of the same sites. Each one appeared only on block pages. Do NOT add
the bare word "captcha", ``challenge-platform`` or ``g-recaptcha``: normal
Wikipedia, Medium, Reddit, Stack Overflow and Indeed pages carry them.
"""
from threetoks.web.errors import BlockedPage

# Script/DOM markers that only a challenge page carries, with the vendor.
_BODY_MARKERS = (
    ("_cf_chl_opt", "cloudflare challenge"),
    ("captcha-delivery.com", "datadome captcha"),
    ("px-captcha", "perimeterx captcha"),
)
# Lower-case title fragments of challenge pages that carry no body marker.
_TITLE_MARKERS = (
    "just a moment...",
    "attention required! | cloudflare",
    "prove your humanity",
    "access to this page has been denied",
)
_TITLE_OPEN = "<title"
_TITLE_CLOSE = "</title>"


def page_title(lowered_html: str) -> str:
    """Return the text of the first title tag.

    The block check uses this to read the page title without a parser.
    ``block_reason`` calls it with HTML that is already in lower case.
    It returns the text between the first title tags, or an empty string.

    Args:
        lowered_html (str): full page HTML, already lower case.

    Returns:
        str: the stripped title text, or "" when the page has no title.
    """
    start = lowered_html.find(_TITLE_OPEN)
    if start < 0:
        return ""
    text_start = lowered_html.find(">", start) + 1
    end = lowered_html.find(_TITLE_CLOSE, text_start)
    if text_start == 0 or end < 0:
        return ""
    return lowered_html[text_start:end].strip()


def block_reason(html: str) -> str | None:
    """Tell why a page is a bot check, or return None for a normal page.

    The fetchers use this to stop a challenge page before text extraction.
    ``raise_if_blocked`` and the browser's challenge wait call it with the
    raw HTML of one page. It looks for vendor markers, then for titles.

    Args:
        html (str): the full HTML of one fetched page.

    Returns:
        str | None: a short reason such as "cloudflare challenge", or None.
    """
    lowered = html.lower()
    for marker, reason in _BODY_MARKERS:
        if marker in lowered:
            return reason
    title = page_title(lowered)
    for fragment in _TITLE_MARKERS:
        if fragment in title:
            return f"challenge page '{title}'"
    return None


def raise_if_blocked(html: str, url: str) -> None:
    """Raise ``BlockedPage`` when the HTML is a bot check.

    Both fetchers use this so a challenge page never reaches the model.
    ``fetch.fetch_html`` and ``browser.BrowserFetcher`` call it with the
    raw HTML of one page and its URL.
    It returns quietly for a normal page.

    Args:
        html (str): the full HTML of one fetched page.
        url (str): the page URL, for the error message.
    """
    reason = block_reason(html)
    if reason:
        raise BlockedPage(f"{url}: blocked by a bot check ({reason})")


if __name__ == "__main__":
    challenge = ("<html><head><title>Just a moment...</title></head><body>"
                 "<script>window._cf_chl_opt={}</script></body></html>")
    assert block_reason(challenge) == "cloudflare challenge"
    reddit = "<title>Reddit - Prove your humanity</title><p>hi</p>"
    assert "prove your humanity" in block_reason(reddit)
    normal = ("<title>Ethanol - Wikipedia</title><script src='/cdn-cgi/"
              "challenge-platform/x.js'></script><div class='g-recaptcha'>")
    assert block_reason(normal) is None
    assert page_title("<p>no title</p>") == ""
    print("smoke OK")
