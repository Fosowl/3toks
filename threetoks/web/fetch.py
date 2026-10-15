"""Plain-HTTP page fetch.

No browser process (see docs/recon-agenticseek.md "Do NOT take"): one
curl_cffi GET that impersonates Chrome (TLS and HTTP/2 fingerprint, with
Chrome's own headers), size-capped and timeout-guarded. A bot-check page
(Cloudflare, DataDome, Reddit "Prove your humanity") raises ``BlockedPage``
whatever its HTTP status. JS-heavy pages go to the browser fetcher.
"""
from curl_cffi import requests as curl_requests

from threetoks.web.blockpage import raise_if_blocked
from threetoks.web.errors import FetchError

DEFAULT_TIMEOUT_S = 15
DEFAULT_MAX_BYTES = 2_000_000
IMPERSONATE = "chrome"
HTML_CONTENT_TYPE = "text/html"
HTTP_OK = 200
FALLBACK_ENCODING = "utf-8"


def _read_capped(response: curl_requests.Response, max_bytes: int) -> bytes:
    """Read a streamed body, but stop soon after the size cap.

    The fetcher uses this so that a huge page does not fill the memory.
    ``_checked_html`` calls it with one open streamed response.
    It joins body chunks and stops at the first chunk past ``max_bytes``.

    Args:
        response (curl_requests.Response): an open response with stream=True.
        max_bytes (int): the largest body size the caller accepts.

    Returns:
        bytes: the body; longer than ``max_bytes`` when the page is too big.
    """
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content():
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            break
    return b"".join(chunks)


def _reject_non_html(response: curl_requests.Response, url: str) -> None:
    """Stop a response that is not an HTML page.

    The fetcher uses this to skip PDFs, images and JSON before it reads them.
    ``_checked_html`` calls it with one HTTP 200 response and its URL.
    It reads the Content-Type header and raises when it is not HTML.

    Args:
        response (curl_requests.Response): the open HTTP 200 response.
        url (str): the page URL, for the error message.

    Returns:
        None: it returns only when the content type is HTML.
    """
    content_type = response.headers.get("Content-Type", "")
    if HTML_CONTENT_TYPE not in content_type.lower():
        raise FetchError(f"{url}: not HTML ({content_type or 'unknown'})")


def _checked_html(response: curl_requests.Response, url: str,
                  max_bytes: int) -> str:
    """Turn one response into page HTML, or raise the right failure.

    The fetcher uses this to sort bot checks from other bad answers.
    ``fetch_html`` calls it with the open streamed response of ``url``.
    It reads the capped body, raises ``BlockedPage`` for a bot check at
    any status, then ``FetchError`` for a bad status or an oversize body.

    Args:
        response (curl_requests.Response): the open streamed response.
        url (str): the page URL, for the error messages.
        max_bytes (int): the largest body size the caller accepts.

    Returns:
        str: the decoded HTML of a normal HTTP 200 page.
    """
    status = response.status_code
    if status == HTTP_OK:
        _reject_non_html(response, url)
    body = _read_capped(response, max_bytes)
    html = _decode(body[:max_bytes], response.charset_encoding)
    raise_if_blocked(html, url)
    if status != HTTP_OK:
        raise FetchError(f"{url}: HTTP {status}")
    if len(body) > max_bytes:
        raise FetchError(f"{url}: body exceeds {max_bytes} bytes")
    return html


def _decode(body: bytes, charset: str | None) -> str:
    """Decode a page body, with UTF-8 when the declared charset is unknown.

    Sites sometimes declare a charset name Python does not know.
    ``_checked_html`` calls it with the capped body and the header charset.
    It decodes with replacement characters, so it never raises.

    Args:
        body (bytes): the page body.
        charset (str | None): the charset from the Content-Type header.

    Returns:
        str: the decoded text.
    """
    try:
        return body.decode(charset or FALLBACK_ENCODING, errors="replace")
    except LookupError:
        return body.decode(FALLBACK_ENCODING, errors="replace")


def fetch_html(url: str, timeout_s: int = DEFAULT_TIMEOUT_S,
               max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Download one web page as HTML, like a Chrome browser would.

    The web agent uses this in ``http`` mode to read pages for answers.
    ``cli.fetch_page`` calls it with a page URL from search.
    It GETs the page with the impersonated Chrome fingerprint and gives
    every failure (network, status, type, size, bot check) as FetchError.

    Args:
        url (str): the http(s) URL of the page.
        timeout_s (int): the request timeout in seconds.
        max_bytes (int): the largest body size the caller accepts.

    Returns:
        str: the decoded HTML of the page.
    """
    response = None
    try:
        response = curl_requests.get(url, impersonate=IMPERSONATE,
                                     timeout=timeout_s, stream=True)
        return _checked_html(response, url, max_bytes)
    except curl_requests.exceptions.RequestException as error:
        raise FetchError(f"{url}: {error}") from error
    finally:
        if response is not None:
            response.close()


if __name__ == "__main__":
    assert _decode(b"caf\xc3\xa9", "no-such-charset") == "café"
    try:
        fetch_html("http://127.0.0.1:9/definitely-down", timeout_s=1)
    except FetchError as error:
        assert "127.0.0.1" in str(error), error
    else:
        raise AssertionError("expected FetchError on unreachable host")
    print("smoke OK")
