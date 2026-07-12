"""Plain-HTTP page fetch.

No browser process (see docs/recon-agenticseek.md "Do NOT take"): a bare
``requests`` GET with a desktop User-Agent, size-capped and timeout-guarded.
JS-heavy pages are out of scope for the prototype; a browser Fetcher can
slot in later behind the same call.
"""
import requests

DEFAULT_TIMEOUT_S = 15
DEFAULT_MAX_BYTES = 2_000_000
DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)
HTML_CONTENT_TYPE = "text/html"


class FetchError(Exception):
    """Raised when a page cannot be fetched as usable HTML."""


def _reject_non_html(response: requests.Response, url: str) -> None:
    """Raise ``FetchError`` unless the response looks like HTML."""
    if response.status_code != 200:
        raise FetchError(f"{url}: HTTP {response.status_code}")
    content_type = response.headers.get("Content-Type", "")
    if HTML_CONTENT_TYPE not in content_type.lower():
        raise FetchError(f"{url}: not HTML ({content_type or 'unknown'})")


def fetch_html(url: str, timeout_s: int = DEFAULT_TIMEOUT_S,
               max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Fetch ``url`` and return its HTML, or raise ``FetchError``.

    Rejects non-200 responses, non-HTML content types, oversize bodies,
    and any network/timeout failure — all as ``FetchError`` so callers
    catch one exception type.
    """
    try:
        response = requests.get(
            url, headers={"User-Agent": DESKTOP_USER_AGENT},
            timeout=timeout_s, stream=True,
        )
    except requests.RequestException as error:
        raise FetchError(f"{url}: {error}") from error
    with response:
        _reject_non_html(response, url)
        body = response.content
    if len(body) > max_bytes:
        raise FetchError(f"{url}: body exceeds {max_bytes} bytes")
    return body.decode(response.encoding or "utf-8", errors="replace")


if __name__ == "__main__":
    for bad in (FetchError, ):
        assert issubclass(bad, Exception)
    try:
        fetch_html("http://127.0.0.1:9/definitely-down", timeout_s=1)
    except FetchError as error:
        assert "127.0.0.1" in str(error), error
    else:
        raise AssertionError("expected FetchError on unreachable host")
    print("smoke OK")
