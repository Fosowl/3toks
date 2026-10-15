"""Fetch failures shared by every page fetcher and the web vertical.

This module imports nothing, so the vertical can catch ``BlockedPage``
without loading the HTTP client or Selenium (the web extras are optional).
"""


class FetchError(Exception):
    """Raised when a page cannot be fetched as usable HTML."""


class BlockedPage(FetchError):
    """Raised when the site answered with a bot check instead of the page.

    The web vertical catches it to skip the whole host for the session.
    """


if __name__ == "__main__":
    assert issubclass(BlockedPage, FetchError)  # callers catching FetchError
    print("smoke OK")
