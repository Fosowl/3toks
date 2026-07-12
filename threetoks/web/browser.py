"""Selenium/Chrome page fetch for JS-rendered pages.

The plain-HTTP fetcher (``fetch.py``) returns raw markup, so single-page
apps and cookie-walled sites hand it only chrome (banners, spinners). This
fetcher drives a real Chrome via Selenium, waits for the DOM to settle, and
hands the rendered ``page_source`` to the same ``html_to_page`` pipeline —
so the vertical sees content instead of noise. Set ``visible=True`` to watch
the agent browse in a live window.

Ported from agenticSeek's ``sources/browser.py`` (get_chrome_path /
create_chrome_options / create_driver), stripped of all anti-bot machinery:
no undetected_chromedriver, no stealth, no fake user agents, no human-jitter
sleeps, no captcha extension. Plain selenium + chrome.

Driver resolution relies on selenium-manager (selenium >= 4.6), which
downloads a matching chromedriver automatically — no separate install.
"""
import os
import sys
import time

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

from threetoks.web.fetch import FetchError
from threetoks.web.textify import PageText, html_to_page

WINDOW_SIZE = "1280,900"
PAGE_LOAD_TIMEOUT_S = 20
READY_STATE_TIMEOUT_S = 8
JS_SETTLE_S = 1.0
CHROME_ENV_VAR = "CHROME_EXECUTABLE_PATH"
_READY_SCRIPT = "return document.readyState"
_READY_COMPLETE = "complete"
_MAC_CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
)
_LINUX_CHROME_PATHS = (
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
)
_WIN_CHROME_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)
_DRIVER_HELP = (
    "Could not start chromedriver. Update selenium (>=4.6 auto-resolves it) "
    "or install a chromedriver matching your Chrome from "
    "https://googlechromelabs.github.io/chrome-for-testing/ and put it on PATH."
)


def _platform_chrome_paths() -> tuple:
    """Return candidate Chrome binary paths for the current OS."""
    if sys.platform.startswith("darwin"):
        return _MAC_CHROME_PATHS
    if sys.platform.startswith("win"):
        return _WIN_CHROME_PATHS
    return _LINUX_CHROME_PATHS


def get_chrome_path() -> str:
    """Return the Chrome binary path, honouring ``CHROME_EXECUTABLE_PATH``.

    Checks the env-var override first, then the standard per-OS install
    locations. Raises ``FetchError`` when no runnable Chrome is found.
    """
    override = os.environ.get(CHROME_ENV_VAR)
    candidates = (override, *_platform_chrome_paths()) if override \
        else _platform_chrome_paths()
    for path in candidates:
        if path and os.path.exists(path) and os.access(path, os.X_OK):
            return path
    raise FetchError(
        f"Google Chrome not found (set {CHROME_ENV_VAR} to its path).")


def create_chrome_options(visible: bool) -> Options:
    """Build plain Chrome options: headless-when-hidden, fixed window, sandbox off."""
    options = Options()
    options.binary_location = get_chrome_path()
    if not visible:
        options.add_argument("--headless=new")
    options.add_argument(f"--window-size={WINDOW_SIZE}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    return options


def create_driver(visible: bool) -> webdriver.Chrome:
    """Start a Chrome WebDriver via selenium-manager, or raise ``FetchError``."""
    options = create_chrome_options(visible)
    try:
        driver = webdriver.Chrome(options=options)
    except WebDriverException as error:
        raise FetchError(f"{_DRIVER_HELP} ({error.msg or error})") from error
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_S)
    return driver


class BrowserFetcher:
    """Fetch pages through a real Chrome, rendering JS before extraction.

    The driver is created lazily on the first ``fetch_page`` call so that
    constructing a fetcher (or importing the TUI) never launches Chrome.
    Call ``close`` when done; it is safe to call more than once.
    """

    def __init__(self, visible: bool = False):
        """Store the visibility flag; no browser starts until first fetch."""
        self.visible = visible
        self.driver = None
        self.last_url = None

    def _ensure_driver(self) -> webdriver.Chrome:
        """Start the driver on first use and reuse it afterwards."""
        if self.driver is None:
            self.driver = create_driver(self.visible)
        return self.driver

    def _wait_until_ready(self, driver: webdriver.Chrome) -> None:
        """Block until ``document.readyState`` is complete, then let JS settle."""
        WebDriverWait(driver, READY_STATE_TIMEOUT_S).until(
            lambda d: d.execute_script(_READY_SCRIPT) == _READY_COMPLETE)
        time.sleep(JS_SETTLE_S)

    def fetch_page(self, url: str) -> PageText:
        """Load ``url`` in Chrome and return its extracted ``PageText``.

        Any browser-side failure (bad URL, load timeout, driver crash) is
        re-raised as ``FetchError`` so the vertical's existing error path
        handles it exactly like an HTTP failure.
        """
        self.last_url = url
        driver = self._ensure_driver()
        try:
            driver.get(url)
            self._wait_until_ready(driver)
            html = driver.page_source
        except WebDriverException as error:
            raise FetchError(f"{url}: {error.msg or error}") from error
        return html_to_page(html, url)

    def close(self) -> None:
        """Quit the driver if it was started; safe to call repeatedly."""
        if self.driver is not None:
            try:
                self.driver.quit()
            except WebDriverException:
                pass
            self.driver = None


def make_fetcher(visible: bool = False) -> BrowserFetcher:
    """Create a ``BrowserFetcher``; pass ``visible=True`` for a live window."""
    return BrowserFetcher(visible=visible)


if __name__ == "__main__":
    class _FakeDriver:
        """Minimal driver double: records calls, returns canned HTML."""

        page_source = "<html><title>T</title><body>" \
            "<p>The capital of France is Paris today.</p></body></html>"

        def __init__(self):
            self.got = []
            self.quit_count = 0

        def get(self, url):
            self.got.append(url)

        def execute_script(self, _script):
            return _READY_COMPLETE

        def set_page_load_timeout(self, _seconds):
            pass

        def quit(self):
            self.quit_count += 1

    fetcher = make_fetcher()
    assert fetcher.driver is None, "driver must be lazy"
    fake = _FakeDriver()
    fetcher.driver = fake  # inject to avoid launching real Chrome
    fetcher._wait_until_ready = lambda _driver: None  # skip the 1s settle
    page = fetcher.fetch_page("https://example.com/")
    assert fetcher.last_url == "https://example.com/", fetcher.last_url
    assert fake.got == ["https://example.com/"], fake.got
    assert any("Paris" in s for s in page.sentences), page.sentences
    fetcher.close()
    fetcher.close()  # idempotent
    assert fake.quit_count == 1, fake.quit_count
    print("smoke OK")
