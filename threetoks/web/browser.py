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
selenium-manager prefers a chromedriver found on PATH even when it cannot
drive the installed Chrome, which kills every session at startup (in
visible mode: a window that flashes open and instantly closes on each
fetch); such a stale driver is masked out of PATH for the startup call so
a matching one is resolved instead.
"""
import os
import re
import shutil
import subprocess
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
VERSION_TIMEOUT_S = 10
CHROME_ENV_VAR = "CHROME_EXECUTABLE_PATH"
_READY_SCRIPT = "return document.readyState"
_READY_COMPLETE = "complete"
_VERSION_RE = re.compile(r"(\d+)\.\d+")
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


def _binary_major_version(executable: str):
    """Major version from ``executable --version`` output, or ``None``.

    Works for Chrome ("Google Chrome 151.0.7922.19 beta") and chromedriver
    ("ChromeDriver 139.0.7258.5 (...)"); any spawn or parse failure means
    "unknown", never an exception.
    """
    try:
        proc = subprocess.run([executable, "--version"], capture_output=True,
                              text=True, timeout=VERSION_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    match = _VERSION_RE.search(proc.stdout)
    return int(match.group(1)) if match else None


def _path_without_chromedriver():
    """``PATH`` with every directory holding a chromedriver removed.

    Returns ``None`` when no chromedriver is on ``PATH`` — nothing to mask.
    """
    if shutil.which("chromedriver") is None:
        return None
    entries = os.environ.get("PATH", "").split(os.pathsep)
    kept = [entry for entry in entries
            if not shutil.which("chromedriver", path=entry)]
    return os.pathsep.join(kept)


def _path_chromedriver_is_stale(chrome_path: str) -> bool:
    """True when the ``PATH`` chromedriver cannot drive the installed Chrome.

    Chromedriver only supports the Chrome sharing its major version.
    Unknown versions count as compatible — never mask a driver we cannot
    read.
    """
    driver = shutil.which("chromedriver")
    if driver is None:
        return False
    driver_major = _binary_major_version(driver)
    chrome_major = _binary_major_version(chrome_path)
    if driver_major is None or chrome_major is None:
        return False
    return driver_major != chrome_major


def _start_chrome(options: Options, path_override) -> webdriver.Chrome:
    """Start ``webdriver.Chrome``, under a temporary ``PATH`` override if given."""
    if path_override is None:
        return webdriver.Chrome(options=options)
    original = os.environ.get("PATH", "")
    os.environ["PATH"] = path_override
    try:
        return webdriver.Chrome(options=options)
    finally:
        os.environ["PATH"] = original


def create_driver(visible: bool) -> webdriver.Chrome:
    """Start a Chrome WebDriver via selenium-manager, or raise ``FetchError``.

    A chromedriver on PATH whose version cannot drive the installed Chrome
    is masked out of PATH for the startup call, so selenium-manager
    resolves a matching driver instead of failing on the stale one (and
    stops warning about it). An unexplained startup failure retries once
    with the mask applied before giving up.
    """
    options = create_chrome_options(visible)
    masked_path = _path_without_chromedriver()
    mask_first = (masked_path is not None
                  and _path_chromedriver_is_stale(options.binary_location))
    try:
        driver = _start_chrome(options, masked_path if mask_first else None)
    except WebDriverException as error:
        if mask_first or masked_path is None:
            raise FetchError(f"{_DRIVER_HELP} ({error.msg or error})") from error
        try:
            driver = _start_chrome(options, masked_path)
        except WebDriverException as retry_error:
            raise FetchError(f"{_DRIVER_HELP} "
                             f"({retry_error.msg or retry_error})") from retry_error
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
            self._discard_dead_driver()
            raise FetchError(f"{url}: {error.msg or error}") from error
        return html_to_page(html, url)

    def _discard_dead_driver(self) -> None:
        """Drop the driver when its session no longer answers.

        A fetch failure can mean a dead browser (window closed by hand,
        Chrome crashed) or just a bad page. Probe the session: one that
        still answers is kept, so the window stays open; a dead one is
        discarded so the next fetch starts a fresh browser instead of
        failing forever.
        """
        try:
            self.driver.window_handles
        except WebDriverException:
            self.close()

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
        window_handles = ["w0"]

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

    class _DeadDriver(_FakeDriver):
        """A driver whose session died: every call raises."""

        @property
        def window_handles(self):
            raise WebDriverException("invalid session id")

        def get(self, url):
            raise WebDriverException("disconnected")

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

    dead_fetcher = make_fetcher()
    dead = _DeadDriver()
    dead_fetcher.driver = dead
    try:
        dead_fetcher.fetch_page("https://example.com/")
        raise AssertionError("expected FetchError from a dead session")
    except FetchError:
        pass
    assert dead_fetcher.driver is None, "dead driver must be discarded"
    assert dead.quit_count == 1, dead.quit_count
    print("smoke OK")
