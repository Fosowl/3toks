"""Anti-bot Chrome fetch: undetected-chromedriver + selenium-stealth.

Plain HTTP (``fetch.py``) and plain Selenium (``browser.py``) both get
bot-detected: GitHub serves an error shell, search engines throw CAPTCHAs.
This fetcher ports agenticSeek's proven stealth stack — a patched
``undetected_chromedriver`` build, ``selenium_stealth`` fingerprint patches,
a rotated ``fake_useragent`` string, a randomized ``--user-data-dir``, and
the ``go_to`` anti-bot waits that block until a "checking your browser" /
CAPTCHA interstitial clears — then hands the rendered ``page_source`` to the
same ``html_to_page`` pipeline the other fetchers use.

Ported AS-IS from ``agenticSeek/sources/browser.py``
(get_chrome_path / create_chrome_options / create_driver / go_to). Dropped:
the nopecha CRX extension, human-jitter movement/scroll, form filling, and
screenshots — none of which the extraction path needs.

``undetected_chromedriver`` downloads and manages its own patched
chromedriver matched to the installed Chrome, so it bypasses any stale
system chromedriver on ``PATH``. Set ``visible=True`` to watch it browse.
"""
import os
import random
import sys
import time
import uuid

from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait

from threetoks.web.fetch import FetchError
from threetoks.web.textify import PageText, html_to_page

WINDOW_SIZE = "1920,1080"
PAGE_LOAD_TIMEOUT_S = 30
READY_STATE_TIMEOUT_S = 8
ANTIBOT_TIMEOUT_S = 10
JS_SETTLE_S = 1.0
CHROME_ENV_VAR = "CHROME_EXECUTABLE_PATH"
PROFILE_DIR_TEMPLATE = "/tmp/threetoks_chrome_profile_{token}"
_READY_SCRIPT = "return document.readyState"
_READY_COMPLETE = "complete"
_ANTIBOT_MARKERS = ("checking your browser", "captcha")
_WEBDRIVER_HIDE = (
    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
_MAC_CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
)
_LINUX_CHROME_PATHS = (
    "/usr/bin/google-chrome",
    "/opt/chrome/chrome",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/usr/local/bin/chrome",
)
_WIN_CHROME_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)
# Static UA pool (agenticSeek's), each paired with its navigator vendor so
# the selenium_stealth patch reports a consistent fingerprint.
_USER_AGENTS = (
    {"ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
     "vendor": "Google Inc.", "platform": "Win64"},
    {"ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
     "vendor": "Apple Inc.", "platform": "MacIntel"},
    {"ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
     "vendor": "Google Inc.", "platform": "Linux x86_64"},
)
_STEALTH_ARGS = (
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-features=TranslateUI",
    "--disable-ipc-flooding-protection",
    "--mute-audio",
    "--disable-notifications",
    "--autoplay-policy=user-gesture-required",
    "--disable-blink-features=AutomationControlled",
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


def get_random_user_agent() -> dict:
    """Pick one UA descriptor (string + vendor + platform) at random."""
    return random.choice(_USER_AGENTS)


def _random_profile_dir() -> str:
    """A throwaway per-session user-data-dir so profiles never collide."""
    return PROFILE_DIR_TEMPLATE.format(token=uuid.uuid4().hex[:8])


def create_chrome_options(visible: bool, agent: dict):
    """Build stealth Chrome options: fingerprint-flattened, fresh profile.

    Ports agenticSeek's ``create_chrome_options`` stealth path (minus the
    nopecha CRX): headless when hidden, a rotated user agent, a randomized
    user-data-dir, and the AutomationControlled blink flag.
    """
    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    options.binary_location = get_chrome_path()
    if not visible:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
    for flag in _STEALTH_ARGS:
        options.add_argument(flag)
    options.add_argument(f"--user-data-dir={_random_profile_dir()}")
    options.add_argument(f"--window-size={WINDOW_SIZE}")
    options.add_argument(f"user-agent={agent['ua']}")
    return options


def _apply_stealth(driver, agent: dict) -> None:
    """Patch the driver fingerprint with selenium_stealth + a webdriver hide."""
    from selenium_stealth import stealth

    driver.execute_script(_WEBDRIVER_HIDE)
    stealth(driver,
            languages=["en-US", "en"],
            vendor=agent["vendor"],
            platform=agent["platform"],
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True)


def create_driver(visible: bool = False):
    """Start an undetected-chromedriver Chrome with stealth patches applied.

    ``uc.Chrome`` downloads its own patched driver matched to the installed
    Chrome, so a stale system chromedriver on ``PATH`` is ignored. Any
    startup failure is re-raised as ``FetchError``.
    """
    import undetected_chromedriver as uc

    agent = get_random_user_agent()
    options = create_chrome_options(visible, agent)
    try:
        driver = uc.Chrome(options=options)
    except Exception as error:  # noqa: BLE001 - uc raises varied low-level errors
        raise FetchError(f"could not start undetected chromedriver: "
                         f"{error}") from error
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_S)
    _apply_stealth(driver, agent)
    return driver


class StealthBrowserFetcher:
    """Fetch pages through a stealth Chrome that passes most bot checks.

    The driver is created lazily on the first ``fetch_page`` and then reused
    for every later fetch — session persistence (cookies, warmed profile)
    helps subsequent pages clear bot checks and makes them fast after the
    slow first startup. Call ``close`` when done; it is safe to call twice.
    """

    def __init__(self, visible: bool = False):
        """Store the visibility flag; no browser starts until first fetch."""
        self.visible = visible
        self.driver = None
        self.last_url = None

    def _ensure_driver(self):
        """Start the stealth driver on first use and reuse it afterwards."""
        if self.driver is None:
            self.driver = create_driver(self.visible)
        return self.driver

    def _wait_past_antibot(self, driver) -> None:
        """Block until a 'checking your browser' / CAPTCHA screen clears.

        Ported from ``go_to``: a bounded wait on the page source. A timeout
        is not fatal — some pages never carry the marker — so we return and
        let extraction proceed on whatever rendered.
        """
        try:
            WebDriverWait(driver, ANTIBOT_TIMEOUT_S).until(
                lambda d: not any(marker in d.page_source.lower()
                                  for marker in _ANTIBOT_MARKERS))
        except TimeoutException:
            pass

    def _wait_until_ready(self, driver) -> None:
        """Wait past bot checks, for ``readyState`` complete, then settle."""
        self._wait_past_antibot(driver)
        WebDriverWait(driver, READY_STATE_TIMEOUT_S).until(
            lambda d: d.execute_script(_READY_SCRIPT) == _READY_COMPLETE)
        time.sleep(JS_SETTLE_S)

    def fetch_page(self, url: str) -> PageText:
        """Load ``url`` in stealth Chrome and return its ``PageText``.

        Any browser-side failure (bad URL, load timeout, driver crash) is
        re-raised as ``FetchError`` so the vertical handles it exactly like
        an HTTP failure.
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
            except Exception:  # noqa: BLE001 - uc teardown can raise late
                pass
            self.driver = None


def make_stealth_fetcher(visible: bool = False) -> StealthBrowserFetcher:
    """Create a ``StealthBrowserFetcher``; ``visible=True`` shows the window."""
    return StealthBrowserFetcher(visible=visible)


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

    fetcher = make_stealth_fetcher()
    assert fetcher.driver is None, "driver must be lazy"
    fake = _FakeDriver()
    fetcher.driver = fake  # inject to avoid launching real Chrome
    fetcher._wait_until_ready = lambda _driver: None  # skip waits
    page = fetcher.fetch_page("https://example.com/")
    assert fetcher.last_url == "https://example.com/", fetcher.last_url
    assert fake.got == ["https://example.com/"], fake.got
    assert any("Paris" in s for s in page.sentences), page.sentences
    fetcher.close()
    fetcher.close()  # idempotent
    assert fake.quit_count == 1, fake.quit_count

    dead_fetcher = make_stealth_fetcher()
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
