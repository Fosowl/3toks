"""Selenium/Chrome page fetch for JS-rendered pages.

The plain-HTTP fetcher (``fetch.py``) returns raw markup, so single-page
apps and cookie-walled sites hand it only chrome (banners, spinners). This
fetcher drives a real Chrome via Selenium, waits for the DOM to settle, and
hands the rendered ``page_source`` to the same ``html_to_page`` pipeline —
so the vertical sees content instead of noise. Set ``visible=True`` to watch
the agent browse in a live window.

Ported from agenticSeek's ``sources/browser.py`` (get_chrome_path /
create_chrome_options / create_driver), without undetected_chromedriver,
selenium-stealth or fake user agents: those patches made the fingerprint
inconsistent (a Windows platform on a Mac user agent, an old Chrome
version). Instead Chrome hides only what automation adds, measured live
on bot.sannysoft.com (6 failed checks before, 0 after):

* no ``--enable-automation`` switch and no AutomationControlled blink
  feature, so ``navigator.webdriver`` is false;
* the GPU stays on, so WebGL reports a real renderer instead of null;
* headless only: the user agent says ``Chrome``, not ``HeadlessChrome``,
  with the browser's own client hints.

Chrome runs on one persistent profile (``chrome_profile.py``) so solved
bot checks and site cookies carry over between sessions. When a page is
a bot check (``blockpage.py``), the fetcher waits for it to clear; in a
visible window the user can solve it. If it does not clear, the fetch
raises ``BlockedPage``.

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

from threetoks.web.blockpage import block_reason, raise_if_blocked
from threetoks.web.chrome_profile import ChromeProfile
from threetoks.web.errors import FetchError
from threetoks.web.textify import PageText, html_to_page

WINDOW_SIZE = "1280,900"
PAGE_LOAD_TIMEOUT_S = 20
READY_STATE_TIMEOUT_S = 8
JS_SETTLE_S = 1.0
CHALLENGE_WAIT_HEADLESS_S = 10     # automatic JS checks clear in seconds
CHALLENGE_WAIT_VISIBLE_S = 45      # time for the user to solve it
CHALLENGE_POLL_S = 1.0
VERSION_TIMEOUT_S = 10
CHROME_ENV_VAR = "CHROME_EXECUTABLE_PATH"
_READY_SCRIPT = "return document.readyState"
_READY_COMPLETE = "complete"
_USER_AGENT_SCRIPT = "return navigator.userAgent"
_CLIENT_HINTS_SCRIPT = (
    "return navigator.userAgentData.getHighEntropyValues(['architecture',"
    "'bitness','fullVersionList','model','platformVersion','wow64'])")
_HEADLESS_TOKEN = "HeadlessChrome"
_CHROME_TOKEN = "Chrome"
_CHROME_FLAGS = (
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
)
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


def create_chrome_options(visible: bool, profile_dir: str) -> Options:
    """Build Chrome options that hide the automation switches.

    ``create_driver`` uses these options to start the fetch browser.
    It calls this with the visibility flag and the claimed profile path.
    It sets headless mode when hidden, the window size, the persistent
    profile, and removes the switches that set ``navigator.webdriver``.
    The GPU stays on: without it WebGL is null, which bot checks flag.

    Args:
        visible (bool): True shows a window; False runs headless.
        profile_dir (str): the Chrome ``--user-data-dir`` to use.

    Returns:
        Options: ready-to-use Selenium Chrome options.
    """
    options = Options()
    options.binary_location = get_chrome_path()
    if not visible:
        options.add_argument("--headless=new")
    options.add_argument(f"--window-size={WINDOW_SIZE}")
    options.add_argument(f"--user-data-dir={profile_dir}")
    for flag in _CHROME_FLAGS:
        options.add_argument(flag)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
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


def _start_with_mask(options: Options) -> webdriver.Chrome:
    """Start Chrome, masking a stale PATH chromedriver, or raise ``FetchError``.

    ``create_driver`` uses this so a wrong chromedriver cannot kill startup.
    It calls this with the finished Chrome options.
    A chromedriver on PATH whose version cannot drive the installed Chrome
    is masked out of PATH for the startup call, so selenium-manager
    resolves a matching driver instead. An unexplained startup failure
    retries once with the mask applied before giving up.

    Args:
        options (Options): Chrome options from ``create_chrome_options``.

    Returns:
        webdriver.Chrome: the started driver.
    """
    masked_path = _path_without_chromedriver()
    mask_first = (masked_path is not None
                  and _path_chromedriver_is_stale(options.binary_location))
    try:
        return _start_chrome(options, masked_path if mask_first else None)
    except WebDriverException as error:
        if mask_first or masked_path is None:
            raise FetchError(f"{_DRIVER_HELP} ({error.msg or error})") from error
    try:
        return _start_chrome(options, masked_path)
    except WebDriverException as retry_error:
        raise FetchError(f"{_DRIVER_HELP} "
                         f"({retry_error.msg or retry_error})") from retry_error


def _unmask_headless_user_agent(driver: webdriver.Chrome) -> None:
    """Replace ``HeadlessChrome`` with ``Chrome`` in the user agent.

    Sites block the ``HeadlessChrome`` token, so headless sessions need it.
    ``create_driver`` calls this once on a new headless driver.
    It reads the real user agent and client hints from the browser and
    sets them back through CDP with only the token changed, so the user
    agent, the client hints and the platform still agree.

    Args:
        driver (webdriver.Chrome): a started headless driver.
    """
    agent = driver.execute_script(_USER_AGENT_SCRIPT)
    hints = driver.execute_script(_CLIENT_HINTS_SCRIPT)
    driver.execute_cdp_cmd("Network.setUserAgentOverride", {
        "userAgent": agent.replace(_HEADLESS_TOKEN, _CHROME_TOKEN),
        "userAgentMetadata": hints})


def create_driver(visible: bool, profile_dir: str) -> webdriver.Chrome:
    """Start the fetch browser, or raise ``FetchError``.

    ``BrowserFetcher`` uses this to start Chrome on its first fetch.
    It calls this with the visibility flag and the claimed profile path.
    It starts Chrome, sets the page-load timeout and, when headless,
    removes the headless token from the user agent.

    Args:
        visible (bool): True shows a window; False runs headless.
        profile_dir (str): the Chrome ``--user-data-dir`` to use.

    Returns:
        webdriver.Chrome: a ready driver.
    """
    driver = _start_with_mask(create_chrome_options(visible, profile_dir))
    try:
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_S)
        if not visible:
            _unmask_headless_user_agent(driver)
    except WebDriverException as error:
        driver.quit()
        raise FetchError(f"Chrome started but setup failed: "
                         f"{error.msg or error}") from error
    return driver


class BrowserFetcher:
    """Fetch pages through a real Chrome, rendering JS before extraction.

    The driver is created lazily on the first ``fetch_page`` call so that
    constructing a fetcher (or importing the TUI) never launches Chrome.
    Call ``close`` when done; it is safe to call more than once.
    """

    def __init__(self, visible: bool = False,
                 profile: ChromeProfile | None = None):
        """Store the settings; no browser starts until the first fetch.

        ``make_fetcher`` and tests make fetchers with this.
        They give the visibility flag and, in tests, a profile object.
        It only stores the values.

        Args:
            visible (bool): True shows a window; False runs headless.
            profile (ChromeProfile | None): the Chrome profile to use, or
                None for the shared persistent profile.
        """
        self.visible = visible
        self.profile = profile or ChromeProfile()
        self.driver = None
        self.last_url = None

    def _ensure_driver(self) -> webdriver.Chrome:
        """Start the driver on first use and reuse it afterwards."""
        if self.driver is None:
            self.driver = create_driver(self.visible, self.profile.claim())
        return self.driver

    def _wait_until_ready(self, driver: webdriver.Chrome) -> None:
        """Block until ``document.readyState`` is complete, then let JS settle."""
        WebDriverWait(driver, READY_STATE_TIMEOUT_S).until(
            lambda d: d.execute_script(_READY_SCRIPT) == _READY_COMPLETE)
        time.sleep(JS_SETTLE_S)

    def _wait_past_challenge(self, driver: webdriver.Chrome) -> str:
        """Wait while the page is a bot check; return the final HTML.

        Automatic checks clear by themselves; in a visible window the user
        can solve a CAPTCHA. ``fetch_page`` calls this after the page load.
        It polls the page until ``block_reason`` finds no bot check or the
        wait ends, then waits for the new page to load.

        Args:
            driver (webdriver.Chrome): the driver showing the loaded page.

        Returns:
            str: the page HTML; still a bot check if it never cleared.
        """
        html = driver.page_source
        wait_s = CHALLENGE_WAIT_VISIBLE_S if self.visible \
            else CHALLENGE_WAIT_HEADLESS_S
        deadline = time.monotonic() + wait_s
        while block_reason(html) and time.monotonic() < deadline:
            time.sleep(CHALLENGE_POLL_S)
            html = driver.page_source
            if block_reason(html) is None:
                self._wait_until_ready(driver)
                html = driver.page_source
        return html

    def fetch_page(self, url: str) -> PageText:
        """Load ``url`` in Chrome and return its extracted ``PageText``.

        Any browser-side failure (bad URL, load timeout, driver crash) is
        re-raised as ``FetchError`` so the vertical's existing error path
        handles it exactly like an HTTP failure. A bot check that does not
        clear raises ``BlockedPage``.
        """
        self.last_url = url
        driver = self._ensure_driver()
        try:
            driver.get(url)
            self._wait_until_ready(driver)
            html = self._wait_past_challenge(driver)
        except WebDriverException as error:
            self._discard_dead_driver()
            raise FetchError(f"{url}: {error.msg or error}") from error
        raise_if_blocked(html, url)
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
        """Quit the driver and free the profile; safe to call repeatedly."""
        if self.driver is not None:
            try:
                self.driver.quit()
            except WebDriverException:
                pass
            self.driver = None
        self.profile.release()


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

    from threetoks.web.errors import BlockedPage

    CHALLENGE_WAIT_HEADLESS_S = 0  # the check never clears: no waiting
    walled_fetcher = make_fetcher()
    walled = _FakeDriver()
    walled.page_source = "<title>Just a moment...</title><p>wait</p>"
    walled_fetcher.driver = walled
    walled_fetcher._wait_until_ready = lambda _driver: None
    try:
        walled_fetcher.fetch_page("https://walled.example/")
        raise AssertionError("expected BlockedPage from a challenge page")
    except BlockedPage as error:
        assert "walled.example" in str(error), error
    print("smoke OK")
