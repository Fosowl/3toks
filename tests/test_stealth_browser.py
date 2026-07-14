"""Offline tests for the stealth browser fetcher (no Chrome, no network).

Two layers are covered without launching a browser:

  * Fetcher wiring (lazy driver, ``html_to_page`` handoff, ``FetchError``
    mapping, idempotent close) — via a fake driver injected onto the fetcher,
    exactly like ``test_web_browser``.
  * ``create_driver`` / ``create_chrome_options`` — via fake
    ``undetected_chromedriver`` and ``selenium_stealth`` modules pushed into
    ``sys.modules``, so the stealth stack is exercised with no real Chrome.
"""
import sys
import types
import unittest
from unittest import mock

from selenium.common.exceptions import WebDriverException

from threetoks.web import stealth_browser
from threetoks.web.fetch import FetchError
from threetoks.web.stealth_browser import make_stealth_fetcher
from threetoks.web.textify import PageText

_PAGE_HTML = ("<html><title>Demo</title><body>"
              "<p>The capital of France is Paris today.</p></body></html>")


class FakeDriver:
    """Records get/quit calls and returns canned HTML as ``page_source``."""

    window_handles = ["w0"]  # a live session answers the liveness probe

    def __init__(self, page_source: str = _PAGE_HTML):
        self.page_source = page_source
        self.got = []
        self.quit_count = 0

    def get(self, url: str) -> None:
        self.got.append(url)

    def execute_script(self, _script: str) -> str:
        return "complete"

    def set_page_load_timeout(self, _seconds: int) -> None:
        pass

    def quit(self) -> None:
        self.quit_count += 1


class DeadSessionDriver(FakeDriver):
    """A driver whose session is gone: navigation and probes both raise."""

    @property
    def window_handles(self):
        raise WebDriverException("invalid session id")

    def get(self, url: str) -> None:
        raise WebDriverException("disconnected")


class LazyCreationTest(unittest.TestCase):
    def test_no_driver_until_first_fetch(self):
        self.assertIsNone(make_stealth_fetcher().driver)

    def test_driver_created_on_first_fetch(self):
        fake = FakeDriver()
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake) as created:
            fetcher = make_stealth_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        created.assert_called_once_with(False)
        self.assertIs(fetcher.driver, fake)

    def test_single_driver_reused_across_fetches(self):
        fake = FakeDriver()
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake) as created:
            fetcher = make_stealth_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://a.example/")
            fetcher.fetch_page("https://b.example/")
        created.assert_called_once()
        self.assertEqual(fake.got,
                         ["https://a.example/", "https://b.example/"])

    def test_visible_flag_passed_to_create_driver(self):
        fake = FakeDriver()
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake) as created:
            fetcher = make_stealth_fetcher(visible=True)
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        created.assert_called_once_with(True)


class FetchHandoffTest(unittest.TestCase):
    def test_html_to_page_called_with_page_source_and_url(self):
        fake = FakeDriver()
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake), \
                mock.patch.object(stealth_browser, "html_to_page") as convert:
            convert.return_value = PageText("Demo")
            fetcher = make_stealth_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        convert.assert_called_once_with(_PAGE_HTML, "https://example.com/")

    def test_returns_extracted_page_text(self):
        fake = FakeDriver()
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake):
            fetcher = make_stealth_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            page = fetcher.fetch_page("https://example.com/")
        self.assertIsInstance(page, PageText)
        self.assertEqual(page.title, "Demo")
        self.assertTrue(any("Paris" in s for s in page.sentences))

    def test_last_url_recorded_even_when_load_fails(self):
        fake = FakeDriver()
        fake.get = mock.Mock(side_effect=WebDriverException("boom"))
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake):
            fetcher = make_stealth_fetcher()
            with self.assertRaises(FetchError):
                fetcher.fetch_page("https://example.com/broken")
        self.assertEqual(fetcher.last_url, "https://example.com/broken")


class ErrorTranslationTest(unittest.TestCase):
    def test_driver_get_exception_becomes_fetch_error(self):
        fake = FakeDriver()
        fake.get = mock.Mock(side_effect=WebDriverException("nav failed"))
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake):
            fetcher = make_stealth_fetcher()
            with self.assertRaises(FetchError) as caught:
                fetcher.fetch_page("https://example.com/")
        self.assertIn("https://example.com/", str(caught.exception))

    def test_ready_wait_exception_becomes_fetch_error(self):
        fake = FakeDriver()
        fake.execute_script = mock.Mock(
            side_effect=WebDriverException("script died"))
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake):
            fetcher = make_stealth_fetcher()
            with self.assertRaises(FetchError):
                fetcher.fetch_page("https://example.com/")


class DeadDriverRecoveryTest(unittest.TestCase):
    def test_dead_session_discarded_after_fetch_failure(self):
        dead = DeadSessionDriver()
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=dead):
            fetcher = make_stealth_fetcher()
            with self.assertRaises(FetchError):
                fetcher.fetch_page("https://example.com/")
        self.assertIsNone(fetcher.driver)  # next fetch starts a fresh browser
        self.assertEqual(dead.quit_count, 1)

    def test_live_driver_kept_after_page_failure(self):
        fake = FakeDriver()
        fake.get = mock.Mock(side_effect=WebDriverException("bad page"))
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake):
            fetcher = make_stealth_fetcher()
            with self.assertRaises(FetchError):
                fetcher.fetch_page("https://example.com/")
        self.assertIs(fetcher.driver, fake)  # the window stays open
        self.assertEqual(fake.quit_count, 0)


class CloseTest(unittest.TestCase):
    def _fetched(self):
        """Return a fetcher whose (fake) driver has been started once."""
        fake = FakeDriver()
        with mock.patch.object(stealth_browser, "create_driver",
                               return_value=fake):
            fetcher = make_stealth_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        return fetcher, fake

    def test_close_quits_started_driver(self):
        fetcher, fake = self._fetched()
        fetcher.close()
        self.assertEqual(fake.quit_count, 1)
        self.assertIsNone(fetcher.driver)

    def test_close_is_idempotent(self):
        fetcher, fake = self._fetched()
        fetcher.close()
        fetcher.close()
        self.assertEqual(fake.quit_count, 1)

    def test_close_before_any_fetch_is_safe(self):
        make_stealth_fetcher().close()  # driver never started

    def test_close_swallows_driver_quit_errors(self):
        fetcher, fake = self._fetched()
        fake.quit = mock.Mock(side_effect=WebDriverException("already dead"))
        fetcher.close()
        self.assertIsNone(fetcher.driver)


class _FakeUcChrome:
    """Fake ``uc.Chrome`` capturing the options it was built with."""

    def __init__(self, options=None):
        self.options = options
        self.scripts = []
        self.timeout = None

    def set_page_load_timeout(self, seconds):
        self.timeout = seconds

    def execute_script(self, script):
        self.scripts.append(script)


class _FakeChromeOptions:
    """Minimal ChromeOptions double recording added arguments."""

    def __init__(self):
        self.arguments = []
        self.binary_location = None

    def add_argument(self, arg):
        self.arguments.append(arg)


def _install_fake_uc(chrome_factory):
    """Return a fake ``undetected_chromedriver`` module."""
    module = types.ModuleType("undetected_chromedriver")
    module.__version__ = "0-fake"
    module.Chrome = chrome_factory
    module.ChromeOptions = _FakeChromeOptions
    return module


def _install_fake_stealth(recorder):
    """Return a fake ``selenium_stealth`` module recording stealth() calls."""
    module = types.ModuleType("selenium_stealth")
    module.stealth = recorder
    return module


class DriverCreationTest(unittest.TestCase):
    """Exercise create_driver/create_chrome_options with fake stealth modules."""

    def setUp(self):
        self.chrome_calls = []
        self.stealth_calls = []
        uc = _install_fake_uc(self._make_chrome)
        stealth_mod = _install_fake_stealth(self._record_stealth)
        patcher = mock.patch.dict(sys.modules, {
            "undetected_chromedriver": uc,
            "selenium_stealth": stealth_mod})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chrome_path = mock.patch.object(
            stealth_browser, "get_chrome_path",
            return_value="/fake/chrome")
        self.chrome_path.start()
        self.addCleanup(self.chrome_path.stop)

    def _make_chrome(self, options=None):
        driver = _FakeUcChrome(options)
        self.chrome_calls.append(driver)
        return driver

    def _record_stealth(self, driver, **kwargs):
        self.stealth_calls.append((driver, kwargs))

    def test_create_driver_starts_uc_and_applies_stealth(self):
        driver = stealth_browser.create_driver(visible=False)
        self.assertIs(driver, self.chrome_calls[0])
        self.assertEqual(driver.timeout, stealth_browser.PAGE_LOAD_TIMEOUT_S)
        self.assertEqual(len(self.stealth_calls), 1)
        self.assertTrue(driver.scripts, "expected the webdriver-hide script")

    def test_headless_flag_present_only_when_hidden(self):
        stealth_browser.create_driver(visible=False)
        hidden_opts = self.chrome_calls[0].options
        self.assertIn("--headless=new", hidden_opts.arguments)
        stealth_browser.create_driver(visible=True)
        shown_opts = self.chrome_calls[1].options
        self.assertNotIn("--headless=new", shown_opts.arguments)

    def test_options_carry_user_agent_and_profile_dir(self):
        stealth_browser.create_driver(visible=False)
        args = self.chrome_calls[0].options.arguments
        self.assertTrue(any(a.startswith("user-agent=") for a in args))
        self.assertTrue(any(a.startswith("--user-data-dir=") for a in args))
        self.assertEqual(self.chrome_calls[0].options.binary_location,
                         "/fake/chrome")

    def test_uc_startup_failure_becomes_fetch_error(self):
        def boom(options=None):
            raise RuntimeError("driver exploded")
        sys.modules["undetected_chromedriver"].Chrome = boom
        with self.assertRaises(FetchError) as caught:
            stealth_browser.create_driver(visible=False)
        self.assertIn("undetected", str(caught.exception))


class ChromePathTest(unittest.TestCase):
    def test_env_override_wins_when_runnable(self):
        with mock.patch.dict(stealth_browser.os.environ,
                             {stealth_browser.CHROME_ENV_VAR: "/custom/chrome"}), \
                mock.patch.object(stealth_browser.os.path, "exists",
                                  return_value=True), \
                mock.patch.object(stealth_browser.os, "access",
                                  return_value=True):
            self.assertEqual(stealth_browser.get_chrome_path(),
                             "/custom/chrome")

    def test_missing_chrome_raises_fetch_error(self):
        with mock.patch.dict(stealth_browser.os.environ, {}, clear=True), \
                mock.patch.object(stealth_browser.os.path, "exists",
                                  return_value=False):
            with self.assertRaises(FetchError):
                stealth_browser.get_chrome_path()


if __name__ == "__main__":
    unittest.main()
