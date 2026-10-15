"""Offline tests for the Selenium browser fetcher (fake driver, no Chrome).

No browser launches: ``create_driver`` is monkeypatched to return a fake
driver object, so these tests assert wiring — lazy creation, the
``html_to_page`` handoff, idempotent close, ``FetchError`` translation,
stale-PATH-chromedriver masking, dead-session recovery, bot-check waits
and the headless user-agent fix — without any Chrome, chromedriver, or
network. The shared Chrome profile is redirected to a temporary cache
directory so no test touches the user's real profile.
"""
import os
import tempfile
import unittest
from unittest import mock

from selenium.common.exceptions import WebDriverException

from threetoks.web import browser
from threetoks.web.chrome_profile import ChromeProfile
from threetoks.web.errors import BlockedPage, FetchError
from threetoks.web.textify import PageText

_PAGE_HTML = ("<html><title>Demo</title><body>"
              "<p>The capital of France is Paris today.</p></body></html>")
_CHALLENGE_HTML = ("<html><head><title>Just a moment...</title></head><body>"
                   "<script>window._cf_chl_opt={}</script></body></html>")
_CACHE_ROOT = None
_CACHE_PATCH = None
_OPEN_FETCHERS = []


def setUpModule():
    global _CACHE_ROOT, _CACHE_PATCH
    _CACHE_ROOT = tempfile.TemporaryDirectory(prefix="threetoks_test_cache_")
    _CACHE_PATCH = mock.patch.dict(os.environ,
                                   {"XDG_CACHE_HOME": _CACHE_ROOT.name})
    _CACHE_PATCH.start()


def tearDownModule():
    for fetcher in _OPEN_FETCHERS:
        fetcher.close()
    _CACHE_PATCH.stop()
    _CACHE_ROOT.cleanup()


def make_fetcher(visible: bool = False) -> browser.BrowserFetcher:
    """``browser.make_fetcher`` whose profile lock is freed after the tests."""
    fetcher = browser.make_fetcher(visible=visible)
    _OPEN_FETCHERS.append(fetcher)
    return fetcher


class FakeDriver:
    """Records get/quit calls and returns canned HTML as ``page_source``."""

    window_handles = ["w0"]  # a live session answers the liveness probe

    def __init__(self, page_source: str = _PAGE_HTML):
        self.page_source = page_source
        self.got = []
        self.quit_count = 0
        self.page_load_timeout = None
        self.cdp_calls = []

    def get(self, url: str) -> None:
        self.got.append(url)

    def execute_script(self, _script: str) -> str:
        return "complete"

    def set_page_load_timeout(self, seconds: int) -> None:
        self.page_load_timeout = seconds

    def execute_cdp_cmd(self, command: str, params: dict) -> None:
        self.cdp_calls.append((command, params))

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
        fetcher = make_fetcher()
        self.assertIsNone(fetcher.driver)

    def test_driver_created_on_first_fetch(self):
        fake = FakeDriver()
        with mock.patch.object(browser, "create_driver",
                               return_value=fake) as created:
            fetcher = make_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        created.assert_called_once_with(False, mock.ANY)
        self.assertIs(fetcher.driver, fake)

    def test_driver_reused_across_fetches(self):
        fake = FakeDriver()
        with mock.patch.object(browser, "create_driver",
                               return_value=fake) as created:
            fetcher = make_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://a.example/")
            fetcher.fetch_page("https://b.example/")
        created.assert_called_once()
        self.assertEqual(fake.got, ["https://a.example/", "https://b.example/"])

    def test_visible_flag_passed_to_create_driver(self):
        fake = FakeDriver()
        with mock.patch.object(browser, "create_driver",
                               return_value=fake) as created:
            fetcher = make_fetcher(visible=True)
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        created.assert_called_once_with(True, mock.ANY)


class FetchHandoffTest(unittest.TestCase):
    def test_html_to_page_called_with_page_source_and_url(self):
        fake = FakeDriver()
        with mock.patch.object(browser, "create_driver", return_value=fake), \
                mock.patch.object(browser, "html_to_page") as converter:
            converter.return_value = PageText("Demo")
            fetcher = make_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        converter.assert_called_once_with(_PAGE_HTML, "https://example.com/")

    def test_returns_extracted_page_text(self):
        fake = FakeDriver()
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            page = fetcher.fetch_page("https://example.com/")
        self.assertIsInstance(page, PageText)
        self.assertEqual(page.title, "Demo")
        self.assertTrue(any("Paris" in s for s in page.sentences))

    def test_last_url_is_recorded(self):
        fake = FakeDriver()
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/page")
        self.assertEqual(fetcher.last_url, "https://example.com/page")

    def test_last_url_recorded_even_when_load_fails(self):
        fake = FakeDriver()
        fake.get = mock.Mock(side_effect=WebDriverException("boom"))
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            with self.assertRaises(FetchError):
                fetcher.fetch_page("https://example.com/broken")
        self.assertEqual(fetcher.last_url, "https://example.com/broken")


class ErrorTranslationTest(unittest.TestCase):
    def test_driver_get_exception_becomes_fetch_error(self):
        fake = FakeDriver()
        fake.get = mock.Mock(side_effect=WebDriverException("navigation failed"))
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            with self.assertRaises(FetchError) as caught:
                fetcher.fetch_page("https://example.com/")
        self.assertIn("https://example.com/", str(caught.exception))

    def test_ready_wait_exception_becomes_fetch_error(self):
        fake = FakeDriver()
        fake.execute_script = mock.Mock(
            side_effect=WebDriverException("script died"))
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            with self.assertRaises(FetchError):
                fetcher.fetch_page("https://example.com/")

    def test_driver_start_failure_raises_fetch_error(self):
        boom = WebDriverException("chromedriver unusable")
        with mock.patch.object(browser.webdriver, "Chrome", side_effect=boom):
            with self.assertRaises(FetchError) as caught:
                browser.create_driver(visible=False, profile_dir="/tmp/p")
        self.assertIn("chromedriver", str(caught.exception))


class CloseTest(unittest.TestCase):
    def test_close_quits_started_driver(self):
        fake = FakeDriver()
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        fetcher.close()
        self.assertEqual(fake.quit_count, 1)
        self.assertIsNone(fetcher.driver)

    def test_close_is_idempotent(self):
        fake = FakeDriver()
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        fetcher.close()
        fetcher.close()  # must not raise or double-quit
        self.assertEqual(fake.quit_count, 1)

    def test_close_before_any_fetch_is_safe(self):
        fetcher = make_fetcher()
        fetcher.close()  # driver never started

    def test_close_swallows_driver_quit_errors(self):
        fake = FakeDriver()
        fake.quit = mock.Mock(side_effect=WebDriverException("already dead"))
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            fetcher._wait_until_ready = lambda _driver: None
            fetcher.fetch_page("https://example.com/")
        fetcher.close()  # error is swallowed
        self.assertIsNone(fetcher.driver)


class DeadDriverRecoveryTest(unittest.TestCase):
    def test_dead_session_discarded_after_fetch_failure(self):
        dead = DeadSessionDriver()
        with mock.patch.object(browser, "create_driver", return_value=dead):
            fetcher = make_fetcher()
            with self.assertRaises(FetchError):
                fetcher.fetch_page("https://example.com/")
        self.assertIsNone(fetcher.driver)  # next fetch starts a fresh browser
        self.assertEqual(dead.quit_count, 1)

    def test_live_driver_kept_after_page_failure(self):
        fake = FakeDriver()
        fake.get = mock.Mock(side_effect=WebDriverException("bad page"))
        with mock.patch.object(browser, "create_driver", return_value=fake):
            fetcher = make_fetcher()
            with self.assertRaises(FetchError):
                fetcher.fetch_page("https://example.com/")
        self.assertIs(fetcher.driver, fake)  # the window stays open
        self.assertEqual(fake.quit_count, 0)


class BinaryMajorVersionTest(unittest.TestCase):
    def test_major_parsed_from_version_output(self):
        proc = mock.Mock(stdout="ChromeDriver 139.0.7258.5 (fbf9452ec)")
        with mock.patch.object(browser.subprocess, "run", return_value=proc):
            self.assertEqual(
                browser._binary_major_version("/x/chromedriver"), 139)

    def test_chrome_beta_output_parses(self):
        proc = mock.Mock(stdout="Google Chrome 151.0.7922.19 beta")
        with mock.patch.object(browser.subprocess, "run", return_value=proc):
            self.assertEqual(browser._binary_major_version("/x/chrome"), 151)

    def test_unparseable_output_gives_none(self):
        proc = mock.Mock(stdout="no digits here")
        with mock.patch.object(browser.subprocess, "run", return_value=proc):
            self.assertIsNone(browser._binary_major_version("/x/chromedriver"))

    def test_spawn_failure_gives_none(self):
        with mock.patch.object(browser.subprocess, "run",
                               side_effect=OSError("gone")):
            self.assertIsNone(browser._binary_major_version("/missing"))


class PathMaskTest(unittest.TestCase):
    def test_none_when_no_chromedriver_on_path(self):
        with tempfile.TemporaryDirectory() as empty:
            with mock.patch.dict(browser.os.environ, {"PATH": empty}):
                self.assertIsNone(browser._path_without_chromedriver())

    def test_dirs_holding_chromedriver_are_dropped(self):
        with tempfile.TemporaryDirectory() as with_driver, \
                tempfile.TemporaryDirectory() as clean:
            driver = os.path.join(with_driver, "chromedriver")
            with open(driver, "w") as handle:
                handle.write("#!/bin/sh\n")
            os.chmod(driver, 0o755)
            joined = os.pathsep.join([with_driver, clean])
            with mock.patch.dict(browser.os.environ, {"PATH": joined}):
                self.assertEqual(browser._path_without_chromedriver(), clean)


class StaleChromedriverTest(unittest.TestCase):
    def _check(self, which, driver_major, chrome_major):
        versions = {"/x/chromedriver": driver_major, "/x/chrome": chrome_major}
        with mock.patch.object(browser.shutil, "which", return_value=which), \
                mock.patch.object(browser, "_binary_major_version",
                                  side_effect=versions.get):
            return browser._path_chromedriver_is_stale("/x/chrome")

    def test_differing_majors_are_stale(self):
        self.assertTrue(self._check("/x/chromedriver", 139, 151))

    def test_matching_majors_are_compatible(self):
        self.assertFalse(self._check("/x/chromedriver", 151, 151))

    def test_unknown_version_counts_compatible(self):
        self.assertFalse(self._check("/x/chromedriver", 139, None))

    def test_no_path_driver_is_not_stale(self):
        self.assertFalse(self._check(None, 139, 151))


class CreateDriverMaskTest(unittest.TestCase):
    """``create_driver`` masks a stale PATH chromedriver during startup."""

    def _create(self, stale: bool, masked, chrome):
        """Run ``create_driver``; return (driver, PATH once startup is done)."""
        options = mock.Mock()
        options.binary_location = "/x/chrome"
        with mock.patch.object(browser, "create_chrome_options",
                               return_value=options), \
                mock.patch.object(browser, "_path_without_chromedriver",
                                  return_value=masked), \
                mock.patch.object(browser, "_path_chromedriver_is_stale",
                                  return_value=stale), \
                mock.patch.object(browser.webdriver, "Chrome",
                                  side_effect=chrome), \
                mock.patch.dict(browser.os.environ, {"PATH": "/original"}):
            driver = browser.create_driver(visible=True, profile_dir="/p")
            return driver, browser.os.environ["PATH"]

    def test_stale_path_driver_masked_during_start(self):
        seen = []

        def chrome(options=None):
            seen.append(browser.os.environ["PATH"])
            return FakeDriver()

        _driver, path_after = self._create(stale=True, masked="/masked",
                                           chrome=chrome)
        self.assertEqual(seen, ["/masked"])
        self.assertEqual(path_after, "/original")  # restored after startup

    def test_compatible_path_driver_left_alone(self):
        seen = []

        def chrome(options=None):
            seen.append(browser.os.environ["PATH"])
            return FakeDriver()

        self._create(stale=False, masked="/masked", chrome=chrome)
        self.assertEqual(seen, ["/original"])

    def test_unexplained_failure_retries_with_mask(self):
        seen = []

        def chrome(options=None):
            seen.append(browser.os.environ["PATH"])
            if len(seen) == 1:
                raise WebDriverException("boom")
            return FakeDriver()

        driver, path_after = self._create(stale=False, masked="/masked",
                                          chrome=chrome)
        self.assertEqual(seen, ["/original", "/masked"])
        self.assertIsInstance(driver, FakeDriver)
        self.assertEqual(path_after, "/original")

    def test_masked_first_failure_does_not_retry(self):
        chrome = mock.Mock(side_effect=WebDriverException("boom"))
        with self.assertRaises(FetchError):
            self._create(stale=True, masked="/masked", chrome=chrome)
        self.assertEqual(chrome.call_count, 1)

    def test_failure_with_no_path_driver_raises_once(self):
        chrome = mock.Mock(side_effect=WebDriverException("boom"))
        with self.assertRaises(FetchError):
            self._create(stale=False, masked=None, chrome=chrome)
        self.assertEqual(chrome.call_count, 1)


class ChromeOptionsTest(unittest.TestCase):
    """Options must not re-add the switches bot checks detect."""

    def _options(self, visible: bool):
        with mock.patch.object(browser, "get_chrome_path",
                               return_value="/x/chrome"):
            return browser.create_chrome_options(visible, "/profile")

    def test_automation_switches_hidden(self):
        options = self._options(visible=False)
        self.assertEqual(options.experimental_options["excludeSwitches"],
                         ["enable-automation"])
        self.assertIn("--disable-blink-features=AutomationControlled",
                      options.arguments)

    def test_gpu_stays_on_so_webgl_is_real(self):
        self.assertNotIn("--disable-gpu", self._options(False).arguments)

    def test_profile_dir_is_used(self):
        self.assertIn("--user-data-dir=/profile",
                      self._options(True).arguments)


class HeadlessUserAgentTest(unittest.TestCase):
    """Headless Chrome must not announce ``HeadlessChrome``."""

    _HINTS = {"brands": [{"brand": "Google Chrome", "version": "155"}],
              "platform": "macOS", "mobile": False}

    def _driver(self):
        fake = FakeDriver()
        fake.execute_script = mock.Mock(side_effect=lambda script: (
            "Mozilla/5.0 (Macintosh) HeadlessChrome/155.0.0.0 Safari/537.36"
            if script == browser._USER_AGENT_SCRIPT else self._HINTS))
        return fake

    def _create(self, visible: bool, fake):
        with mock.patch.object(browser, "create_chrome_options"), \
                mock.patch.object(browser, "_start_with_mask",
                                  return_value=fake):
            return browser.create_driver(visible, "/profile")

    def test_headless_token_replaced_and_hints_kept(self):
        fake = self._driver()
        self._create(visible=False, fake=fake)
        [(command, params)] = fake.cdp_calls
        self.assertEqual(command, "Network.setUserAgentOverride")
        self.assertNotIn("Headless", params["userAgent"])
        self.assertIn("Chrome/155.0.0.0", params["userAgent"])
        self.assertEqual(params["userAgentMetadata"], self._HINTS)

    def test_visible_window_keeps_its_own_user_agent(self):
        fake = self._driver()
        self._create(visible=True, fake=fake)
        self.assertEqual(fake.cdp_calls, [])

    def test_setup_failure_quits_chrome(self):
        fake = self._driver()
        fake.execute_cdp_cmd = mock.Mock(side_effect=WebDriverException("x"))
        with self.assertRaises(FetchError):
            self._create(visible=False, fake=fake)
        self.assertEqual(fake.quit_count, 1)


class SequenceDriver(FakeDriver):
    """Serves one ``page_source`` per read, repeating the last one."""

    def __init__(self, pages):
        super().__init__()
        self._pages = list(pages)

    @property
    def page_source(self):
        return self._pages.pop(0) if len(self._pages) > 1 else self._pages[0]

    @page_source.setter
    def page_source(self, _value):
        pass


class BotCheckTest(unittest.TestCase):
    """A bot check is waited out, or reported as ``BlockedPage``."""

    def _fetch(self, driver):
        fetcher = make_fetcher()
        fetcher.driver = driver
        fetcher._wait_until_ready = lambda _driver: None
        with mock.patch.object(browser.time, "sleep"):
            return fetcher, fetcher.fetch_page("https://walled.example/")

    def test_check_that_clears_returns_the_real_page(self):
        driver = SequenceDriver([_CHALLENGE_HTML, _CHALLENGE_HTML, _PAGE_HTML])
        _fetcher, page = self._fetch(driver)
        self.assertTrue(any("Paris" in s for s in page.sentences))

    def test_check_that_never_clears_raises_blocked_page(self):
        with mock.patch.object(browser, "CHALLENGE_WAIT_HEADLESS_S", 0):
            with self.assertRaises(BlockedPage) as caught:
                self._fetch(FakeDriver(_CHALLENGE_HTML))
        self.assertIn("walled.example", str(caught.exception))

    def test_blocked_page_keeps_the_browser_open(self):
        driver = FakeDriver(_CHALLENGE_HTML)
        fetcher = make_fetcher()
        fetcher.driver = driver
        fetcher._wait_until_ready = lambda _driver: None
        with mock.patch.object(browser, "CHALLENGE_WAIT_HEADLESS_S", 0), \
                self.assertRaises(BlockedPage):
            fetcher.fetch_page("https://walled.example/")
        self.assertIs(fetcher.driver, driver)
        self.assertEqual(driver.quit_count, 0)


class ProfileLifecycleTest(unittest.TestCase):
    """Closing the fetcher frees the shared profile for the next session."""

    def test_close_releases_the_profile_lock(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.object(browser, "create_driver",
                                  return_value=FakeDriver()):
            shared = os.path.join(root, "profile")
            first = browser.BrowserFetcher(profile=ChromeProfile(shared))
            first._wait_until_ready = lambda _driver: None
            first.fetch_page("https://example.com/")
            first.close()
            self.assertEqual(ChromeProfile(shared).claim(), shared)


class ChromePathTest(unittest.TestCase):
    def test_env_override_wins_when_runnable(self):
        with mock.patch.dict(browser.os.environ,
                             {browser.CHROME_ENV_VAR: "/custom/chrome"}), \
                mock.patch.object(browser.os.path, "exists",
                                  return_value=True), \
                mock.patch.object(browser.os, "access", return_value=True):
            self.assertEqual(browser.get_chrome_path(), "/custom/chrome")

    def test_missing_chrome_raises_fetch_error(self):
        with mock.patch.dict(browser.os.environ, {}, clear=True), \
                mock.patch.object(browser.os.path, "exists",
                                  return_value=False):
            with self.assertRaises(FetchError):
                browser.get_chrome_path()


if __name__ == "__main__":
    unittest.main()
