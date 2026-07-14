"""Offline tests for the Selenium browser fetcher (fake driver, no Chrome).

No browser launches: ``create_driver`` is monkeypatched to return a fake
driver object, so these tests assert wiring — lazy creation, the
``html_to_page`` handoff, idempotent close, ``FetchError`` translation,
stale-PATH-chromedriver masking, and dead-session recovery — without any
Chrome, chromedriver, or network.
"""
import os
import tempfile
import unittest
from unittest import mock

from selenium.common.exceptions import WebDriverException

from threetoks.web import browser
from threetoks.web.browser import make_fetcher
from threetoks.web.fetch import FetchError
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
        self.page_load_timeout = None

    def get(self, url: str) -> None:
        self.got.append(url)

    def execute_script(self, _script: str) -> str:
        return "complete"

    def set_page_load_timeout(self, seconds: int) -> None:
        self.page_load_timeout = seconds

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
        created.assert_called_once_with(False)
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
        created.assert_called_once_with(True)


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
                browser.create_driver(visible=False)
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
            driver = browser.create_driver(visible=False)
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
