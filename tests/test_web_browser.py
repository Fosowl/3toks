"""Offline tests for the Selenium browser fetcher (fake driver, no Chrome).

No browser launches: ``create_driver`` is monkeypatched to return a fake
driver object, so these tests assert wiring — lazy creation, the
``html_to_page`` handoff, idempotent close, and ``FetchError`` translation —
without any Chrome, chromedriver, or network.
"""
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
