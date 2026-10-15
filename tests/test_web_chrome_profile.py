"""Tests for the shared Chrome profile lock (real files, no Chrome)."""
import os
import tempfile
import unittest
from unittest import mock

from threetoks.web import chrome_profile
from threetoks.web.chrome_profile import ChromeProfile


class ClaimTest(unittest.TestCase):
    def setUp(self):
        self._root = tempfile.TemporaryDirectory()
        self.shared = os.path.join(self._root.name, "profile")

    def tearDown(self):
        self._root.cleanup()

    def test_free_shared_profile_is_used(self):
        profile = ChromeProfile(self.shared)
        self.assertEqual(profile.claim(), self.shared)
        profile.release()

    def test_claim_twice_returns_the_same_path(self):
        profile = ChromeProfile(self.shared)
        self.assertEqual(profile.claim(), profile.claim())
        profile.release()

    def test_busy_shared_profile_falls_back_to_a_temporary_one(self):
        owner = ChromeProfile(self.shared)
        owner.claim()
        second = ChromeProfile(self.shared)
        fallback = second.claim()
        self.assertNotEqual(fallback, self.shared)
        self.assertTrue(os.path.isdir(fallback))
        second.release()
        owner.release()

    def test_release_deletes_only_a_temporary_profile(self):
        owner = ChromeProfile(self.shared)
        owner.claim()
        second = ChromeProfile(self.shared)
        fallback = second.claim()
        second.release()
        owner.release()
        self.assertFalse(os.path.exists(fallback))
        self.assertTrue(os.path.isdir(self.shared))

    def test_release_frees_the_shared_profile(self):
        first = ChromeProfile(self.shared)
        first.claim()
        first.release()
        second = ChromeProfile(self.shared)
        self.assertEqual(second.claim(), self.shared)
        second.release()

    def test_release_without_claim_is_safe(self):
        ChromeProfile(self.shared).release()

    def test_unwritable_cache_falls_back_to_a_temporary_profile(self):
        blocker = os.path.join(self._root.name, "a-file")
        open(blocker, "w").close()
        profile = ChromeProfile(os.path.join(blocker, "profile"))
        path = profile.claim()
        self.assertTrue(os.path.isdir(path))
        profile.release()
        self.assertFalse(os.path.exists(path))


class DefaultDirTest(unittest.TestCase):
    def test_follows_xdg_cache_home(self):
        with mock.patch.dict(os.environ, {"XDG_CACHE_HOME": "/cache"}), \
                mock.patch.object(chrome_profile.os, "name", "posix"):
            self.assertEqual(chrome_profile.default_profile_dir(),
                             os.path.join("/cache", "threetoks",
                                          "chrome-profile"))


if __name__ == "__main__":
    unittest.main()
