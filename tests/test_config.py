"""Offline tests for central configuration loading (no filesystem deps).

Covers pure defaults when no file exists, INI parsing with per-field
bad-value fallback, the ``$THREETOKS_CONFIG`` path override, the per-user
config location per platform, the cwd > user resolution order, and the
save/round-trip path used by the onboarding wizard. Temp files are written
under ``tempfile`` and cleaned up per test.
"""
import contextlib
import os
import tempfile
import unittest
from unittest import mock

from threetoks import config
from threetoks.config import (CONFIG_ENV_VAR, ThreetoksConfig, config_problem,
                             find_config_path, load_config, onboarding_target,
                             save_config, user_config_path)


def _write_ini(text: str) -> str:
    """Write ``text`` to a temp .ini file and return its path."""
    handle = tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


class DefaultsTest(unittest.TestCase):
    def test_missing_file_yields_pure_defaults(self):
        cfg = load_config("/nonexistent/does-not-exist.ini")
        self.assertIsInstance(cfg, ThreetoksConfig)
        self.assertEqual(cfg.llm.model, config.DEFAULT_MODEL)
        self.assertEqual(cfg.llm.host, config.DEFAULT_LLM_HOST)
        self.assertEqual(cfg.llm.vote_k, config.DEFAULT_VOTE_K)
        self.assertTrue(cfg.llm.send_seed)
        self.assertEqual(cfg.browser.mode, config.DEFAULT_BROWSER_MODE)
        self.assertFalse(cfg.browser.visible)
        self.assertEqual(cfg.search.searxng_url, config.DEFAULT_SEARXNG_URL)
        self.assertEqual(cfg.search.min_interval_s,
                         config.DEFAULT_MIN_INTERVAL_S)
        self.assertEqual(cfg.research.max_rounds, config.DEFAULT_MAX_ROUNDS)
        self.assertEqual(cfg.research.max_steps, config.DEFAULT_MAX_STEPS)
        self.assertEqual(cfg.files.root, config.DEFAULT_FILES_ROOT)

    def test_empty_env_and_no_cwd_file_are_safe(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(config.configparser.ConfigParser, "read",
                                  return_value=[]):
            cfg = load_config()
        self.assertEqual(cfg.browser.mode, "http")


class ParsingTest(unittest.TestCase):
    def test_full_ini_overrides_every_field(self):
        path = _write_ini(
            "[llm]\nmodel = my-model\nhost = http://h:1\nvote_k = 3\n"
            "send_seed = false\n"
            "[browser]\nmode = stealth\nvisible = true\n"
            "[search]\nsearxng_url = http://s:2\nmin_interval_s = 2.5\n"
            "[research]\nmax_rounds = 7\nmax_steps = 40\n"
            "[files]\nroot = /data\n")
        try:
            cfg = load_config(path)
        finally:
            os.unlink(path)
        self.assertEqual(cfg.llm.model, "my-model")
        self.assertEqual(cfg.llm.host, "http://h:1")
        self.assertEqual(cfg.llm.vote_k, 3)
        self.assertFalse(cfg.llm.send_seed)
        self.assertEqual(cfg.browser.mode, "stealth")
        self.assertTrue(cfg.browser.visible)
        self.assertEqual(cfg.search.searxng_url, "http://s:2")
        self.assertEqual(cfg.search.min_interval_s, 2.5)
        self.assertEqual(cfg.research.max_rounds, 7)
        self.assertEqual(cfg.research.max_steps, 40)
        self.assertEqual(cfg.files.root, "/data")

    def test_bad_int_falls_back_to_default(self):
        path = _write_ini("[llm]\nvote_k = not-a-number\n"
                          "[research]\nmax_rounds = twelve\n")
        try:
            cfg = load_config(path)
        finally:
            os.unlink(path)
        self.assertEqual(cfg.llm.vote_k, config.DEFAULT_VOTE_K)
        self.assertEqual(cfg.research.max_rounds, config.DEFAULT_MAX_ROUNDS)

    def test_bad_float_falls_back_to_default(self):
        path = _write_ini("[search]\nmin_interval_s = soon\n")
        try:
            cfg = load_config(path)
        finally:
            os.unlink(path)
        self.assertEqual(cfg.search.min_interval_s,
                         config.DEFAULT_MIN_INTERVAL_S)

    def test_unknown_browser_mode_falls_back_to_default(self):
        path = _write_ini("[browser]\nmode = teleport\n")
        try:
            cfg = load_config(path)
        finally:
            os.unlink(path)
        self.assertEqual(cfg.browser.mode, config.DEFAULT_BROWSER_MODE)

    def test_boolean_word_variants_parse(self):
        for word, expected in (("yes", True), ("ON", True), ("1", True),
                               ("no", False), ("off", False), ("0", False)):
            path = _write_ini(f"[browser]\nvisible = {word}\n")
            try:
                cfg = load_config(path)
            finally:
                os.unlink(path)
            self.assertEqual(cfg.browser.visible, expected, word)

    def test_bad_boolean_falls_back_to_default(self):
        path = _write_ini("[browser]\nvisible = maybe\n")
        try:
            cfg = load_config(path)
        finally:
            os.unlink(path)
        self.assertEqual(cfg.browser.visible, config.DEFAULT_BROWSER_VISIBLE)

    def test_literal_percent_in_value_never_raises(self):
        path = _write_ini("[search]\nsearxng_url = http://s/q?x=%20y\n"
                          "[files]\nroot = %APPDATA%\\threetoks\n")
        try:
            cfg = load_config(path)
        finally:
            os.unlink(path)
        self.assertEqual(cfg.search.searxng_url, "http://s/q?x=%20y")
        self.assertEqual(cfg.files.root, "%APPDATA%\\threetoks")

    def test_blank_value_falls_back_to_default(self):
        path = _write_ini("[llm]\nmodel =   \n")
        try:
            cfg = load_config(path)
        finally:
            os.unlink(path)
        self.assertEqual(cfg.llm.model, config.DEFAULT_MODEL)

    def test_partial_section_keeps_other_defaults(self):
        path = _write_ini("[browser]\nmode = plain\n")
        try:
            cfg = load_config(path)
        finally:
            os.unlink(path)
        self.assertEqual(cfg.browser.mode, "plain")
        self.assertEqual(cfg.llm.model, config.DEFAULT_MODEL)


class EnvOverrideTest(unittest.TestCase):
    def test_env_path_is_used_when_no_explicit_arg(self):
        path = _write_ini("[llm]\nmodel = env-model\n")
        try:
            with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: path}):
                cfg = load_config()
        finally:
            os.unlink(path)
        self.assertEqual(cfg.llm.model, "env-model")

    def test_explicit_arg_beats_env_path(self):
        env_path = _write_ini("[llm]\nmodel = env-model\n")
        arg_path = _write_ini("[llm]\nmodel = arg-model\n")
        try:
            with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: env_path}):
                cfg = load_config(arg_path)
        finally:
            os.unlink(env_path)
            os.unlink(arg_path)
        self.assertEqual(cfg.llm.model, "arg-model")


@contextlib.contextmanager
def _isolated_dirs():
    """A temp cwd plus a temp XDG config home, environment cleared.

    Yields the config-home directory so tests can drop a user config file
    into ``<home>/threetoks/config.ini`` without touching the real one.
    """
    with tempfile.TemporaryDirectory() as workdir, \
            tempfile.TemporaryDirectory() as confhome:
        previous = os.getcwd()
        os.chdir(workdir)
        try:
            with mock.patch.dict(os.environ,
                                 {"XDG_CONFIG_HOME": confhome,
                                  "APPDATA": confhome},  # nt hosts too
                                 clear=True):
                yield confhome
        finally:
            os.chdir(previous)


def _write_user_config(confhome: str, text: str) -> str:
    """Create ``<confhome>/threetoks/config.ini`` with ``text``."""
    folder = os.path.join(confhome, "threetoks")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "config.ini")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class UserPathTest(unittest.TestCase):
    def test_posix_defaults_to_home_dot_config(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            path = user_config_path("posix")
        expected = os.path.join(os.path.expanduser("~"), ".config",
                                "threetoks", "config.ini")
        self.assertEqual(path, expected)

    def test_xdg_config_home_wins_on_posix(self):
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": "/xdg"}):
            path = user_config_path("posix")
        self.assertEqual(path, os.path.join("/xdg", "threetoks", "config.ini"))

    def test_windows_uses_appdata(self):
        with mock.patch.dict(os.environ, {"APPDATA": "/roaming"}):
            path = user_config_path("nt")
        self.assertEqual(path,
                         os.path.join("/roaming", "threetoks", "config.ini"))

    def test_windows_without_appdata_falls_back_under_home(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            path = user_config_path("nt")
        expected = os.path.join(os.path.expanduser("~"), "AppData", "Roaming",
                                "threetoks", "config.ini")
        self.assertEqual(path, expected)


class ResolutionOrderTest(unittest.TestCase):
    def test_user_config_used_when_no_cwd_file(self):
        with _isolated_dirs() as confhome:
            _write_user_config(confhome, "[llm]\nmodel = user-model\n")
            cfg = load_config()
        self.assertEqual(cfg.llm.model, "user-model")

    def test_cwd_file_beats_user_config(self):
        with _isolated_dirs() as confhome:
            _write_user_config(confhome, "[llm]\nmodel = user-model\n")
            with open("config.ini", "w", encoding="utf-8") as handle:
                handle.write("[llm]\nmodel = cwd-model\n")
            cfg = load_config()
        self.assertEqual(cfg.llm.model, "cwd-model")

    def test_env_beats_cwd_and_user(self):
        env_file = _write_ini("[llm]\nmodel = env-model\n")
        try:
            with _isolated_dirs() as confhome:
                _write_user_config(confhome, "[llm]\nmodel = user-model\n")
                with open("config.ini", "w", encoding="utf-8") as handle:
                    handle.write("[llm]\nmodel = cwd-model\n")
                with mock.patch.dict(os.environ,
                                     {CONFIG_ENV_VAR: env_file}):
                    cfg = load_config()
        finally:
            os.unlink(env_file)
        self.assertEqual(cfg.llm.model, "env-model")


class FindConfigPathTest(unittest.TestCase):
    def test_none_when_no_file_exists_anywhere(self):
        with _isolated_dirs():
            self.assertIsNone(find_config_path())

    def test_finds_cwd_then_user_file(self):
        with _isolated_dirs() as confhome:
            user_file = _write_user_config(confhome, "[llm]\n")
            self.assertEqual(find_config_path(), user_file)
            with open("config.ini", "w", encoding="utf-8") as handle:
                handle.write("[llm]\n")
            self.assertEqual(find_config_path(), config.DEFAULT_CONFIG_NAME)

    def test_set_but_missing_env_path_reports_none(self):
        with _isolated_dirs() as confhome:
            _write_user_config(confhome, "[llm]\n")
            with mock.patch.dict(os.environ,
                                 {CONFIG_ENV_VAR: "/missing/tt.ini"}):
                self.assertIsNone(find_config_path())

    def test_existing_env_path_is_found(self):
        env_file = _write_ini("[llm]\n")
        try:
            with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: env_file}):
                self.assertEqual(find_config_path(), env_file)
        finally:
            os.unlink(env_file)


class SaveConfigTest(unittest.TestCase):
    def test_round_trip_preserves_every_field(self):
        source = _write_ini(
            "[llm]\nmodel = m\nhost = http://h:1\nvote_k = 4\n"
            "[browser]\nmode = stealth\nvisible = true\n"
            "[search]\nsearxng_url = http://s:2\nmin_interval_s = 2.5\n"
            "[research]\nmax_rounds = 7\nmax_steps = 40\n"
            "[files]\nroot = /data\n"
            "[memory]\npath = /tmp/mem.json\nenabled = false\n")
        try:
            original = load_config(source)
        finally:
            os.unlink(source)
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_config(original, os.path.join(tmp, "cfg.ini"))
            self.assertEqual(load_config(saved), original)

    def test_percent_heavy_values_survive_the_round_trip(self):
        wizardish = ThreetoksConfig(
            search=config.SearchConfig(searxng_url="http://s/q?x=%20%25"),
            files=config.FilesConfig(root="%APPDATA%\\threetoks"),
            memory=config.MemoryConfig(path="C:\\Users\\%u%\\mem.json"))
        with tempfile.TemporaryDirectory() as tmp:
            saved = save_config(wizardish, os.path.join(tmp, "cfg.ini"))
            self.assertEqual(load_config(saved), wizardish)

    def test_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "deep", "nested", "config.ini")
            self.assertEqual(save_config(ThreetoksConfig(), target), target)
            self.assertTrue(os.path.isfile(target))

    def test_default_target_is_user_path_and_env_overrides_it(self):
        with _isolated_dirs() as confhome:
            self.assertEqual(onboarding_target(), user_config_path())
            saved = save_config(ThreetoksConfig())
            self.assertTrue(saved.startswith(confhome))
            self.assertTrue(os.path.isfile(saved))
            env_target = os.path.join(confhome, "explicit.ini")
            with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: env_target}):
                self.assertEqual(onboarding_target(), env_target)
                self.assertEqual(save_config(ThreetoksConfig()), env_target)


class ConfigProblemTest(unittest.TestCase):
    """A file that parses to nothing must not fail silently."""

    def test_valid_file_and_absent_file_report_no_problem(self):
        path = _write_ini("[llm]\nmodel = mine\n")
        self.addCleanup(os.unlink, path)
        self.assertIsNone(config_problem(path))
        self.assertIsNone(config_problem("/nonexistent/threetoks.ini"))

    def test_comment_missing_its_hash_is_reported_not_swallowed(self):
        # exactly the real break: a leading '#' clobbered by an editor
        path = _write_ini("g ThreeToks configuration\n[llm]\nmodel = mine\n")
        self.addCleanup(os.unlink, path)
        self.assertEqual(load_config(path).llm.model, config.DEFAULT_MODEL)
        problem = config_problem(path)
        self.assertIsNotNone(problem)
        self.assertIn(path, problem)
        self.assertIn("fell back", problem)

    def test_message_is_one_line(self):
        path = _write_ini("junk\n[llm]\nmodel = mine\n")
        self.addCleanup(os.unlink, path)
        self.assertNotIn("\n", config_problem(path))

    def test_duplicate_section_is_reported(self):
        path = _write_ini("[llm]\nmodel = a\n[llm]\nmodel = b\n")
        self.addCleanup(os.unlink, path)
        self.assertIsNotNone(config_problem(path))

    def test_it_follows_the_same_resolution_order_as_load_config(self):
        path = _write_ini("g broken\n[llm]\nmodel = mine\n")
        self.addCleanup(os.unlink, path)
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: path}):
            self.assertIn(path, config_problem())  # no argument: resolves


if __name__ == "__main__":
    unittest.main()
