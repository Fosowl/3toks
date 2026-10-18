"""Tests for the in-place INI saver: it must keep what the user wrote."""
import configparser
import os
import tempfile
import unittest

from threetoks.config import ThreetoksConfig, load_config, save_config
from threetoks.fields import find_field, set_value
from threetoks.ini_edit import save_changes, update_ini

SAMPLE = """\
# my own header
[llm]
# why I use this model
model = old
host = http://h:1
custom_key = keep-me

[browser]
mode : http
"""


class UpdateIniTest(unittest.TestCase):
    def test_changes_one_value_and_keeps_every_other_line(self):
        out = update_ini(SAMPLE, {("llm", "model"): "new"})
        self.assertEqual(out, SAMPLE.replace("model = old", "model = new"))

    def test_colon_delimiter_is_kept(self):
        out = update_ini(SAMPLE, {("browser", "mode"): "plain"})
        self.assertIn("mode : plain", out)

    def test_same_key_in_another_section_is_not_touched(self):
        text = "[a]\nenabled = true\n[b]\nenabled = true\n"
        out = update_ini(text, {("b", "enabled"): "false"})
        self.assertEqual(out, "[a]\nenabled = true\n[b]\nenabled = false\n")

    def test_missing_key_lands_inside_its_section(self):
        out = update_ini(SAMPLE, {("llm", "vote_k"): "3"})
        lines = out.split("\n")
        self.assertEqual(lines[lines.index("custom_key = keep-me") + 1],
                         "vote_k = 3")

    def test_missing_section_is_appended_once(self):
        out = update_ini(SAMPLE, {("voice", "lang"): "fr",
                                  ("voice", "enabled"): "true"})
        self.assertEqual(out.count("[voice]"), 1)
        self.assertTrue(out.endswith("lang = fr\nenabled = true\n")
                        or out.endswith("enabled = true\nlang = fr\n"), out)

    def test_continuation_lines_of_a_changed_value_go_away(self):
        text = "[llm]\nmodel = old\n  second line\nhost = h\n"
        out = update_ini(text, {("llm", "model"): "new"})
        self.assertEqual(out, "[llm]\nmodel = new\nhost = h\n")

    def test_missing_keys_go_below_a_continued_value_in_order(self):
        text = "[llm]\nmodel = a\n  b\n\n[browser]\nmode = http\n"
        out = update_ini(text, {("llm", "vote_k"): "2", ("llm", "host"): "H"})
        self.assertEqual(parsed(out)["llm"]["model"], "a\nb")
        self.assertIn("  b\nvote_k = 2\nhost = H\n", out)


def parsed(text):
    """Parse like the loader does; raises when the file would be ignored."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text)
    return parser


class LikeConfigparserTest(unittest.TestCase):
    """Regression: every saved file must still load back."""

    def test_header_with_trailing_text_is_still_a_header(self):
        text = "[llm]\nmodel = a\n[browser]   # fetch\nmode = http\n"
        out = update_ini(text, {("browser", "mode"): "plain"})
        self.assertEqual(parsed(out)["browser"]["mode"], "plain")
        self.assertEqual(out.count("[browser]"), 1)

    def test_indented_first_key_is_a_key_not_a_continuation(self):
        out = update_ini("[llm]\n  model = old\nhost = h\n",
                         {("llm", "model"): "new"})
        self.assertEqual(parsed(out)["llm"]["model"], "new")

    def test_continuation_of_an_unchanged_key_survives(self):
        text = "[llm]\nmodel = old\n  second\n[browser]\nmode = http\n"
        out = update_ini(text, {("browser", "mode"): "plain"})
        self.assertIn("  second", out)
        self.assertEqual(parsed(out)["llm"]["model"], "old\nsecond")

    def test_crlf_files_stay_crlf(self):
        out = update_ini("[llm]\r\nmodel = old\r\n",
                         {("llm", "model"): "new", ("llm", "vote_k"): "2"})
        self.assertEqual(out, "[llm]\r\nmodel = new\r\nvote_k = 2\r\n")

    def test_section_names_are_case_sensitive_like_the_loader(self):
        out = update_ini("[LLM]\nmodel = old\n", {("llm", "model"): "new"})
        self.assertEqual(parsed(out)["LLM"]["model"], "old")
        self.assertEqual(parsed(out)["llm"]["model"], "new")


class SaveChangesTest(unittest.TestCase):
    def _tmp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return folder.name

    def test_saved_file_loads_back_and_keeps_comments(self):
        path = os.path.join(self._tmp(), "c.ini")
        save_config(ThreetoksConfig(), path)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("# tail note\n")
        new = set_value(load_config(path), find_field("llm.model"), "mine")
        save_changes(new, path)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertEqual(load_config(path), new)
        self.assertIn("# tail note", text)
        self.assertIn("# Model used for every decision node", text)

    def test_no_difference_does_not_rewrite_the_file(self):
        path = os.path.join(self._tmp(), "c.ini")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("[llm]\nmodel = x\n")
        before = os.stat(path).st_mtime_ns
        save_changes(load_config(path), path)
        self.assertEqual(os.stat(path).st_mtime_ns, before)

    def test_symlinked_config_is_edited_in_place_with_its_mode(self):
        folder = self._tmp()
        real = os.path.join(folder, "real.ini")
        link = os.path.join(folder, "link.ini")
        with open(real, "w", encoding="utf-8") as handle:
            handle.write("[llm]\nmodel = old\n")
        os.chmod(real, 0o644)
        os.symlink(real, link)
        save_changes(set_value(load_config(link), find_field("llm.model"),
                               "new"), link)
        self.assertTrue(os.path.islink(link))
        self.assertEqual(load_config(real).llm.model, "new")
        self.assertEqual(os.stat(real).st_mode & 0o777, 0o644)

    def test_missing_file_is_created_with_parent_folders(self):
        path = os.path.join(self._tmp(), "deep", "c.ini")
        new = set_value(ThreetoksConfig(), find_field("relay.pin"), 22)
        save_changes(new, path)
        self.assertEqual(load_config(path).relay.pin, 22)

    def test_a_value_the_loader_ignored_is_repaired_by_a_save(self):
        path = os.path.join(self._tmp(), "c.ini")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("[llm]\nvote_k = abc\nmodel = x\n")
        save_changes(set_value(load_config(path), find_field("llm.model"),
                               "y"), path)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "[llm]\nvote_k = 1\nmodel = y\n")


if __name__ == "__main__":
    unittest.main()
