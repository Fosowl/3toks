"""Offline tests for the symbol index (walk, decoding, callees) and the
shortlist ranker (tokens, exact hits, BM25F ordering). No model."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from threetoks.code.edit import index as index_module
from threetoks.code.edit import nav, rank
from threetoks.code.edit.index import SymbolIndex


def _index(test_case, tree):
    holder = tempfile.TemporaryDirectory()
    test_case.addCleanup(holder.cleanup)
    root = Path(holder.name)
    for rel, content in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    return root, SymbolIndex(root)


def _big_function() -> str:
    lines = ["def process_everything(data):"]
    lines += [f"    value_{i} = data.get('key{i}')" for i in range(150)]
    lines += ["    config = data['config']", "    header = data['header']",
              "    path = data['file']", "    reader = open(path)",
              "    return config, header, reader"]
    return "\n".join(lines) + "\n"


class TokensTest(unittest.TestCase):
    def test_word_forms_share_a_stem(self):
        forms = [rank.tokens(word) for word in
                 ("parse", "parser", "parsing", "parsed", "parses")]
        self.assertTrue(all(form == forms[0] for form in forms), forms)

    def test_snake_and_camel_case_split_and_stopwords_drop(self):
        self.assertEqual(rank.tokens("parseHTTPHeader"),
                         rank.tokens("parse_http_header"))
        self.assertEqual(rank.tokens("fix the bug in the"), [])

    def test_words_keeps_only_the_requested_extra_characters(self):
        self.assertEqual(rank.words("Box.width(x_y)"), ["Box", "width", "x",
                                                        "y"])
        self.assertEqual(rank.words("Box.width(x_y)", keep="_."),
                         ["Box.width", "x_y"])


class ShortlistTest(unittest.TestCase):
    def test_small_precise_function_beats_a_big_scattered_one(self):
        _, index = _index(self, {
            "big.py": _big_function(),
            "small.py": ('def read_header(text):\n'
                         '    """Read the config file header."""\n'
                         '    return text.split("\\n")[0]\n'),
        })
        ranked = rank.shortlist(index, "reading the config file header fails")
        self.assertEqual(ranked[0].name, "read_header")

    def test_stemming_matches_the_parser_request_to_parse_line(self):
        _, index = _index(self, {"m.py": (
            "def parse_line(text):\n    return text.split()\n\n\n"
            "def format_line(words):\n    return ' '.join(words)\n")})
        ranked = rank.shortlist(index, "the parser rejects valid input")
        self.assertEqual([s.name for s in ranked], ["parse_line"])

    def test_exact_hit_comes_first(self):
        _, index = _index(self, {"m.py": (
            "def parse_config(text):\n"
            '    """Parse the config text."""\n    return text\n\n\n'
            "def helper(x):\n    return x\n")})
        ranked = rank.shortlist(index, "parse the config in helper")
        self.assertEqual([s.name for s in ranked], ["helper", "parse_config"])

    def test_class_name_ranks_its_methods_not_the_class(self):
        _, index = _index(self, {"pager.py": (
            "class Paginator:\n    def slice_page(self, n):\n"
            "        return n\n")})
        self.assertEqual(rank.exact_hits(index, "Paginator is broken"), [])
        ranked = rank.shortlist(index, "Paginator is broken")
        self.assertEqual([s.qualname for s in ranked], ["Paginator.slice_page"])
        with_classes = rank.shortlist(index, "Paginator is broken",
                                      kinds=("function", "method", "class"))
        self.assertEqual(with_classes[0].qualname, "Paginator")

    def test_dotted_name_selects_only_that_method(self):
        _, index = _index(self, {"m.py": (
            "class A:\n    def run(self):\n        return 1\n\n\n"
            "class B:\n    def run(self):\n        return 2\n")})
        self.assertEqual(len(rank.exact_hits(index, "fix run")), 2)
        hits = rank.exact_hits(index, "fix B.run() please")
        self.assertEqual([s.qualname for s in hits], ["B.run"])

    def test_limit_caps_the_list_beyond_one_menu_page(self):
        self.assertGreater(rank.SHORTLIST_SIZE, nav.PAGE_SIZE)
        tree = {f"m{i}.py": f"def load_item_{i}(x):\n    return x\n"
                for i in range(rank.SHORTLIST_SIZE + 4)}
        _, index = _index(self, tree)
        self.assertEqual(len(rank.shortlist(index, "load an item")),
                         rank.SHORTLIST_SIZE)

    def test_many_exact_hits_are_ranked_by_score(self):
        tree = {f"w{i}.py": (f"class Widget{i}:\n    def render(self):\n"
                             f"        return {i}\n")
                for i in range(nav.PAGE_SIZE + 2)}
        tree["w9.py"] = ("class Widget9:\n    def render(self):\n"
                         '        """Draw the border frame."""\n'
                         "        return 9\n")
        _, index = _index(self, tree)
        ranked = rank.shortlist(index, "render draws the border wrong")
        self.assertEqual(ranked[0].qualname, "Widget9.render")
        self.assertEqual(len(ranked), nav.PAGE_SIZE + 2)


class LabelTest(unittest.TestCase):
    def test_same_named_defs_get_distinct_labels_with_the_line(self):
        _, index = _index(self, {"box.py": (
            "class Box:\n    @property\n    def width(self):\n"
            "        return 1\n\n    @width.setter\n"
            "    def width(self, value):\n        pass\n")})
        labels = [nav.symbol_label(s) for s in index.symbols
                  if s.kind == "method"]
        self.assertEqual(labels, ["Box.width — box.py:3",
                                  "Box.width — box.py:7"])

    def test_deep_path_keeps_the_file_name_and_kind_prefix(self):
        _, index = _index(self, {"a/" * 30 + "money.py":
                                 "class Money:\n    pass\n"})
        label = nav.symbol_label(index.symbols[0], with_kind=True)
        self.assertTrue(label.startswith("class Money — "), label)
        self.assertTrue(label.endswith("money.py:1"), label)
        self.assertLessEqual(len(label), nav.LABEL_CLIP)

    def test_clip_lines_counts_the_hidden_lines(self):
        self.assertEqual(nav.clip_lines(["a", "b", "c"], 2), (["a", "b"], 1))
        self.assertEqual(nav.clip_lines(["a"], 5), (["a"], 0))


class IndexWalkTest(unittest.TestCase):
    def test_junk_directories_are_not_indexed(self):
        _, index = _index(self, {
            ".venv/lib/site.py": "def hidden():\n    pass\n",
            "node_modules/pkg/tool.py": "def vendored():\n    pass\n",
            "__pycache__/x.py": "def cached():\n    pass\n",
            "real/z.py": "def kept():\n    pass\n",
        })
        self.assertEqual(index.files, ["real/z.py"])
        self.assertEqual([s.name for s in index.symbols], ["kept"])

    @unittest.skipUnless(hasattr(os, "symlink"), "needs symlinks")
    def test_symlinked_directory_is_not_followed(self):
        root, _ = _index(self, {"real/z.py": "def kept():\n    pass\n"})
        (root / "link").symlink_to(root / "real", target_is_directory=True)
        self.assertEqual(SymbolIndex(root).files, ["real/z.py"])

    def test_bom_and_coding_cookie_files_are_indexed(self):
        _, index = _index(self, {
            "bom.py": b"\xef\xbb\xbfdef with_bom():\r\n    return 1\r\n",
            "latin.py": (b"# -*- coding: latin-1 -*-\n"
                         b"def cafe():\n    return 'caf\xe9'\n"),
        })
        names = {s.name for s in index.symbols}
        self.assertEqual(names, {"with_bom", "cafe"})
        self.assertIn("café", index.source_of("latin.py"))
        self.assertNotIn("\r", index.source_of("bom.py"))

    def test_undecodable_file_is_skipped_without_crashing(self):
        _, index = _index(self, {
            "bad.py": b"def bad():\n    return '\xff\xfe'\n",
            "good.py": "def good():\n    return 1\n",
        })
        self.assertEqual(index.files, ["good.py"])

    def test_source_is_cached_and_unknown_paths_give_empty_text(self):
        root, index = _index(self, {"m.py": "def f():\n    return 1\n"})
        (root / "m.py").unlink()
        self.assertIn("return 1", index.source_of("m.py"))
        self.assertEqual(index.source_of("missing.py"), "")

    def test_file_cap_sets_truncated(self):
        tree = {f"m{i}.py": "X = 1\n" for i in range(3)}
        with mock.patch.object(index_module, "MAX_INDEXED_FILES", 2):
            _, index = _index(self, tree)
        self.assertTrue(index.truncated)
        self.assertEqual(len(index.files), 2)


class CalleesTest(unittest.TestCase):
    def test_callees_resolve_same_file_first_then_unique_name(self):
        _, index = _index(self, {
            "a.py": ("def main():\n    helper()\n    shared()\n"
                     "    helper()\n    main()\n    ambiguous()\n\n\n"
                     "def helper():\n    return 1\n"),
            "b.py": ("def helper():\n    return 2\n\n\n"
                     "def shared():\n    return 3\n"),
            "c.py": "def ambiguous():\n    return 4\n",
            "d.py": "def ambiguous():\n    return 5\n",
        })
        main = index.symbol_at("a.py", "main")
        callees = [(s.file, s.qualname) for s in index.callees_of(main)]
        self.assertEqual(callees, [("a.py", "helper"), ("b.py", "shared")])

    def test_self_calls_resolve_to_the_own_class(self):
        _, index = _index(self, {"m.py": (
            "class A:\n    def run(self):\n        return self.step()\n\n"
            "    def step(self):\n        return 1\n\n\n"
            "class B:\n    def step(self):\n        return 2\n")})
        run = index.symbol_at("m.py", "A.run")
        self.assertEqual([s.qualname for s in index.callees_of(run)],
                         ["A.step"])


if __name__ == "__main__":
    unittest.main()
