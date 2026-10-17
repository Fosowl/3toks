"""Offline tests for the code agent's four modes, end to end."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from threetoks.agents import code as code_agent
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.policy import Policy, PolicyConfig
from threetoks.services import Services

EDIT_MENU_KEYWORDS = ("edit — fix", "yes — the problem",
                      "replace the target")
BUGGY_ADD = "def add(a, b):\n    return a - b\n"
EDIT_REQUEST = "fix the bug in add in pkg/mod.py"


class KeywordBackend:
    """Answers menus by keyword; other prompts pop a completion queue.

    A menu with no keyword match gets an invalid reply, so tests see the
    harness's own fallback. The last completion repeats when exhausted.
    """

    def __init__(self, menu_keywords=(), completions=("unused",)):
        self.menu_keywords = list(menu_keywords)
        self.completions = list(completions)
        self.calls = 0
        self.menus = 0

    def complete(self, model, raw_prompt, opts):
        self.calls += 1
        if "ACTIONS:" in raw_prompt:
            self.menus += 1
            live_menu = raw_prompt[raw_prompt.rindex("ACTIONS:"):]
            return self._answer_menu(live_menu)
        text = self.completions.pop(0) if len(self.completions) > 1 \
            else self.completions[0]
        return GenResult(text, 8, 4, 0.0, "stop")

    def _answer_menu(self, menu):
        for keyword in self.menu_keywords:
            for line in menu.splitlines():
                if line[:1].isdigit() and " = " in line and keyword in line:
                    return GenResult(f" {line.split(' = ')[0]}",
                                     5, 1, 0.0, "stop")
        return GenResult(" x", 5, 1, 0.0, "stop")


def _policy(backend):
    return Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


def _corpus(test, tree):
    holder = tempfile.TemporaryDirectory()
    test.addCleanup(holder.cleanup)
    root = Path(holder.name)
    for rel, source in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source if isinstance(source, bytes)
                         else source.encode())
    return root


class AuthorModeTest(unittest.TestCase):
    def test_write_request_costs_one_menu_and_returns_the_module(self):
        backend = KeywordBackend(["author —"],
                                 ["square(n): multiply n by itself",
                                  "return n * n"])
        outcome = code_agent.run("write a function that squares a number",
                                 Services(files_root=_corpus(self, {})),
                                 _policy(backend))
        self.assertEqual((outcome["mode"], outcome["routed_by"]),
                         ("author", "menu"))
        self.assertIn("return n * n", outcome["answer"])
        self.assertNotIn("INCOMPLETE", outcome["answer"])

    def test_unwritten_function_is_named_in_a_note_before_the_code(self):
        backend = KeywordBackend(["author —"],
                                 ["square(n): multiply n by itself",
                                  "return((("])       # repeats forever
        outcome = code_agent.run("write a function that squares a number",
                                 Services(files_root=_corpus(self, {})),
                                 _policy(backend))
        first_line = outcome["answer"].splitlines()[0]
        self.assertEqual(first_line, code_agent.INCOMPLETE_NOTE.format(
            count=1, total=1, names="square"))
        self.assertEqual(outcome["stubbed"], ["square"])


class ComputeModeTest(unittest.TestCase):
    def test_compute_answers_with_the_echoed_return_value(self):
        backend = KeywordBackend(["compute —"],
                                 ["main(): return the answer",
                                  "return 42"])
        outcome = code_agent.run("sum the numbers in this data for me",
                                 Services(files_root=_corpus(self, {})),
                                 _policy(backend))
        self.assertEqual(outcome["mode"], "compute")
        self.assertEqual(outcome["answer"], "42")
        self.assertIn("return 42", outcome["script"])

    def test_failed_run_falls_back_to_the_error_and_the_script(self):
        backend = KeywordBackend(["compute —"],
                                 ["main(): divide by zero",
                                  "return 1 / 0"])
        outcome = code_agent.run("sum the numbers in this data for me",
                                 Services(files_root=_corpus(self, {})),
                                 _policy(backend))
        self.assertEqual(outcome["mode"], "compute")
        self.assertIn("did not produce output", outcome["answer"])
        self.assertIn(outcome["script"], outcome["answer"])

    def test_unwritten_function_is_named_instead_of_printed_nothing(self):
        # No entry can run (the only zero-argument function is not the
        # entry once a sibling is stubbed): the answer must say why.
        backend = KeywordBackend(["compute —"],
                                 ["load(): get the data\n2. total(items): "
                                  "sum the items",
                                  "return [1, 2]", "return((("])
        outcome = code_agent.run("sum the numbers in this data for me",
                                 Services(files_root=_corpus(self, {})),
                                 _policy(backend))
        self.assertEqual(outcome["stubbed"], ["total"])
        self.assertIn(code_agent.UNWRITTEN_REASON.format(names="total"),
                      outcome["answer"])
        self.assertFalse(outcome["run"]["ok"])


class NavigateModeTest(unittest.TestCase):
    def test_unique_symbol_answers_verbatim_with_zero_model_calls(self):
        root = _corpus(self, {"pkg/mod.py": (
            'def greet(name):\n    """Say hi."""\n    return f"hi {name}"\n')})
        backend = KeywordBackend()
        outcome = code_agent.run("where is the greet function defined?",
                                 Services(files_root=root), _policy(backend))
        self.assertEqual(outcome["mode"], "navigate")
        self.assertEqual(backend.calls, 0)          # fully free
        self.assertIn("pkg/mod.py:1", outcome["answer"])
        self.assertIn('return f"hi {name}"', outcome["answer"])

    def test_several_candidates_ask_one_menu_and_answer_the_pick(self):
        root = _corpus(self, {
            "a.py": "def read_row(line):\n    return line.split(',')\n",
            "b.py": "def read_header(line):\n    return line.upper()\n"})
        backend = KeywordBackend(["read_header"])
        outcome = code_agent.run(
            "where is the read function for a row or a header?",
            Services(files_root=root), _policy(backend))
        self.assertEqual(outcome["mode"], "navigate")
        self.assertEqual(backend.menus, 1)
        self.assertIn("b.py:1", outcome["answer"])
        self.assertIn("line.upper()", outcome["answer"])
        self.assertNotIn("line.split", outcome["answer"])

    def test_class_answer_lists_methods_not_bodies(self):
        root = _corpus(self, {"cache.py": (
            "class Cache:\n    \"\"\"Keep values.\"\"\"\n\n"
            "    def get(self, key):\n        return self.data[key]\n\n"
            "    def put(self, key, value):\n"
            "        self.data[key] = value\n")})
        outcome = code_agent.run("where is the Cache class implemented?",
                                 Services(files_root=root),
                                 _policy(KeywordBackend()))
        self.assertIn("class Cache:", outcome["answer"])
        self.assertIn("methods: get, put", outcome["answer"])
        self.assertNotIn("self.data[key] = value", outcome["answer"])

    def test_no_match_is_an_honest_miss(self):
        root = _corpus(self, {"pkg/mod.py": "def greet():\n    return 1\n"})
        outcome = code_agent.run("where is the frobnicate function defined?",
                                 Services(files_root=root),
                                 _policy(KeywordBackend()))
        self.assertEqual(outcome["mode"], "navigate")
        self.assertIn("no matching", outcome["answer"])

    def test_menu_fallback_says_the_request_was_not_classified(self):
        root = _corpus(self, {"mod.py": "def greet():\n    return 1\n"})
        outcome = code_agent.run("hmm, greet stuff",
                                 Services(files_root=root),
                                 _policy(KeywordBackend()))
        self.assertEqual(outcome["routed_by"], "menu-fallback")
        self.assertTrue(outcome["answer"].startswith(
            code_agent.MENU_FALLBACK_NOTE), outcome["answer"])
        self.assertIn("mod.py:1", outcome["answer"])


class EditModeTest(unittest.TestCase):
    def _run_edit(self, root, confirm_edit):
        backend = KeywordBackend(EDIT_MENU_KEYWORDS, ["return a + b"])
        services = Services(files_root=root, confirm_edit=confirm_edit)
        outcome = code_agent.run(EDIT_REQUEST, services, _policy(backend))
        self.assertEqual(outcome["mode"], "edit")
        self.assertTrue(outcome["success"], outcome)
        return outcome

    def test_without_confirm_callback_the_edit_is_not_applied(self):
        root = _corpus(self, {"pkg/mod.py": BUGGY_ADD})
        outcome = self._run_edit(root, None)
        self.assertFalse(outcome["applied"])
        self.assertIn("not applied", outcome["answer"])
        self.assertIn("+    return a + b", outcome["answer"])   # the diff
        self.assertEqual((root / "pkg" / "mod.py").read_text(), BUGGY_ADD)

    def test_confirmed_edit_is_applied_and_keeps_crlf(self):
        root = _corpus(self, {"pkg/mod.py": BUGGY_ADD.replace(
            "\n", "\r\n").encode()})
        asked = []
        outcome = self._run_edit(
            root, lambda rel, diff: asked.append((rel, diff)) or True)
        written = (root / "pkg" / "mod.py").read_bytes()
        self.assertTrue(outcome["applied"])
        self.assertEqual(asked[0][0], "pkg/mod.py")
        self.assertIn("-    return a - b", asked[0][1])
        self.assertIn(b"return a + b\r\n", written)
        self.assertEqual(written.count(b"\n"), written.count(b"\r\n"))

    def test_declined_edit_leaves_the_file_unchanged(self):
        root = _corpus(self, {"pkg/mod.py": BUGGY_ADD})
        outcome = self._run_edit(root, lambda rel, diff: False)
        self.assertFalse(outcome["applied"])
        self.assertIn(code_agent.DECLINED_REASON, outcome["answer"])
        self.assertEqual((root / "pkg" / "mod.py").read_text(), BUGGY_ADD)

    def test_file_changed_before_apply_is_not_overwritten(self):
        root = _corpus(self, {"pkg/mod.py": BUGGY_ADD})
        target = root / "pkg" / "mod.py"
        users_text = BUGGY_ADD + "\n\nUSER_EDIT = 1\n"

        def edit_then_accept(rel, diff):
            target.write_text(users_text)   # user edits during the question
            return True

        outcome = self._run_edit(root, edit_then_accept)
        self.assertFalse(outcome["applied"])
        self.assertIn(code_agent.CHANGED_REASON, outcome["answer"])
        self.assertEqual(target.read_text(), users_text)

    def test_undecodable_file_at_apply_time_is_not_overwritten(self):
        root = _corpus(self, {"pkg/mod.py": BUGGY_ADD})
        target = root / "pkg" / "mod.py"
        garbage = b"\xff\xfe\x00 not utf-8\n"

        def corrupt_then_accept(rel, diff):
            target.write_bytes(garbage)
            return True

        outcome = self._run_edit(root, corrupt_then_accept)
        self.assertFalse(outcome["applied"])
        self.assertIn(code_agent.UNDECODABLE_REASON, outcome["answer"])
        self.assertNotIn(code_agent.CHANGED_REASON, outcome["answer"])
        self.assertEqual(target.read_bytes(), garbage)

    def test_failed_edit_on_a_truncated_index_says_so(self):
        root = _corpus(self, {"pkg/mod.py": BUGGY_ADD})
        backend = KeywordBackend(EDIT_MENU_KEYWORDS, ["return a + b"])
        with mock.patch("threetoks.code.edit.index.MAX_INDEXED_FILES", 0):
            outcome = code_agent.run(EDIT_REQUEST, Services(files_root=root),
                                     _policy(backend))
        self.assertFalse(outcome["success"])
        self.assertIn("no edit made", outcome["answer"])
        self.assertIn(code_agent.TRUNCATED_NOTE, outcome["answer"])


if __name__ == "__main__":
    unittest.main()
