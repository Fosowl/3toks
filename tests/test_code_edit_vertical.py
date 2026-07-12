"""Offline end-to-end tests for the edit vertical's state machine.

A scripted backend answers menus by keyword-grepping the rendered options
(first matching keyword wins) and answers generations from a queue of
canned completions — the same pattern as tests/test_agents_core.py, no
model or network anywhere.
"""
import re
import tempfile
import unittest
from pathlib import Path

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code.edit.vertical import EditVertical
from threetoks.engine import run_episode
from threetoks.policy import Policy, PolicyConfig


class ScriptedBackend:
    """Keyword-matched menu picks + a queue of generation completions."""

    def __init__(self, menu_keywords=(), completions=()):
        self.menu_keywords = list(menu_keywords)
        self.completions = list(completions)
        self.menus_seen: list[str] = []
        self.generations = 0

    def complete(self, model, raw_prompt, opts):
        if "ACTIONS:" in raw_prompt:
            return self._answer_menu(raw_prompt)
        self.generations += 1
        completion = self.completions[0] if len(self.completions) == 1 \
            else self.completions.pop(0)
        return GenResult(completion, 10, 20, 0.0, "stop")

    def _answer_menu(self, raw_prompt):
        menu = raw_prompt[raw_prompt.rindex("ACTIONS:"):]
        self.menus_seen.append(menu)
        for keyword in self.menu_keywords:
            for line in menu.splitlines():
                if re.match(r"^\d+ = ", line) and keyword in line:
                    return GenResult(f" {line.split(' = ')[0]}", 10, 1,
                                     0.0, "stop")
        return GenResult(" 1", 10, 1, 0.0, "stop")


def _run(vertical, backend, max_steps=30):
    policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    return run_episode(vertical, policy, max_steps=max_steps)


def _corpus(tree):
    holder = tempfile.TemporaryDirectory()
    root = Path(holder.name)
    for rel, source in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return holder, root


class DirectReplaceTest(unittest.TestCase):
    def test_unique_symbol_replace_succeeds_verified(self):
        holder, root = _corpus(
            {"pkg/mod.py": "def add(a, b):\n    return a - b\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("fix the bug in add", root,
                                "from pkg.mod import add\nassert add(2, 2) == 4\n")
        backend = ScriptedBackend(["replace the target"], ["return a + b"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"])
        self.assertTrue(outcome["verified"])
        self.assertEqual(outcome["path_taken"], "direct")
        self.assertEqual(outcome["repairs"], 0)
        self.assertIn("a + b", (root / "pkg" / "mod.py").read_text())

    def test_passing_test_short_circuits_with_zero_model_calls(self):
        holder, root = _corpus(
            {"mod.py": "def add(a, b):\n    return a + b\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(1, 1) == 2\n")
        backend = ScriptedBackend()
        outcome = _run(vertical, backend)
        self.assertFalse(outcome["success"])
        self.assertIn("nothing to fix", outcome["reason"])
        self.assertEqual(backend.menus_seen, [])
        self.assertEqual(backend.generations, 0)


class OperationInferenceTest(unittest.TestCase):
    def test_undefined_helper_skips_the_operation_menu(self):
        # A backend rigged to always answer "replace" would sabotage this
        # fix if the operation menu were consulted; inference must bypass it.
        holder, root = _corpus({"mod.py": (
            "def average(nums):\n    total = 0\n"
            "    for n in nums:\n        total = _safe_add(total, n)\n"
            "    return total / len(nums)\n")})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("average crashes with a NameError", root,
                                "from mod import average\n"
                                "assert average([2, 4]) == 3\n")
        backend = ScriptedBackend(["replace the target"], ["return a + b"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "insert_after")
        self.assertEqual(backend.menus_seen, [])   # inference asked nothing


class DeleteStrikeOutTest(unittest.TestCase):
    def test_failed_delete_is_struck_from_the_retry_menu(self):
        holder, root = _corpus({"mod.py": (
            "def shout(s):\n    s = s.upper()\n    s = s.lower()\n"
            "    return s\n")})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("shout returns lowercase, remove the bad line",
                                root, "from mod import shout\n"
                                "assert shout('a') == 'A'\n")
        # First delete round: "upper" matches and is deleted (wrong, test
        # still fails). Second round: upper is struck out, "lower" matches.
        backend = ScriptedBackend(["delete the offending", "upper()",
                                   "lower()"], [])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "delete")
        self.assertEqual(outcome["repairs"], 1)
        self.assertEqual(backend.generations, 0)
        final = (root / "mod.py").read_text()
        self.assertIn("upper", final)
        self.assertNotIn("lower", final)


class OscillationEscalationTest(unittest.TestCase):
    def test_repeated_wrong_body_escalates_and_ends_bounded(self):
        holder, root = _corpus(
            {"mod.py": "def add(a, b):\n    return a - b\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")
        backend = ScriptedBackend(["replace the target"], ["return a * b"])
        outcome = _run(vertical, backend, max_steps=40)
        self.assertFalse(outcome["success"])
        # one real check burned, then the guard rejected every regeneration
        # of the same wrong body and escalation ran out of stages
        self.assertGreaterEqual(outcome["repairs"], 3)
        self.assertEqual((root / "mod.py").read_text(),
                         "def add(a, b):\n    return a - b\n")   # restored


class DisambiguationTest(unittest.TestCase):
    CORPUS = {
        "shop/orders.py": ('def validate(v):\n    """Check an order '
                           'total."""\n    return True\n'),
        "shop/users.py": ('def validate(v):\n    """Check a user email.'
                          '"""\n    return "@" in v\n'),
    }

    def test_keyword_overlap_free_takes_the_clear_winner(self):
        holder, root = _corpus(self.CORPUS)
        self.addCleanup(holder.cleanup)
        vertical = EditVertical(
            "validate lets zero-total orders through, fix it", root,
            "from shop.orders import validate\nassert validate(0) is False\n")
        # a rigged backend that would pick users.py if the menu were shown
        backend = ScriptedBackend(["users.py", "replace the target"],
                                  ["return v > 0"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target_file"], "shop/orders.py")
        menu_texts = "".join(backend.menus_seen)
        self.assertNotIn("users.py", menu_texts)   # no disambiguation menu

    def test_tie_falls_back_to_a_ranked_menu(self):
        holder, root = _corpus(self.CORPUS)
        self.addCleanup(holder.cleanup)
        vertical = EditVertical(
            "fix the validate bug", root,
            "from shop.users import validate\n"
            "assert validate('a@b') is False\n")   # no dot: must be rejected
        backend = ScriptedBackend(["users.py", "replace the target"],
                                  ["return '@' in v and '.' in v"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target_file"], "shop/users.py")
        self.assertTrue(any("users.py" in m for m in backend.menus_seen))

    def test_relocate_recovers_from_a_misleading_free_take(self):
        # The request keyword points at orders.py but the failing test
        # lives in users.py: escalation must eventually retarget.
        corpus = {
            "shop/orders.py": "def validate(v):\n    return v > 0\n",
            "shop/users.py": "def validate(v):\n    return True\n",
        }
        holder, root = _corpus(corpus)
        self.addCleanup(holder.cleanup)
        vertical = EditVertical(
            "validate lets bad orders through", root,
            "from shop.users import validate\nassert validate('') is False\n")
        backend = ScriptedBackend(["replace the target"], ["return bool(v)"])
        outcome = _run(vertical, backend, max_steps=40)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target_file"], "shop/users.py")
        self.assertGreaterEqual(outcome["repairs"], 1)


class NavigationTest(unittest.TestCase):
    def test_single_option_levels_resolve_free(self):
        holder, root = _corpus(
            {"only/mod.py": "def fix_me(x):\n    return x - 1\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("the increment is broken", root,
                                "from only.mod import fix_me\n"
                                "assert fix_me(1) == 2\n")
        backend = ScriptedBackend(["replace the target"], ["return x + 1"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["path_taken"], "navigate")
        # navigation was all free-takes; only the operation menu was shown
        self.assertEqual(len(backend.menus_seen), 1)


class NoTestModeTest(unittest.TestCase):
    def test_edit_without_a_test_gates_on_import_and_reports_unverified(self):
        holder, root = _corpus(
            {"mod.py": "def add(a, b):\n    return a - b\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("make add actually add", root, test_code=None)
        backend = ScriptedBackend(["replace the target"], ["return a + b"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"])
        self.assertFalse(outcome["verified"])
        self.assertIn("no test", outcome["reason"])
        self.assertIn("a + b", (root / "mod.py").read_text())


class BudgetTest(unittest.TestCase):
    def test_step_budget_forces_an_honest_failure(self):
        holder, root = _corpus(
            {"mod.py": "def add(a, b):\n    return a - b\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 2) == 4\n")
        backend = ScriptedBackend(["replace the target"], ["return a * b"])
        outcome = _run(vertical, backend, max_steps=2)
        self.assertFalse(outcome["success"])
        self.assertIn("budget", outcome["reason"])


if __name__ == "__main__":
    unittest.main()
