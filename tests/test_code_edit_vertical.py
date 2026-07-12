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
    def test_imported_names_are_never_inferred_as_missing_helpers(self):
        # Panel finding: a call to a `from helpers import ...` name must
        # not trigger insert-after inference (it would shadow the import);
        # the real fix here is a replace on the target's own bug.
        holder, root = _corpus({
            "helpers.py": "def normalize(text):\n    return text.strip()\n",
            "mod.py": ("from helpers import normalize\n\n\n"
                       "def clean(text):\n"
                       "    total = normalize(text)\n"
                       "    return total.upper()\n"),
        })
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("clean should lowercase, fix clean", root,
                                "from mod import clean\n"
                                "assert clean(' A ') == 'a'\n")
        backend = ScriptedBackend(["replace the target"],
                                  ["return normalize(text).lower()"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "replace")
        source = (root / "mod.py").read_text()
        self.assertEqual(source.count("def normalize"), 0)   # no shadowing

    def test_escalation_reaches_the_real_menu_after_inference_fails(self):
        # The re-ask stage must not just re-run the same deterministic
        # inference; scenario: inferred insert-after keeps failing, stage 1
        # must show the 3-option operation menu.
        holder, root = _corpus({"mod.py": (
            "def total(nums):\n    result = 0\n"
            "    for n in nums:\n        result = _acc(result, n)\n"
            "    return result\n")})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("total is broken", root,
                                "from mod import total\n"
                                "assert total([1, 2]) == 3\n")
        backend = ScriptedBackend(["replace the target"], ["return a - b"])
        _run(vertical, backend, max_steps=40)
        self.assertTrue(any("insert new helper code" in m
                            for m in backend.menus_seen), backend.menus_seen)

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
    def test_keyword_matched_line_is_deleted_free(self):
        # "lowercase" prefix-matches the .lower() statement and nothing
        # else scores, so the harness deletes the right line with no menu.
        holder, root = _corpus({"mod.py": (
            "def shout(s):\n    s = s.upper()\n    s = s.lower()\n"
            "    return s\n")})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("shout returns lowercase, remove the bad line",
                                root, "from mod import shout\n"
                                "assert shout('a') == 'A'\n")
        backend = ScriptedBackend(["delete the offending"], [])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "delete")
        self.assertEqual(outcome["repairs"], 0)
        self.assertEqual(backend.generations, 0)
        self.assertEqual(len(backend.menus_seen), 1)   # only the op menu
        self.assertNotIn("lower", (root / "mod.py").read_text())

    def test_failed_delete_is_struck_from_the_retry_menu(self):
        # No statement out-scores the others, so the menu shows; the first
        # (wrong) pick is struck out of the retry menu.
        holder, root = _corpus({"mod.py": (
            "def tidy(s):\n    s = s.strip()\n    s = s.swapcase()\n"
            "    return s\n")})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("tidy mangles what it is given, one of its "
                                "steps has to go", root,
                                "from mod import tidy\n"
                                "assert tidy(' abc ') == 'abc'\n")
        backend = ScriptedBackend(["delete the offending", "strip()",
                                   "swapcase()"], [])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "delete")
        self.assertEqual(outcome["repairs"], 1)
        self.assertEqual(backend.generations, 0)
        final = (root / "mod.py").read_text()
        self.assertIn("strip", final)
        self.assertNotIn("swapcase", final)


class StatelessGenerationTest(unittest.TestCase):
    def test_generation_prompts_never_carry_failure_feedback(self):
        # E5b: error-feedback repair underperforms blind resampling, so a
        # regeneration prompt must not contain the previous failure line —
        # while the re-asked operation MENU legitimately sees it.
        holder, root = _corpus(
            {"mod.py": "def add(a, b):\n    return a - b\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")

        generation_prompts = []

        class Recorder(ScriptedBackend):
            def complete(self, model, raw_prompt, opts):
                if "ACTIONS:" not in raw_prompt:
                    generation_prompts.append(raw_prompt)
                return super().complete(model, raw_prompt, opts)

        backend = Recorder(["replace the target"],
                           ["return a * b", "return a + b"])
        outcome = _run(vertical, backend, max_steps=40)
        self.assertTrue(outcome["success"], outcome)
        self.assertGreaterEqual(len(generation_prompts), 2)
        for prompt in generation_prompts:
            self.assertNotIn("edit failed", prompt)
            self.assertNotIn("> target:", prompt)
            self.assertIn("return a - b", prompt)   # verbatim span present
        # ...and the menu episode DID keep the failure line for menus
        self.assertIn("edit failed",
                      vertical._menu_episode.render_base())


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


class OscillationSeedTest(unittest.TestCase):
    def test_regenerating_an_annotated_original_verbatim_is_caught(self):
        # Bug-hunt finding: the seed hash was taken from verbatim file
        # text while candidates hash a canonical reconstruction, so any
        # type annotation defeated the guard. The model here regenerates
        # the exact original buggy body (sans docstring, as the prefill
        # never shows one) — it must be rejected without an oracle run.
        holder, root = _corpus({"mod.py": (
            "def add(a: int, b: int) -> int:\n"
            '    """Add two numbers."""\n'
            "    return a - b\n")})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")
        oracle_runs = []
        original_check = vertical._check_candidate
        vertical._check_candidate = lambda c: (oracle_runs.append(c),
                                               original_check(c))[1]
        backend = ScriptedBackend(["replace the target"], ["return a - b"])
        outcome = _run(vertical, backend, max_steps=40)
        self.assertFalse(outcome["success"])
        # the regenerated original bug never reached the oracle
        self.assertEqual(oracle_runs, [])


class MethodEscapeBacktrackTest(unittest.TestCase):
    def test_escaping_the_method_menu_reaches_a_sibling_function(self):
        # Bug-hunt finding: a method-menu escape used to strike out the
        # whole FILE, making the correct sibling def unreachable.
        holder, root = _corpus({"mod.py": (
            "class Widget:\n"
            "    def up(self):\n        return 1\n\n"
            "    def down(self):\n        return -1\n\n\n"
            "def bump(value):\n    return value - 1\n")})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("incrementing gives one less than expected",
                                root, "from mod import bump\n"
                                "assert bump(1) == 2\n")
        backend = ScriptedBackend(["Widget", "none of these fit", "bump",
                                   "replace the target"],
                                  ["return value + 1"])
        outcome = _run(vertical, backend, max_steps=40)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target"], "bump")


class ViableOperationsTest(unittest.TestCase):
    def test_insert_is_not_offered_without_a_missing_helper(self):
        holder, root = _corpus(
            {"mod.py": "def add(a, b):\n    return a - b\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")
        backend = ScriptedBackend(["insert new helper", "replace the target"],
                                  ["return a + b"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "replace")
        self.assertFalse(any("insert new helper" in m
                             for m in backend.menus_seen))

    def test_one_liner_target_offers_no_delete(self):
        # def and body share a line: deleting the sole statement would
        # erase the whole function, so delete must not be offered.
        holder, root = _corpus(
            {"mod.py": "def add(a, b): return a - b\n"})
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")
        backend = ScriptedBackend(["delete the offending",
                                   "replace the target"], ["return a + b"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "replace")
        self.assertFalse(any("delete the offending" in m
                             for m in backend.menus_seen))


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
        # navigation free-takes every level AND the operation menu (only
        # replace is viable for a helper-free two-liner): zero menus.
        self.assertEqual(len(backend.menus_seen), 0)


class RankedNavigationTest(unittest.TestCase):
    def test_keyword_overlap_navigates_with_zero_menus(self):
        # Request words match one file's def names/docstrings at every
        # level, so folder, file, and def all resolve free (E8 scenario 4
        # died on a bare-filename menu; enriched ranking removes the menu).
        holder, root = _corpus({
            "textutils/strings.py": (
                'def normalize_whitespace(text):\n'
                '    """Collapse runs of whitespace into single spaces."""\n'
                '    return text.replace("\\t", " ").strip()\n'),
            "textutils/wordcount.py": (
                'def unique_words(text):\n'
                '    """Count distinct words."""\n'
                '    return len(set(text.split()))\n'),
            "mathutils/rounding.py": (
                'def round_half_up(value):\n'
                '    """Round to nearest int."""\n'
                '    return int(value + 0.5)\n'),
        })
        self.addCleanup(holder.cleanup)
        vertical = EditVertical(
            "the whitespace cleanup keeps double spaces instead of "
            "collapsing them", root,
            "from textutils.strings import normalize_whitespace\n"
            "assert normalize_whitespace('a   b') == 'a b'\n")
        backend = ScriptedBackend(["replace the target"],
                                  ["import re\n    return re.sub(r'\\s+', "
                                   "' ', text).strip()"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["path_taken"], "navigate")
        self.assertEqual(len(backend.menus_seen), 1)   # only the op menu

    def test_escape_mid_navigation_fails_bounded_not_hanging(self):
        holder, root = _corpus({
            "pkg/alpha.py": "def one(x):\n    return x\n",
            "pkg/beta.py": "def two(y):\n    return y\n",
        })
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("something is off", root,
                                "from pkg.alpha import one\n"
                                "assert one(1) == 2\n")
        backend = ScriptedBackend(["none of these fit"], [])
        outcome = _run(vertical, backend)
        self.assertFalse(outcome["success"])
        self.assertIn("navigation abandoned", outcome["reason"])
        self.assertLessEqual(len(backend.menus_seen), 3)

    def test_corpus_with_root_level_files_is_navigable(self):
        holder, root = _corpus({
            "parsing.py": ('def parse_num(s):\n'
                           '    """Parse a decimal number."""\n'
                           '    return int(s) + 1\n'),
            "printing.py": ('def show(v):\n'
                            '    """Print a value."""\n'
                            '    return str(v)\n'),
        })
        self.addCleanup(holder.cleanup)
        vertical = EditVertical("number parsing is off by one", root,
                                "from parsing import parse_num\n"
                                "assert parse_num('4') == 4\n")
        backend = ScriptedBackend(["replace the target"], ["return int(s)"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target_file"], "parsing.py")


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
