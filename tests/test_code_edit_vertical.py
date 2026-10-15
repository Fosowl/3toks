"""Offline end-to-end tests for the edit vertical's state machine.

A scripted backend answers menus from a queue of keywords (the queue head
is used only when it names an option of the current menu, else the
fallback keywords answer) and answers generations from a queue of canned
completions. No model or network anywhere; checks run real subprocesses.
"""
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code.edit import nav, oracle, rank
from threetoks.code.edit.index import SymbolIndex
from threetoks.code.edit.operations import MAX_GENERATE_TOKENS
from threetoks.code.edit.splice import GENERATE_MAX_TOKENS
from threetoks.code.edit.vertical import EditVertical
from threetoks.engine import run_episode
from threetoks.policy import Policy, PolicyConfig

REPLACE = "replace the target"
YES = "yes — the problem"
ELSEWHERE = "somewhere else"
ADD_BUG = "def add(a, b):\n    return a - b\n"


class ScriptedBackend:
    """Scripted menu picks + a queue of generation completions."""

    def __init__(self, menus=(), completions=(), fallback=(YES, REPLACE)):
        self.menus = list(menus)
        self.fallback = list(fallback)
        self.completions = list(completions)
        self.menus_seen: list[str] = []
        self.generations = 0
        self.generation_caps: list[int] = []

    def complete(self, model, raw_prompt, opts):
        if "ACTIONS:" in raw_prompt:
            return self._answer_menu(raw_prompt)
        self.generations += 1
        self.generation_caps.append(opts.max_tokens)
        completion = self.completions[0] if len(self.completions) == 1 \
            else self.completions.pop(0)
        return GenResult(completion, 10, 20, 0.0, "stop")

    def _answer_menu(self, raw_prompt):
        menu = raw_prompt[raw_prompt.rindex("ACTIONS:"):]
        self.menus_seen.append(menu)
        if self.menus and _digit_for(menu, self.menus[0]):
            return _menu_answer(_digit_for(menu, self.menus.pop(0)))
        for keyword in self.fallback:
            if _digit_for(menu, keyword):
                return _menu_answer(_digit_for(menu, keyword))
        raise AssertionError(f"unscripted menu:\n{menu}")


def _digit_for(menu: str, keyword: str) -> str:
    for line in menu.split("\n"):
        digit, sep, label = line.partition(" = ")
        if sep and digit.isdigit() and keyword in label:
            return digit
    return ""


def _menu_answer(digit: str) -> GenResult:
    return GenResult(f" {digit}", 10, 1, 0.0, "stop")


def _run(vertical, backend, max_steps=30):
    policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    return run_episode(vertical, policy, max_steps=max_steps)


def _corpus(test_case, tree):
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
    return root


class DirectReplaceTest(unittest.TestCase):
    def test_success_returns_candidate_and_leaves_disk_untouched(self):
        root = _corpus(self, {"pkg/mod.py": ADD_BUG})
        vertical = EditVertical("fix the bug in add", root,
                                "from pkg.mod import add\n"
                                "assert add(2, 2) == 4\n")
        backend = ScriptedBackend(completions=["return a + b"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertTrue(outcome["verified"])
        self.assertEqual(outcome["path_taken"], "direct")
        self.assertEqual(outcome["edited_symbol"], "add")
        self.assertEqual(outcome["repairs"], 0)
        self.assertEqual(len(backend.menus_seen), 1)    # the confirm only
        self.assertIn("yes — the problem is in this function",
                      backend.menus_seen[0])
        self.assertIn("return a + b", outcome["candidate"])
        self.assertEqual(outcome["original"], ADD_BUG)
        self.assertEqual((root / "pkg" / "mod.py").read_text(), ADD_BUG)
        self.assertEqual(outcome["tests"], {})
        self.assertFalse(outcome["index_truncated"])

    def test_passing_test_short_circuits_with_zero_model_calls(self):
        root = _corpus(self, {"mod.py": "def add(a, b):\n    return a + b\n"})
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(1, 1) == 2\n")
        backend = ScriptedBackend()
        outcome = _run(vertical, backend)
        self.assertFalse(outcome["success"])
        self.assertIn("nothing to fix", outcome["reason"])
        self.assertEqual(backend.menus_seen, [])
        self.assertEqual(backend.generations, 0)

    def test_one_liner_target_offers_no_delete(self):
        root = _corpus(self, {"mod.py": "def add(a, b): return a - b\n"})
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")
        backend = ScriptedBackend(completions=["return a + b"],
                                  fallback=[YES, "delete the offending",
                                            REPLACE])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "replace")
        self.assertFalse(any("delete the offending" in m
                             for m in backend.menus_seen))

    def test_unique_name_is_confirmed_and_somewhere_else_moves_on(self):
        # "update" is a request word AND a function name: the confirm
        # menu lets the model reject it before any generation.
        root = _corpus(self, {"cfg.py": (
            "def update(d, k, v):\n    d[k] = v\n    return d\n\n\n"
            "def load_config(path):\n"
            '    """Load the config file."""\n'
            "    handle = open(path)\n    return handle.read()\n")})
        vertical = EditVertical(
            "update the config loader so it closes the file", root)
        backend = ScriptedBackend(
            [ELSEWHERE, YES],
            ["with open(path) as handle:\n        return handle.read()"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target"], "load_config")
        self.assertEqual(outcome["path_taken"], "shortlist")
        self.assertEqual(backend.generations, 1)
        self.assertIn("yes — the problem", backend.menus_seen[0])
        self.assertFalse(any("update —" in m for m in backend.menus_seen))

    def test_long_target_gets_a_larger_token_cap(self):
        steps = [f"    total = total + {i}  # add step number {i}"
                 for i in range(60)]
        source = "\n".join(["def big(x):", "    total = x", *steps,
                            "    return total - 1"]) + "\n"
        root = _corpus(self, {"mod.py": source})
        vertical = EditVertical("fix big", root,
                                "from mod import big\n"
                                "assert big(0) == sum(range(60))\n")
        backend = ScriptedBackend(completions=["return x + sum(range(60))"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertGreater(backend.generation_caps[0], GENERATE_MAX_TOKENS)
        self.assertLessEqual(backend.generation_caps[0], MAX_GENERATE_TOKENS)


class ShortlistTest(unittest.TestCase):
    def test_deep_file_is_reachable_through_one_shortlist_menu(self):
        root = _corpus(self, {
            "pkg/sub/deep/m.py": (
                "def convert_currency(amount, rate):\n"
                '    """Convert an amount of money with an exchange rate."""\n'
                "    return amount - rate\n"),
            "pkg/report.py": (
                "def format_amount(amount):\n"
                '    """Format a money amount for display."""\n'
                '    return f"{amount:.2f}"\n'),
            "pkg/util.py": "def unrelated(x):\n    return x\n",
        })
        vertical = EditVertical(
            "converting money with the exchange rate gives the wrong amount",
            root, "from pkg.sub.deep.m import convert_currency\n"
                  "assert convert_currency(2, 3) == 6\n")
        backend = ScriptedBackend(["convert_currency", YES],
                                  ["return amount * rate"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["path_taken"], "shortlist")
        self.assertEqual(outcome["target_file"], "pkg/sub/deep/m.py")
        self.assertIn("pkg/sub/deep/m.py", backend.menus_seen[0])
        self.assertNotIn("unrelated", backend.menus_seen[0])

    def test_no_matching_function_ends_with_zero_model_calls(self):
        root = _corpus(self, {"mod.py": ADD_BUG})
        vertical = EditVertical("the weather widget flickers", root)
        backend = ScriptedBackend()
        outcome = _run(vertical, backend)
        self.assertFalse(outcome["success"])
        self.assertEqual(outcome["reason"], "no function matches the request")
        self.assertEqual(backend.menus_seen, [])
        self.assertEqual(backend.generations, 0)

    def test_escape_at_the_shortlist_finishes_honestly(self):
        root = _corpus(self, {
            "a.py": "def parse_date(s):\n    return s\n",
            "b.py": "def parse_time(s):\n    return s\n",
        })
        vertical = EditVertical("parsing is slow", root)
        backend = ScriptedBackend(fallback=["none of these fit"])
        outcome = _run(vertical, backend)
        self.assertFalse(outcome["success"])
        self.assertIn("none of the shortlisted", outcome["reason"])
        self.assertEqual(len(backend.menus_seen), 1)
        self.assertEqual(backend.generations, 0)

    def test_class_name_is_never_replaced_by_a_def(self):
        root = _corpus(self, {"pager.py": (
            "class Paginator:\n"
            '    """Split items into pages."""\n\n'
            "    def __init__(self, items, size):\n"
            "        self.items = items\n"
            "        self.size = size\n\n"
            "    def slice_page(self, number):\n"
            '        """Return the items of one page."""\n'
            "        start = number * self.size\n"
            "        return self.items[start:start + self.size + 1]\n")})
        vertical = EditVertical(
            "Paginator returns one item too many per page", root,
            "from pager import Paginator\n"
            "assert Paginator([1, 2, 3, 4], 2).slice_page(0) == [1, 2]\n")
        backend = ScriptedBackend(
            ["Paginator.slice_page", YES],
            ["start = number * self.size\n"
             "    return self.items[start:start + self.size]"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target"], "Paginator.slice_page")
        self.assertIn("class Paginator:", outcome["candidate"])
        self.assertIn("def __init__(self, items, size):", outcome["candidate"])
        self.assertFalse(any("= Paginator —" in m for m in backend.menus_seen))

    def test_shortlist_pages_past_the_first_menu(self):
        tree = {f"items/m{i}.py": f"def load_item_{i}(x):\n    return x - 1\n"
                for i in range(10)}
        root = _corpus(self, tree)
        request = "loading an item is off by one"
        ranked = rank.shortlist(SymbolIndex(root), request)
        self.assertEqual(len(ranked), 10)
        last = ranked[-1]
        module = last.file.removesuffix(".py").replace("/", ".")
        vertical = EditVertical(
            request, root, f"from {module} import {last.name}\n"
                           f"assert {last.name}(1) == 2\n")
        backend = ScriptedBackend([nav.MORE_LABEL, f"{last.name} —", YES],
                                  ["return x + 1"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target"], last.name)
        self.assertNotIn(f"{last.name} —", backend.menus_seen[0])
        self.assertIn(nav.MORE_LABEL, backend.menus_seen[0])

    def test_property_setter_is_edited_with_its_own_header(self):
        root = _corpus(self, {"box.py": (
            "class Box:\n"
            "    def __init__(self):\n        self._w = 0\n\n"
            "    @property\n"
            "    def width(self):\n        return self._w\n\n"
            "    @width.setter\n"
            "    def width(self, value):\n        self._w = value\n")})
        vertical = EditVertical(
            "the Box.width setter must clamp negative values to zero", root,
            "from box import Box\nb = Box()\nb.width = -3\n"
            "assert b.width == 0\n")
        backend = ScriptedBackend(["box.py:10", YES],
                                  ["self._w = max(0, value)"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertIn("    @width.setter\n    def width(self, value):\n"
                      "        self._w = max(0, value)", outcome["candidate"])
        self.assertIn("    @property\n    def width(self):\n"
                      "        return self._w", outcome["candidate"])

    def test_star_import_never_gets_a_free_helper_insert(self):
        root = _corpus(self, {"geom.py": (
            "from math import *\n\n\n"
            "def hypot_len(a, b):\n    return sqrt(a * a - b * b)\n")})
        vertical = EditVertical("fix hypot_len", root,
                                "from geom import hypot_len\n"
                                "assert hypot_len(3, 4) == 5\n")
        backend = ScriptedBackend(completions=["return sqrt(a * a + b * b)"],
                                  fallback=[YES, "insert new helper", REPLACE])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "replace")
        self.assertNotIn("def sqrt", outcome["candidate"])
        self.assertFalse(any("insert new helper" in m
                             for m in backend.menus_seen))


class ConfirmTest(unittest.TestCase):
    CORPUS = {"shop.py": (
        "def order_total(prices):\n"
        '    """Sum the prices of an order."""\n'
        "    return sum(prices)\n\n\n"
        "def order_discount(total):\n"
        '    """Apply the order discount to a total."""\n'
        "    return total * 2\n")}

    def test_somewhere_else_strikes_the_pick_and_moves_on(self):
        root = _corpus(self, self.CORPUS)
        vertical = EditVertical(
            "the order discount is applied wrong", root,
            "from shop import order_discount\n"
            "assert order_discount(10) == 9\n")
        backend = ScriptedBackend(["order_total", "somewhere else", YES],
                                  ["return total * 0.9"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target"], "order_discount")
        self.assertEqual(backend.generations, 1)
        # the lone remaining candidate is taken without a second pick menu
        picks = [m for m in backend.menus_seen if "order_total —" in m]
        self.assertEqual(len(picks), 1)

    def test_callee_hop_moves_the_target(self):
        root = _corpus(self, {"m.py": (
            "def order_total(items):\n"
            '    """Total price of an order."""\n'
            "    return sum_prices(items) + 0\n\n\n"
            "def sum_prices(items):\n"
            "    return sum(item - 1 for item in items)\n")})
        vertical = EditVertical(
            "the order total price is off by one per item", root,
            "from m import order_total\nassert order_total([2, 3]) == 5\n")
        backend = ScriptedBackend(["order_total", "sum_prices()", YES],
                                  ["return sum(items)"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target"], "sum_prices")
        self.assertEqual(outcome["edited_symbol"], "sum_prices")
        self.assertIn("return sum_prices(items) + 0", outcome["candidate"])


class UnchangedTest(unittest.TestCase):
    def test_unchanged_output_moves_on_without_resampling(self):
        root = _corpus(self, {"geo.py": (
            "def scale_point(x):\n"
            '    """Scale a point coordinate."""\n'
            "    return x * 2\n\n\n"
            "def shift_point(x):\n"
            '    """Shift a point coordinate."""\n'
            "    return x - 1\n")})
        vertical = EditVertical(
            "moving a point coordinate gives the wrong result", root,
            "from geo import shift_point\nassert shift_point(1) == 2\n")
        backend = ScriptedBackend(["scale_point", YES, YES],
                                  ["return x * 2", "return x + 1"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target"], "shift_point")
        self.assertEqual(backend.generations, 2)
        self.assertEqual(outcome["repairs"], 0)


class DiskSafetyTest(unittest.TestCase):
    def test_interrupt_during_the_oracle_restores_crlf_bytes(self):
        original = b"def add(a, b):\r\n    return a - b\r\n"
        root = _corpus(self, {"mod.py": original})
        seen_on_disk = []

        def fake_run_test(run_root, test_code):
            if not seen_on_disk:
                seen_on_disk.append(b"")
                return False, "AssertionError"
            seen_on_disk.append((root / "mod.py").read_bytes())
            raise KeyboardInterrupt

        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")
        backend = ScriptedBackend(completions=["return a + b"])
        with mock.patch.object(oracle, "run_test", side_effect=fake_run_test):
            with self.assertRaises(KeyboardInterrupt):
                _run(vertical, backend)
        self.assertIn(b"a + b", seen_on_disk[-1])     # candidate was checked
        self.assertEqual((root / "mod.py").read_bytes(), original)

    def test_import_failure_before_the_edit_ends_with_zero_generations(self):
        root = _corpus(self, {"mod.py": "import not_a_real_module_xyz\n\n\n"
                                        + ADD_BUG})
        vertical = EditVertical("fix add", root)
        backend = ScriptedBackend(completions=["return a + b"])
        outcome = _run(vertical, backend)
        self.assertFalse(outcome["success"])
        self.assertIn("does not import", outcome["reason"])
        self.assertEqual(backend.generations, 0)
        self.assertEqual(len(backend.menus_seen), 1)    # the confirm only

    def test_no_test_mode_gates_on_import_and_reports_unverified(self):
        root = _corpus(self, {"mod.py": ADD_BUG})
        vertical = EditVertical("make add actually add", root)
        backend = ScriptedBackend(completions=["return a + b"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertFalse(outcome["verified"])
        self.assertIn("no test", outcome["reason"])
        self.assertEqual((root / "mod.py").read_text(), ADD_BUG)

    def test_unencodable_candidate_fails_honestly_and_keeps_bytes(self):
        original = (b"# -*- coding: latin-1 -*-\n"
                    b"def label(x):\n    return 'caf\xe9 ' + str(x)\n")
        root = _corpus(self, {"mod.py": original})
        vertical = EditVertical("fix label", root,
                                "from mod import label\n"
                                "assert label(1) == 'café — 1'\n")
        backend = ScriptedBackend(completions=["return 'café — ' + str(x)"])
        outcome = _run(vertical, backend)
        self.assertFalse(outcome["success"])
        self.assertEqual((root / "mod.py").read_bytes(), original)
        self.assertIn("cannot be saved in the file's encoding",
                      vertical._menu_episode.render_base())

    @unittest.skipIf(os.name != "posix" or os.geteuid() == 0,
                     "needs a non-root POSIX user")
    def test_read_only_target_fails_honestly_and_keeps_bytes(self):
        root = _corpus(self, {"mod.py": ADD_BUG})
        path = root / "mod.py"
        path.chmod(stat.S_IRUSR)
        self.addCleanup(path.chmod, stat.S_IRUSR | stat.S_IWUSR)
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")
        backend = ScriptedBackend(completions=["return a + b"])
        outcome = _run(vertical, backend)
        self.assertFalse(outcome["success"])
        self.assertEqual(path.read_text(), ADD_BUG)
        self.assertIn("cannot read or write the target file",
                      vertical._menu_episode.render_base())


class RepoTestOracleTest(unittest.TestCase):
    def test_baseline_compare_rejects_breakage_and_marks_verified(self):
        root = _corpus(self, {
            "calc.py": ADD_BUG,
            "tests/test_calc.py": (
                "import unittest\n\nfrom calc import add\n\n\n"
                "class AddTest(unittest.TestCase):\n"
                "    def test_sum(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n\n"
                "    def test_identity(self):\n"
                "        self.assertEqual(add(5, 0), 5)\n"),
        })
        vertical = EditVertical("fix add", root)
        backend = ScriptedBackend(completions=["return a * b",
                                               "return a + b"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertTrue(outcome["verified"])
        self.assertEqual(outcome["repairs"], 1)     # a * b broke identity
        self.assertEqual(backend.generations, 2)
        self.assertEqual(outcome["tests"]["files"], ["tests/test_calc.py"])
        self.assertEqual(outcome["tests"]["broken"], [])
        self.assertTrue(any("test_sum" in t for t in outcome["tests"]["fixed"]))
        self.assertEqual((root / "calc.py").read_text(), ADD_BUG)


class OperationTest(unittest.TestCase):
    def test_undefined_helper_skips_the_operation_menu(self):
        root = _corpus(self, {"mod.py": (
            "def average(nums):\n    total = 0\n"
            "    for n in nums:\n        total = _safe_add(total, n)\n"
            "    return total / len(nums)\n")})
        vertical = EditVertical("average crashes with a NameError", root,
                                "from mod import average\n"
                                "assert average([2, 4]) == 3\n")
        backend = ScriptedBackend(completions=["return arg1 + arg2"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "insert_after")
        self.assertEqual(outcome["edited_symbol"], "_safe_add")
        self.assertIn("def _safe_add", outcome["candidate"])
        self.assertFalse(any("How should this be fixed?" in m
                             for m in backend.menus_seen))

    def test_escalation_reaches_the_real_menu_after_inference_fails(self):
        root = _corpus(self, {"mod.py": (
            "def total(nums):\n    result = 0\n"
            "    for n in nums:\n        result = _acc(result, n)\n"
            "    return result\n")})
        vertical = EditVertical("total is broken", root,
                                "from mod import total\n"
                                "assert total([1, 2]) == 3\n")
        backend = ScriptedBackend(completions=["return arg1 - arg2"])
        _run(vertical, backend, max_steps=40)
        self.assertTrue(any("insert new helper code" in m
                            for m in backend.menus_seen), backend.menus_seen)

    def test_failed_delete_is_struck_from_the_retry_menu(self):
        root = _corpus(self, {"mod.py": (
            "def tidy(s):\n    s = s.strip()\n    s = s.swapcase()\n"
            "    return s\n")})
        vertical = EditVertical("tidy mangles what it is given", root,
                                "from mod import tidy\n"
                                "assert tidy(' abc ') == 'abc'\n")
        backend = ScriptedBackend(["delete the offending", "strip()",
                                   "swapcase()"])
        outcome = _run(vertical, backend)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["operation"], "delete")
        self.assertEqual(outcome["repairs"], 1)
        self.assertEqual(backend.generations, 0)
        self.assertIn("strip", outcome["candidate"])
        self.assertNotIn("swapcase", outcome["candidate"])


class RelocateTest(unittest.TestCase):
    def test_failed_target_is_struck_and_the_other_candidate_fixed(self):
        root = _corpus(self, {
            "shop/orders.py": "def validate(v):\n    return v > 0\n",
            "shop/users.py": "def validate(v):\n    return True\n",
        })
        vertical = EditVertical(
            "validate lets bad input through", root,
            "from shop.users import validate\nassert validate('') is False\n")
        backend = ScriptedBackend(["orders.py", YES, YES], ["return bool(v)"])
        outcome = _run(vertical, backend, max_steps=40)
        self.assertTrue(outcome["success"], outcome)
        self.assertEqual(outcome["target_file"], "shop/users.py")
        self.assertGreaterEqual(outcome["repairs"], 1)


class StatelessGenerationTest(unittest.TestCase):
    def test_generation_prompts_never_carry_failure_feedback(self):
        root = _corpus(self, {"mod.py": ADD_BUG})
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 3) == 5\n")
        generation_prompts = []

        class Recorder(ScriptedBackend):
            def complete(self, model, raw_prompt, opts):
                if "ACTIONS:" not in raw_prompt:
                    generation_prompts.append(raw_prompt)
                return super().complete(model, raw_prompt, opts)

        backend = Recorder(completions=["return a * b", "return a + b"])
        outcome = _run(vertical, backend, max_steps=40)
        self.assertTrue(outcome["success"], outcome)
        self.assertGreaterEqual(len(generation_prompts), 2)
        for prompt in generation_prompts:
            self.assertNotIn("edit failed", prompt)
            self.assertNotIn("> target:", prompt)
            self.assertIn("return a - b", prompt)   # verbatim span present
        self.assertIn("edit failed", vertical._menu_episode.render_base())


class BudgetTest(unittest.TestCase):
    def test_step_budget_forces_an_honest_failure(self):
        root = _corpus(self, {"mod.py": ADD_BUG})
        vertical = EditVertical("fix add", root,
                                "from mod import add\nassert add(2, 2) == 4\n")
        backend = ScriptedBackend(completions=["return a * b"])
        outcome = _run(vertical, backend, max_steps=2)
        self.assertFalse(outcome["success"])
        self.assertIn("budget", outcome["reason"])
        self.assertEqual(outcome["candidate"], "")


if __name__ == "__main__":
    unittest.main()
