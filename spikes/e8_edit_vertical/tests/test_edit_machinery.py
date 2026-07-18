"""Offline unittest for the splice + gate + oracle machinery (E8).

No LLM, no network -- a scripted fake backend stands in for the model,
exactly like tests/test_agents_core.py's ScriptedBackend at the repo root.
Covers the two load-bearing guarantees this spike depends on:

1. A splice that breaks ast.parse is rejected for free, and Policy's own
   retry ladder (temps 0.0/0.0/0.4) resamples blind until a candidate
   parses -- no bespoke retry code lives in this spike.
2. The trusted-oracle gate: a scenario's test must fail before an edit and
   pass after, exactly the discipline threetoks/code/gates.py uses for its
   subprocess checks.

Run: python3 spikes/e8_edit_vertical/tests/test_edit_machinery.py
 or: python3 -m unittest discover -s spikes/e8_edit_vertical/tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
SPIKE_DIR = TEST_DIR.parent
REPO_ROOT = SPIKE_DIR.parent.parent
for _p in (str(SPIKE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import edit_ops  # noqa: E402
import oracle  # noqa: E402
import scenarios as scen  # noqa: E402
from vertical import EditVertical  # noqa: E402

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec  # noqa: E402
from threetoks.engine import run_episode  # noqa: E402
from threetoks.policy import Policy, PolicyConfig  # noqa: E402
from threetoks.trace import Tracer  # noqa: E402


class ScriptedBackend:
    """Returns queued texts in order, like the repo's own test helper."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts = []

    def complete(self, model, raw_prompt, opts):
        self.prompts.append(raw_prompt)
        text = self.texts.pop(0) if self.texts else "return None"
        return GenResult(text, 10, 5, 0.0, "stop")


def make_policy(texts):
    return Policy(ScriptedBackend(texts),
                  PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


class SpliceGateTest(unittest.TestCase):
    """The pure splice + gate functions (edit_ops.py)."""

    SOURCE = "def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y\n"

    def test_replace_span_swaps_only_the_named_lines(self):
        out = edit_ops.replace_span(self.SOURCE, 1, 2,
                                    "def f(x):\n    return x * 2")
        self.assertIn("return x * 2", out)
        self.assertIn("def g(y):", out)          # sibling untouched

    def test_insert_after_span_keeps_both_functions(self):
        out = edit_ops.insert_after_span(self.SOURCE, 2, "def h(z):\n    return z")
        self.assertIn("def h(z):", out)
        self.assertIn("def f(x):", out)
        self.assertIn("def g(y):", out)

    def test_delete_span_removes_only_that_statement(self):
        out = edit_ops.delete_span(self.SOURCE, 2, 2)
        self.assertNotIn("return x + 1", out)
        self.assertIn("def g(y):", out)

    def test_replace_splice_fn_rejects_syntax_break_for_free(self):
        fn = edit_ops.replace_splice_fn(self.SOURCE, 1, 2, "f", "x")
        self.assertIsNone(fn("return (((mismatched"))

    def test_replace_splice_fn_rejects_placeholder_dodge(self):
        fn = edit_ops.replace_splice_fn(self.SOURCE, 1, 2, "f", "x")
        self.assertIsNone(fn("raise NotImplementedError"))
        self.assertIsNone(fn("pass"))

    def test_replace_splice_fn_accepts_a_real_fix(self):
        fn = edit_ops.replace_splice_fn(self.SOURCE, 1, 2, "f", "x")
        candidate = fn("return x + 2")
        self.assertIsNotNone(candidate)
        self.assertIn("return x + 2", candidate)
        import ast
        ast.parse(candidate)                     # whole file still parses

    def test_replace_splice_fn_recovers_from_indent_drift(self):
        """qwen3.5:2b live output observed in E8: every
        continuation line indented one column deeper (5 spaces) than the
        prefill's 4 -- would break ast.parse without renormalize_indent."""
        fn = edit_ops.replace_splice_fn(self.SOURCE, 1, 2, "f", "x")
        drifted = ('"""doc"""\n     y = x + 1\n     return y')
        candidate = fn(drifted)
        self.assertIsNotNone(candidate)
        import ast
        ast.parse(candidate)

    def test_find_missing_helper_gets_name_and_arity_for_free(self):
        source = ("def average(nums):\n    total = 0\n"
                 "    for n in nums:\n        total = _safe_add(total, n)\n"
                 "    return total / len(nums)\n")
        helper = edit_ops.find_missing_helper(source, "average", {"average"})
        self.assertEqual(helper, ("_safe_add", "a, b"))


class GeneratePolicyRetryTest(unittest.TestCase):
    """A syntax-breaking first attempt is rejected free and resampled by
    Policy's own retry ladder -- no bespoke retry logic in this spike."""

    def test_ladder_resamples_after_a_broken_splice(self):
        from threetoks.render import Episode

        source = "def f(x):\n    return x + 1\n"
        splice_fn = edit_ops.replace_splice_fn(source, 1, 2, "f", "x")
        node = edit_ops.GenerateSpanNode("fix f", "def f(x):\n    ", splice_fn)
        policy = make_policy(["return (((", "return x + 99"])
        episode = Episode("SYS", "fix f")
        decision = policy.decide(episode, node)
        self.assertTrue(decision.valid)
        self.assertIn("return x + 99", decision.value)
        self.assertEqual(len(policy.backend.prompts), 2, "should have retried")

    def test_ladder_gives_up_after_max_attempts_all_broken(self):
        from threetoks.render import Episode

        source = "def f(x):\n    return x + 1\n"
        splice_fn = edit_ops.replace_splice_fn(source, 1, 2, "f", "x")
        node = edit_ops.GenerateSpanNode("fix f", "def f(x):\n    ", splice_fn)
        policy = make_policy(["return (((", "pass", "raise NotImplementedError"])
        episode = Episode("SYS", "fix f")
        decision = policy.decide(episode, node)
        self.assertFalse(decision.valid)
        self.assertEqual(len(policy.backend.prompts), 3)


class OracleGateTest(unittest.TestCase):
    """The trusted-oracle test: must fail before, pass after."""

    def test_oracle_fails_before_and_passes_after_a_real_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mod.py").write_text("def add(a, b):\n    return a - b\n")
            test = "from mod import add\nassert add(2, 2) == 4, add(2, 2)\n"
            before_ok, before_err = oracle.run_test(root, test)
            self.assertFalse(before_ok)
            self.assertIn("AssertionError", before_err)

            (root / "mod.py").write_text("def add(a, b):\n    return a + b\n")
            after_ok, after_err = oracle.run_test(root, test)
            self.assertTrue(after_ok)
            self.assertEqual(after_err, "")


class ScenarioEndToEndTest(unittest.TestCase):
    """Every planted scenario's fixture actually reproduces its bug, and
    the vertical's own splice mechanics (given the scenario's own correct
    completion) resolve it -- run through the full engine.run_episode
    loop with a scripted, keyword-seeking backend, no real model."""

    def test_all_five_scenarios_pass_offline(self):
        import re

        action_line = re.compile(r"^(\d+) = (.*)$")

        class KeywordBackend:
            def __init__(self, keywords, completion):
                self.keywords = keywords
                self.completion = completion

            def complete(self, model, raw_prompt, opts):
                if "ACTIONS:" in raw_prompt:
                    for kw in self.keywords:
                        for line in raw_prompt.splitlines():
                            match = action_line.match(line.strip())
                            if match and kw in match.group(2):
                                return GenResult(f" {match.group(1)}", 10, 1,
                                                 0.0, "stop")
                    return GenResult(" 1", 10, 1, 0.0, "stop")
                return GenResult(self.completion or "pass", 10, 10, 0.0, "stop")

        import vertical as vert

        with tempfile.TemporaryDirectory() as tmp:
            for scenario in scen.ALL_SCENARIOS:
                root = Path(tmp) / scenario.key
                scen.materialize(scenario, root)
                keywords = list(scenario.nav_keywords)
                if scenario.disambiguate_keyword:
                    keywords.append(scenario.disambiguate_keyword)
                keywords.append(vert.OPERATION_LABELS[scenario.expected_operation])
                if scenario.delete_keyword:
                    keywords.append(scenario.delete_keyword)
                backend = KeywordBackend(keywords, scenario.good_completion)
                policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)),
                                Tracer(None))
                edit_vertical = EditVertical(scenario.request, root,
                                             scenario.test_code)
                outcome = run_episode(edit_vertical, policy, max_steps=30)
                self.assertTrue(outcome["success"],
                               f"{scenario.key}: {outcome['reason']}")


if __name__ == "__main__":
    unittest.main()
