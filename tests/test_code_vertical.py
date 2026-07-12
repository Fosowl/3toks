"""End-to-end offline tests for the CodeVertical state machine.

Each test drives a real episode with a scripted backend that returns queued
completion strings in node order: first the PLAN completion (after the
"FUNCTIONS:\n1. " prefill), then per method an IMPLEMENT completion (the body
after the "def name(args):\n    " prefill), then a TEST completion (asserts
after the "assert " prefill) if that method reaches the test phase. The gates
shell out to a real python3 subprocess, so bodies here are tiny and fast.
"""
import ast
import unittest

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code.vertical import CodeVertical
from threetoks.engine import run_episode
from threetoks.policy import Policy, PolicyConfig

MAX_STEPS = 30


class ScriptedBackend:
    """Returns queued completion texts; a benign default once drained."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, model, raw_prompt, opts):
        self.calls += 1
        text = self.texts.pop(0) if self.texts else "return None"
        return GenResult(text, 8, 4, 0.0, "stop")


def make_policy(texts):
    """A Policy wrapping a ScriptedBackend over ChatML; returns (policy, backend)."""
    backend = ScriptedBackend(texts)
    policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    return policy, backend


def _row(outcome, name):
    """The result row for a method by name."""
    return next(r for r in outcome["methods"] if r["name"] == name)


class HappyPathTest(unittest.TestCase):
    def test_correct_body_and_passing_test_is_tested_and_verified(self):
        policy, _ = make_policy([
            "double(n): returns n times two",   # plan
            "return n * 2",                      # implement double
            "double(3) == 6",                    # test (after "assert ")
        ])
        vertical = CodeVertical("double a number", gen_tests=True)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "tested")
        self.assertTrue(row["verified"])
        self.assertIn("return n * 2", outcome["answer"])
        ast.parse(outcome["answer"])


class WrongTestDoesNotVetoCorrectCodeTest(unittest.TestCase):
    def test_correct_body_with_wrong_assert_stays_tested_but_unverified(self):
        # The central guarantee: a bad model-written test must not discard a
        # gate-passing body. A failing advisory assert (discard=False) triggers
        # a resample rather than an immediate finalize, so we feed the SAME
        # correct body on every implement attempt; the method must still end
        # tested/unverified with the correct body intact (never stubbed).
        policy, _ = make_policy([
            "double(n): returns n times two",   # plan
            "return n * 2",                      # implement attempt 1 (correct)
            "double(3) == 99",                   # WRONG advisory assert
            "return n * 2",                      # implement resample (same body)
        ])
        vertical = CodeVertical("double a number", gen_tests=True)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "tested")
        self.assertFalse(row["verified"])
        self.assertIn("return n * 2", outcome["answer"])  # correct body kept
        self.assertNotIn("raise NotImplementedError", outcome["answer"])
        ast.parse(outcome["answer"])


class SyntaxFailureExhaustsToStubTest(unittest.TestCase):
    # A syntax-error body makes the ImplementNode.parse invalid, so it is
    # eaten by the POLICY retry ladder (max_attempts=3) *within* one vertical
    # method attempt; the vertical then resamples the method up to
    # MAX_METHOD_ATTEMPTS=3 times. So exhausting to a stub needs 3x3 = 9
    # non-parsing completions (both nested loops must drain), else the drained
    # default "return None" parses and the method succeeds.
    _NEVER_PARSES = ["return (((", "return )))", "return ]["] * 3

    def test_never_parsing_body_ends_stubbed_and_module_still_parses(self):
        policy, _ = make_policy(
            ["double(n): returns n times two", *self._NEVER_PARSES])
        vertical = CodeVertical("double a number", gen_tests=True)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "stubbed")
        self.assertIsNone(vertical.store.methods[0].body)  # body absent
        self.assertIn("raise NotImplementedError", outcome["answer"])
        ast.parse(outcome["answer"])


class PlaceholderBodyIsRejectedTest(unittest.TestCase):
    # A pass / NotImplementedError body parses and imports, so without the
    # placeholder gate it would be accepted as a fake "done" method. It must
    # instead fail the gate, resample, and (if the model keeps cheating) end
    # as an honest stub — never a shipped pass-body.
    def test_placeholder_bodies_never_ship_as_tested(self):
        policy, _ = make_policy([
            "double(n): returns n times two",
            "pass",                              # cheat 1
            "raise NotImplementedError",         # cheat 2
            "...",                               # cheat 3
        ])
        vertical = CodeVertical("double a number", gen_tests=True)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "stubbed")
        self.assertEqual(row["attempts"], 3)  # each cheat was rejected
        ast.parse(outcome["answer"])

    def test_real_body_after_a_cheat_is_accepted(self):
        policy, _ = make_policy([
            "double(n): returns n times two",
            "pass",                              # cheat, rejected -> resample
            "return n * 2",                      # real body accepted
        ])
        vertical = CodeVertical("double a number", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "tested")
        self.assertIn("return n * 2", outcome["answer"])


class TrustedExampleGatesTest(unittest.TestCase):
    def test_body_failing_the_trusted_example_is_stubbed_on_exhaustion(self):
        # A body that fails the trusted example is known bad, so on
        # exhaustion it must be discarded and stubbed — a trusted anchor is
        # gating, unlike an advisory model assert which keeps the body.
        policy, _ = make_policy([
            "double(n): returns n times two",   # plan (name matches trusted key)
            "return n * 3",                      # wrong (parses, fails trusted)
            "return n + 1",                      # wrong
            "return n - 1",                      # wrong (final attempt)
        ])
        vertical = CodeVertical("double a number", gen_tests=True,
                                trusted_examples={"double": "double(3)==6"})
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "stubbed")
        self.assertFalse(row["verified"])
        self.assertNotIn("return n - 1", outcome["answer"])
        self.assertIn("NotImplementedError", outcome["answer"])
        ast.parse(outcome["answer"])

    def test_body_passing_the_trusted_example_is_verified(self):
        policy, _ = make_policy([
            "double(n): returns n times two",   # plan
            "return n * 2",                      # passes trusted double(3)==6
        ])
        vertical = CodeVertical("double a number", gen_tests=True,
                                trusted_examples={"double": "double(3)==6"})
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "tested")
        self.assertTrue(row["verified"])
        self.assertIn("return n * 2", outcome["answer"])


class GarbagePlanLinesAreDroppedTest(unittest.TestCase):
    def test_expression_args_and_builtin_names_never_reach_the_store(self):
        # Live-run regression: the model planned `print(i * i)` — a
        # builtin-shadowing name with an expression parameter list. Its
        # stub renders as a SyntaxError, breaking the always-parses
        # invariant and failing every sibling's candidate-import gate.
        policy, _ = make_policy([
            "square(n): multiplies n by itself\n"
            "2. print(i * i): shows the result\n"      # junk: builtin + expr
            "3. sum(values): adds them up",            # junk: builtin shadow
            "return n * n",                             # implement square
        ])
        vertical = CodeVertical("square numbers", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        self.assertEqual([r["name"] for r in outcome["methods"]], ["square"])
        self.assertEqual(_row(outcome, "square")["status"], "tested")
        ast.parse(outcome["answer"])

    def test_all_junk_plan_falls_back_to_main(self):
        policy, _ = make_policy([
            "print(i * i): shows the result",           # only junk planned
            "return 42",                                 # implement main()
        ])
        vertical = CodeVertical("show something", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        self.assertEqual([r["name"] for r in outcome["methods"]], ["main"])
        ast.parse(outcome["answer"])


class EmptyPlanFallbackTest(unittest.TestCase):
    def test_unparsable_plan_falls_back_to_a_single_main_method(self):
        # Plan text with no structural signal yields no methods -> FALLBACK.
        policy, _ = make_policy([
            "some prose that names no function at all",  # plan -> empty
            "return 42",                                  # implement main()
        ])
        vertical = CodeVertical("solve it", gen_tests=True)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        self.assertEqual([r["name"] for r in outcome["methods"]], ["main"])
        self.assertIn("def main():", outcome["answer"])
        ast.parse(outcome["answer"])


class GenTestsFalseTest(unittest.TestCase):
    def test_gate_passing_body_is_finalized_without_a_test_call(self):
        # gen_tests=False must request no TestNode: plan + implement = 2 calls.
        policy, backend = make_policy([
            "double(n): returns n times two",   # plan
            "return n * 2",                      # implement
        ])
        vertical = CodeVertical("double a number", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "tested")
        self.assertFalse(row["verified"])  # gated, but no asserts ran
        self.assertIn("return n * 2", outcome["answer"])
        self.assertEqual(backend.calls, 2)  # no third (test) generation


class RunnerRepairLoopTest(unittest.TestCase):
    def test_crashing_helper_is_blamed_and_repaired(self):
        # helper's crash only surfaces when entry's trusted example runs;
        # the blame must land on helper (the traceback's deepest frame),
        # entry must keep its innocent body, and the repaired module must
        # pass the final run.
        policy, _ = make_policy([
            "helper(n): doubles n\n2. entry(n): helper plus one",  # plan
            "return n // 0",           # helper: gated fine, crashes at runtime
            "return helper(n) + 1",    # entry: correct, example trips helper
            "return n * 2",            # helper repair: correct
        ])
        vertical = CodeVertical("entry(n) is double n plus one",
                                gen_tests=False,
                                trusted_examples={"entry": "entry(3) == 7"})
        outcome = run_episode(vertical, policy, MAX_STEPS)
        self.assertEqual(_row(outcome, "helper")["repairs"], 1)
        self.assertEqual(_row(outcome, "entry")["attempts"], 1)  # never blamed
        self.assertTrue(_row(outcome, "entry")["verified"])
        self.assertTrue(outcome["run"]["ok"])
        self.assertIn("return n * 2", outcome["answer"])
        ast.parse(outcome["answer"])

    def test_example_calling_an_unwritten_sibling_defers_to_the_runner(self):
        # entry is implemented first and its example hits the helper STUB
        # (NotImplementedError). That must not burn entry's attempts: the
        # check defers, helper gets written, and the final run verifies
        # entry against the finished module.
        policy, _ = make_policy([
            "entry(n): helper plus one\n2. helper(n): doubles n",  # plan
            "return helper(n) + 1",    # entry: calls the still-stubbed helper
            "return n * 2",            # helper: correct
        ])
        vertical = CodeVertical("entry(n) is double n plus one",
                                gen_tests=False,
                                trusted_examples={"entry": "entry(3) == 7"})
        outcome = run_episode(vertical, policy, MAX_STEPS)
        self.assertEqual(_row(outcome, "entry")["attempts"], 1)
        self.assertTrue(_row(outcome, "entry")["verified"])
        self.assertEqual(_row(outcome, "helper")["repairs"], 0)
        self.assertTrue(outcome["run"]["ok"])
        ast.parse(outcome["answer"])

    def test_zero_arg_entry_is_smoke_run_and_repaired(self):
        # No trusted example at all: the runner still executes main() and
        # a runtime crash unlocks it for one blind repair.
        policy, _ = make_policy([
            "main(): computes the answer",  # plan
            "return 1 // 0",                 # gated fine, crashes when run
            "return 42",                     # repair
        ])
        vertical = CodeVertical("compute the answer", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "main")
        self.assertEqual(row["repairs"], 1)
        self.assertEqual(row["status"], "tested")
        self.assertTrue(outcome["run"]["ok"])
        self.assertEqual(outcome["run"]["rounds"], 1)
        self.assertIn("return 42", outcome["answer"])

    def test_repair_rounds_are_capped_and_reported(self):
        # A method that crashes on every rewrite must stop at the round cap
        # with an honest failing run report, not loop forever.
        policy, _ = make_policy([
            "main(): computes the answer",
            "return 1 // 0",                 # initial
            "return 2 // 0",                 # repair round 1, still crashes
            "return 3 // 0",                 # repair round 2, still crashes
        ])
        vertical = CodeVertical("compute the answer", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        self.assertEqual(_row(outcome, "main")["repairs"], 2)
        self.assertFalse(outcome["run"]["ok"])
        self.assertEqual(outcome["run"]["rounds"], 2)
        [failure] = outcome["run"]["failures"]
        self.assertEqual(failure["blamed"], "main")
        self.assertIn("ZeroDivisionError", failure["error"])
        ast.parse(outcome["answer"])

    def test_interactive_input_code_is_not_treated_as_broken(self):
        # input() dies with EOFError in the sandbox; that is inconclusive,
        # so the body must survive with zero repairs and a clean report.
        policy, _ = make_policy([
            "main(): asks the user a question",
            "return input('? ')",
        ])
        vertical = CodeVertical("ask the user a question", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        self.assertEqual(_row(outcome, "main")["repairs"], 0)
        self.assertTrue(outcome["run"]["ok"])
        self.assertTrue(outcome["run"]["ran"])
        self.assertIn("input(", outcome["answer"])

    def test_crash_merely_mentioning_eoferror_is_still_repaired(self):
        # Regression: the inconclusive skip must not swallow a real crash
        # whose message contains the word EOFError.
        policy, _ = make_policy([
            "main(): computes the answer",
            "raise ValueError('parser hit EOFError token')",
            "return 42",
        ])
        vertical = CodeVertical("compute the answer", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        self.assertEqual(_row(outcome, "main")["repairs"], 1)
        self.assertTrue(outcome["run"]["ok"])
        self.assertIn("return 42", outcome["answer"])

    def test_example_dying_on_eof_keeps_the_body_unverified(self):
        # A trusted example that hits input() EOF is inconclusive at
        # implement time too — same rule as the runner phase, so the two
        # paths cannot disagree about the same body.
        policy, _ = make_policy([
            "ask(): reads the answer from the user",
            "return input('? ')",
        ])
        vertical = CodeVertical("ask the user", gen_tests=False,
                                trusted_examples={"ask": "ask() == 'y'"})
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "ask")
        self.assertEqual(row["status"], "tested")
        self.assertFalse(row["verified"])
        self.assertEqual(row["attempts"], 1)   # never resampled over EOF
        self.assertIn("input(", outcome["answer"])


class ScriptGuardTest(unittest.TestCase):
    GUARD = 'if __name__ == "__main__":'

    def _drive(self, goal, texts, trusted=None):
        policy, _ = make_policy(texts)
        vertical = CodeVertical(goal, gen_tests=False,
                                trusted_examples=trusted)
        return run_episode(vertical, policy, MAX_STEPS)

    def test_bodied_zero_arg_main_gets_the_guard(self):
        outcome = self._drive("compute the answer", [
            "main(): computes the answer", "return 42"])
        self.assertTrue(outcome["answer"].endswith(
            self.GUARD + "\n    main()\n"), outcome["answer"])
        ast.parse(outcome["answer"])

    def test_argful_module_gets_no_guard(self):
        outcome = self._drive("double a number", [
            "double(n): returns n times two", "return n * 2"])
        self.assertNotIn(self.GUARD, outcome["answer"])

    def test_stubbed_entry_gets_no_guard(self):
        # A guard calling a NotImplementedError stub would make the module
        # crash on run — worse than no guard.
        outcome = self._drive("compute the answer", [
            "main(): computes the answer",
            *["return ((("] * 9])              # never parses -> stubbed
        self.assertNotIn(self.GUARD, outcome["answer"])

    def test_goal_named_zero_arg_entry_is_called(self):
        outcome = self._drive("write report() that prints a report", [
            "report(): prints the report", "print('report')"])
        self.assertIn(self.GUARD + "\n    report()\n", outcome["answer"])

    def test_sole_zero_arg_function_is_called_without_a_name_match(self):
        outcome = self._drive("show the squares", [
            "show_squares(): prints the squares", "print([1, 4, 9])"])
        self.assertIn(self.GUARD + "\n    show_squares()\n",
                      outcome["answer"])


class ReturnShapedCheatsAreRejectedTest(unittest.TestCase):
    def test_return_not_implemented_error_never_ships(self):
        # `return NotImplementedError` parses, imports, and even "runs" —
        # only the widened placeholder gate stops it.
        policy, _ = make_policy([
            "double(n): returns n times two",
            "return NotImplementedError",        # cheat 1
            "return None",                        # cheat 2
            "return",                             # cheat 3
        ])
        vertical = CodeVertical("double a number", gen_tests=False)
        outcome = run_episode(vertical, policy, MAX_STEPS)
        row = _row(outcome, "double")
        self.assertEqual(row["status"], "stubbed")
        self.assertEqual(row["attempts"], 3)
        self.assertNotIn("return NotImplementedError", outcome["answer"])


if __name__ == "__main__":
    unittest.main()
