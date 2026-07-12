"""Two-anchor trusted contracts (the E9 is_prime lesson).

One anchor says nothing about behavior at the domain boundary: E9's
retrieval spike accepted an is_prime that returns True for 0 and 1
because the single anchor only checked is_prime(17). These tests pin the
widened plumbing: every anchor in the list gates, in the per-method check
and in the runner phase alike.
"""
import unittest

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code import runner
from threetoks.code.store import MethodStore
from threetoks.code.vertical import CodeVertical
from threetoks.engine import run_episode
from threetoks.policy import Policy, PolicyConfig

NAIVE_PRIME = ("for i in range(2, int(n ** 0.5) + 1):\n"
               "        if n % i == 0:\n            return False\n"
               "    return True")
CORRECT_PRIME = ("if n < 2:\n        return False\n"
                 "    for i in range(2, int(n ** 0.5) + 1):\n"
                 "        if n % i == 0:\n            return False\n"
                 "    return True")


class QueueBackend:
    def __init__(self, texts):
        self.texts = list(texts)

    def complete(self, model, raw_prompt, opts):
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return GenResult(text, 8, 4, 0.0, "stop")


def _policy(backend):
    return Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


class StoreExamplesTest(unittest.TestCase):
    def test_add_normalizes_a_single_string_to_a_list(self):
        store = MethodStore("g")
        record = store.add("f", "x", "c", "f(1) == 2")
        self.assertEqual(record.examples, ["f(1) == 2"])

    def test_add_accepts_a_list_and_none(self):
        store = MethodStore("g")
        both = store.add("f", "x", "c", ["f(1) == 2", "f(0) == 1"])
        none = store.add("h", "x", "c")
        self.assertEqual(both.examples, ["f(1) == 2", "f(0) == 1"])
        self.assertEqual(none.examples, [])


class BoundaryAnchorGateTest(unittest.TestCase):
    ANCHORS = {"is_prime": ["is_prime(17) == True", "is_prime(1) == False"]}

    def test_body_passing_only_the_normal_anchor_is_rejected(self):
        backend = QueueBackend(["is_prime(n): true when n is prime",
                                NAIVE_PRIME, CORRECT_PRIME])
        vertical = CodeVertical("check primality with is_prime(n)",
                                gen_tests=False,
                                trusted_examples=self.ANCHORS)
        outcome = run_episode(vertical, _policy(backend), max_steps=20)
        self.assertIn("n < 2", outcome["answer"])       # boundary-safe body
        row = outcome["methods"][0]
        self.assertEqual(row["status"], "tested")
        self.assertTrue(row["verified"])
        self.assertGreaterEqual(row["attempts"], 2)     # naive one rejected

    def test_single_anchor_would_have_accepted_the_naive_body(self):
        # The control: with only the normal-case anchor, the E9 blind spot
        # ships the boundary-broken body as "verified".
        backend = QueueBackend(["is_prime(n): true when n is prime",
                                NAIVE_PRIME])
        vertical = CodeVertical(
            "check primality with is_prime(n)", gen_tests=False,
            trusted_examples={"is_prime": "is_prime(17) == True"})
        outcome = run_episode(vertical, _policy(backend), max_steps=20)
        self.assertNotIn("n < 2", outcome["answer"])
        self.assertTrue(outcome["methods"][0]["verified"])


class RunnerMultiAnchorTest(unittest.TestCase):
    def test_collect_checks_emits_one_check_per_anchor(self):
        class _M:
            def __init__(self, name, args, body, examples=()):
                self.name, self.args = name, args
                self.body, self.examples = body, list(examples)

        checks = runner.collect_checks(
            [_M("f", "n", "x", ["f(1) == 2", "f(0) == 1"])])
        self.assertEqual([c.statement for c in checks],
                         ["assert f(1) == 2", "assert f(0) == 1"])


if __name__ == "__main__":
    unittest.main()
