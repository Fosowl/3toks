"""Offline tests for the e7 code-mode routing pipeline.

No Ollama, no network: the model-consulting half uses a scripted fake
backend, the same pattern as ``ScriptedBackend`` in
``tests/test_agents_core.py`` (a fixed queue of canned completions). This
proves the pre-router + menu + measurement plumbing all work without a
live model before spending any real Ollama calls.

Run from the repo root:
    PYTHONPATH="$PWD:$PWD/spikes/e7_code_routing" \\
        python3 -m unittest spikes.e7_code_routing.test_offline -v
or directly from this directory:
    PYTHONPATH="<repo_root>:." python3 test_offline.py
"""
import unittest
from pathlib import Path

from measure import ScriptedBackend, route_all, summarize
from pre_router import AUTHOR, COMPUTE, EDIT, MODES, NAVIGATE, pre_classify
from router_menu import OPTION_TEXT, classify_with_menu

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.policy import Policy, PolicyConfig

FIXTURES = Path(__file__).parent / "fixtures"


class PreRouterTest(unittest.TestCase):
    """Deterministic checks: zero model calls, ever."""

    def test_edit_verb_plus_real_path_resolves_free(self):
        mode, _ = pre_classify("fix the bug in parser.py", [FIXTURES])
        self.assertEqual(mode, EDIT)

    def test_navigate_verb_plus_real_path_resolves_free(self):
        mode, _ = pre_classify(
            "where is the Cache class implemented in pkg/utils.py?", [FIXTURES])
        self.assertEqual(mode, NAVIGATE)

    def test_compute_verb_plus_real_path_resolves_free(self):
        mode, _ = pre_classify(
            "what's the sum of the revenue column in sales.csv?", [FIXTURES])
        self.assertEqual(mode, COMPUTE)

    def test_author_phrasing_needs_no_real_path(self):
        mode, _ = pre_classify("write a script that reverses a string",
                               [FIXTURES])
        self.assertEqual(mode, AUTHOR)

    def test_author_phrasing_beats_an_incidental_path_mention(self):
        """A file named only as an example must not steer this to compute."""
        mode, _ = pre_classify(
            "write a script that parses log files like access.log and "
            "reports the top 5 IPs", [FIXTURES])
        self.assertEqual(mode, AUTHOR)

    def test_path_mention_without_a_verb_cue_falls_through(self):
        mode, reason = pre_classify("parser.py looks weird today", [FIXTURES])
        self.assertIsNone(mode)
        self.assertIn("parser.py", reason)

    def test_unrelated_chatter_falls_through(self):
        mode, _ = pre_classify("hello there, how are you", [FIXTURES])
        self.assertIsNone(mode)

    def test_nonexistent_path_is_not_mistaken_for_a_real_one(self):
        """"db.py" isn't a fixture: this must not free-resolve as edit."""
        mode, _ = pre_classify("fix the bug in db.py", [FIXTURES])
        self.assertIsNone(mode)


class MenuRouterTest(unittest.TestCase):
    """The one-token MenuNode fallback, driven by a scripted backend."""

    def _policy(self, texts):
        return Policy(ScriptedBackend(texts),
                     PolicyConfig(ModelSpec("m", FAMILY_CHATML)))

    def test_scripted_reply_maps_back_to_a_real_mode(self):
        mode, decision = classify_with_menu("some ambiguous coding request",
                                            self._policy([" 2"]))
        self.assertIn(mode, MODES)
        self.assertTrue(decision.valid)

    def test_garbage_reply_falls_back_to_the_default_mode(self):
        from router_menu import FALLBACK_MODE
        mode, decision = classify_with_menu("some ambiguous coding request",
                                            self._policy(["garbage", "garbage",
                                                          "garbage"]))
        self.assertEqual(mode, FALLBACK_MODE)

    def test_every_option_text_round_trips_through_its_own_mode(self):
        for mode in MODES:
            self.assertIn(mode, OPTION_TEXT)
            self.assertTrue(OPTION_TEXT[mode].startswith(mode))


class MeasurementHarnessTest(unittest.TestCase):
    """route_all / summarize never touch the backend for a free pre-router hit,
    and the aggregate counts match a hand-checked confusion table."""

    def test_pre_router_hits_never_call_the_backend(self):
        backend = ScriptedBackend([" 1"])
        items = [{"request": "fix the bug in parser.py", "label": EDIT}]
        route_all(backend, items, [FIXTURES])
        self.assertEqual(backend.calls, 0)

    def test_fallthrough_item_spends_exactly_one_backend_call(self):
        backend = ScriptedBackend([" 2"])
        items = [{"request": "hello there, how are you", "label": AUTHOR}]
        route_all(backend, items, [FIXTURES])
        self.assertEqual(backend.calls, 1)

    def test_summarize_matches_a_hand_checked_confusion_table(self):
        records = [
            {"true": "edit", "pred": "edit", "source": "pre_router",
             "attempts": 0, "out_tokens": 0},
            {"true": "author", "pred": "compute", "source": "menu",
             "attempts": 1, "out_tokens": 3},
            {"true": "author", "pred": "author", "source": "menu",
             "attempts": 2, "out_tokens": 6},
        ]
        summary = summarize(records)
        self.assertEqual(summary["pre_router"]["coverage"], 1 / 3)
        self.assertEqual(summary["pre_router"]["precision"], 1.0)
        self.assertEqual(summary["menu"]["accuracy"], 0.5)
        self.assertEqual(summary["menu"]["per_true_mode"]["author"]["n"], 2)
        self.assertEqual(summary["menu"]["retried"], 1)  # the 2-attempt item
        self.assertAlmostEqual(summary["overall_accuracy"], 2 / 3)
        self.assertEqual(summary["confusion"]["edit->edit"], 1)
        self.assertEqual(summary["confusion"]["author->compute"], 1)


if __name__ == "__main__":
    unittest.main()
