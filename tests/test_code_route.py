"""Offline tests for the code-mode router (pre-checks + one-token menu)."""
import tempfile
import unittest
from pathlib import Path

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code import route
from threetoks.policy import Policy, PolicyConfig


class PreRouteTest(unittest.TestCase):
    def setUp(self):
        self._holder = tempfile.TemporaryDirectory()
        self.addCleanup(self._holder.cleanup)
        self.root = Path(self._holder.name)
        (self.root / "parser.py").write_text("def parse():\n    return 1\n")

    def test_path_plus_edit_verb_is_edit(self):
        mode, reason = route.pre_route("fix the crash in parser.py", self.root)
        self.assertEqual(mode, route.EDIT)
        self.assertIn("edit verb", reason)

    def test_path_plus_navigate_verb_is_navigate(self):
        mode, _ = route.pre_route("what does parser.py actually do?",
                                  self.root)
        self.assertEqual(mode, route.NAVIGATE)

    def test_path_plus_compute_verb_is_compute(self):
        mode, _ = route.pre_route("count the defs in parser.py", self.root)
        self.assertEqual(mode, route.COMPUTE)

    def test_path_without_verb_cue_falls_through(self):
        mode, _ = route.pre_route("parser.py looks weird today", self.root)
        self.assertIsNone(mode)

    def test_nonexistent_path_is_not_a_path_signal(self):
        mode, _ = route.pre_route("fix the crash in nothere.py", self.root)
        self.assertIsNone(mode)

    def test_author_phrasing_wins_even_when_a_real_path_is_named(self):
        mode, _ = route.pre_route(
            "write a script that parses log files like parser.py", self.root)
        self.assertEqual(mode, route.AUTHOR)

    def test_interrogative_code_entity_shape_is_navigate(self):
        mode, _ = route.pre_route(
            "where is the retry function defined?", None)
        self.assertEqual(mode, route.NAVIGATE)

    def test_compute_verb_plus_data_noun_is_compute(self):
        mode, _ = route.pre_route(
            "sum the second column of the csv", None)
        self.assertEqual(mode, route.COMPUTE)

    def test_bare_top_is_not_a_compute_cue_but_top_k_is(self):
        self.assertIsNone(route.pre_route(
            "the top function in that module is odd", None)[0])
        self.assertEqual(route.pre_route(
            "give me the top 5 IPs in the log", None)[0], route.COMPUTE)

    def test_ambiguous_requests_fall_through(self):
        for request in ("hello there", "hmm, code stuff",
                        "this script is supposed to help"):
            self.assertIsNone(route.pre_route(request, self.root)[0],
                              request)


class _MenuBackend:
    """Answers the mode menu by grepping the options for a keyword."""

    def __init__(self, keyword):
        self.keyword = keyword
        self.calls = 0

    def complete(self, model, raw_prompt, opts):
        self.calls += 1
        live_menu = raw_prompt[raw_prompt.rindex("ACTIONS:"):]
        for line in live_menu.splitlines():
            if line[:1].isdigit() and self.keyword in line:
                return GenResult(f" {line.split(' = ')[0]}", 5, 1, 0.0, "stop")
        return GenResult(" x", 5, 1, 0.0, "stop")


def _policy(backend):
    return Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


class RouteModeTest(unittest.TestCase):
    def test_pre_check_skips_the_model_entirely(self):
        backend = _MenuBackend("navigate")
        mode, how = route.route_mode("write a script that sorts numbers",
                                     _policy(backend), None)
        self.assertEqual(mode, route.AUTHOR)
        self.assertTrue(how.startswith("pre:"))
        self.assertEqual(backend.calls, 0)

    def test_fall_through_uses_one_menu(self):
        backend = _MenuBackend("compute")
        mode, how = route.route_mode("hmm, something about numbers",
                                     _policy(backend), None)
        self.assertEqual(mode, route.COMPUTE)
        self.assertEqual(how, "menu")

    def test_garbage_menu_reply_falls_back_to_navigate(self):
        backend = _MenuBackend("no-such-option")
        mode, how = route.route_mode("hmm, something about numbers",
                                     _policy(backend), None)
        self.assertEqual(mode, route.FALLBACK_MODE)
        self.assertEqual(how, "menu-fallback")

    def test_prefix_carries_the_e7_contrast_examples(self):
        # E7's confusion table: every error funneled into compute/author
        # from navigate/edit; these contrast examples are the measured fix.
        self.assertIn("how is the discount computed", route.CODE_ROUTER_PREFIX)
        self.assertIn("debug why parse_row returns", route.CODE_ROUTER_PREFIX)
        self.assertIn("tell me which country ordered most",
                      route.CODE_ROUTER_PREFIX)


if __name__ == "__main__":
    unittest.main()
