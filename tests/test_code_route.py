"""Offline tests for the code-mode router (navigate pre-check + one menu)."""
import unittest

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code import route
from threetoks.policy import Policy, PolicyConfig


class PreRouteTest(unittest.TestCase):
    """Only read-only navigate questions may skip the menu."""

    def test_navigate_question_naming_a_class_routes_free(self):
        mode, _ = route.pre_route("where is the Cache class implemented?")
        self.assertEqual(mode, route.NAVIGATE)

    def test_navigate_question_naming_a_path_routes_free(self):
        mode, reason = route.pre_route("what does parser.py actually do?")
        self.assertEqual(mode, route.NAVIGATE)
        self.assertIn("path-like", reason)

    def test_navigate_question_with_an_edit_word_is_never_edit_free(self):
        # "add" is an edit cue: the request may mean "add a function"
        mode, _ = route.pre_route("where is the add function in parser.py?")
        self.assertIn(mode, (None, route.NAVIGATE))
        self.assertNotEqual(mode, route.EDIT)

    def test_edit_request_goes_to_the_menu(self):
        self.assertIsNone(route.pre_route("fix the bug in parser.py")[0])

    def test_author_request_goes_to_the_menu(self):
        self.assertIsNone(route.pre_route(
            "write a script that reverses a string")[0])

    def test_compute_request_goes_to_the_menu(self):
        self.assertIsNone(route.pre_route(
            "sum the second column of the csv")[0])

    def test_cue_words_match_whole_words_only(self):
        # "consume" holds "sum", "prefix" holds "fix": neither blocks
        mode, _ = route.pre_route("where is the consume_prefix function?")
        self.assertEqual(mode, route.NAVIGATE)

    def test_navigate_cue_without_code_entity_goes_to_the_menu(self):
        self.assertIsNone(route.pre_route("what is the weather today?")[0])

    def test_a_version_number_is_not_a_path(self):
        self.assertIsNone(route.pre_route("what is new in 3.12?")[0])

    def test_ambiguous_requests_fall_through(self):
        for request in ("hello there", "hmm, code stuff",
                        "this script is supposed to help"):
            self.assertIsNone(route.pre_route(request)[0], request)


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
    def test_free_navigate_skips_the_model_entirely(self):
        backend = _MenuBackend("edit")
        mode, how = route.route_mode("where is the retry function defined?",
                                     _policy(backend))
        self.assertEqual(mode, route.NAVIGATE)
        self.assertTrue(how.startswith("pre:"))
        self.assertEqual(backend.calls, 0)

    def test_author_request_costs_one_menu(self):
        backend = _MenuBackend("author")
        mode, how = route.route_mode("write a script that sorts numbers",
                                     _policy(backend))
        self.assertEqual((mode, how), (route.AUTHOR, "menu"))
        self.assertEqual(backend.calls, 1)

    def test_edit_request_is_decided_by_the_menu(self):
        backend = _MenuBackend("edit")
        mode, how = route.route_mode("fix the bug in parser.py",
                                     _policy(backend))
        self.assertEqual((mode, how), (route.EDIT, "menu"))

    def test_garbage_menu_reply_falls_back_to_navigate(self):
        backend = _MenuBackend("no-such-option")
        mode, how = route.route_mode("hmm, something about numbers",
                                     _policy(backend))
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
