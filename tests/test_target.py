"""Unit tests for the target-belief module (offline)."""
import unittest

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.nodes import MenuNode
from threetoks.policy import Policy, PolicyConfig
from threetoks.web.notes import NoteStore
from threetoks.web.target import (TAXONOMY, TargetBelief, normalize_query,
                                 strategy_queries, strip_quotes,
                                 subject_terms, too_similar, update_belief)


class ScriptedBackend:
    def __init__(self, texts):
        self.texts = list(texts)

    def complete(self, model, raw_prompt, opts):
        return GenResult(self.texts.pop(0), 5, 2, 0.0, "stop")


def make_policy(texts):
    return Policy(ScriptedBackend(texts),
                  PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


def menu_reply(option, options, step=0, attempt=0):
    """Digit that selects ``option`` under the policy's real permutation."""
    node = MenuNode("q", list(options), escape=False)
    perm = make_policy([])._permutation(node, step, attempt)
    return f" {perm.index(options.index(option)) + 1}"


def store_with(*texts):
    store = NoteStore()
    for index, text in enumerate(texts):
        store.add(text, "http://x", index)
    return store


class SubjectTermsTest(unittest.TestCase):
    def test_mid_sentence_capitalized_word_wins(self):
        self.assertEqual(
            subject_terms("don't code, search what Fosowl has build"),
            ["Fosowl"])

    def test_falls_back_to_longest_rare_words(self):
        terms = subject_terms("what is the biggest lighthouse ever built")
        self.assertEqual(terms[0], "lighthouse")

    def test_duplicates_collapse_case_insensitively(self):
        self.assertEqual(subject_terms("is Rust the Rust language"),
                         ["Rust"])


class TooSimilarTest(unittest.TestCase):
    def test_paraphrase_of_a_tried_query_is_similar(self):
        self.assertTrue(too_similar("Fosowl project overview",
                                    ["Fosowl project details"]))

    def test_new_angle_is_not_similar(self):
        self.assertFalse(too_similar("site:github.com Fosowl",
                                     ["Fosowl project details"]))

    def test_empty_query_counts_as_similar(self):
        self.assertTrue(too_similar("", ["anything"]))
        self.assertTrue(too_similar("the and for", ["anything"]))


class StrategyQueriesTest(unittest.TestCase):
    def test_github_belief_composes_site_filter(self):
        belief = TargetBelief(label=TAXONOMY[0])
        queries = strategy_queries("who is Fosowl", belief, ["who is Fosowl"])
        self.assertIn("site:github.com Fosowl", queries)

    def test_anchor_keyword_query_comes_first(self):
        belief = TargetBelief(label=TAXONOMY[0],
                              anchor="Fosowl /agenticSeek: open source agent")
        queries = strategy_queries("who is Fosowl", belief, [])
        self.assertEqual(queries[0], "Fosowl agenticSeek")

    def test_exhausted_label_angles_leave_no_strategies(self):
        belief = TargetBelief(label=TAXONOMY[0])
        queries = strategy_queries(
            "who is Fosowl", belief,
            ["site:github.com Fosowl", "Fosowl github repositories"])
        self.assertEqual(queries, [])  # caller falls back to a written query

    def test_no_belief_still_offers_generic_angles(self):
        queries = strategy_queries("who is Fosowl", TargetBelief(), [])
        self.assertTrue(any("github" in q for q in queries))
        self.assertLessEqual(len(queries), 5)

    def test_two_word_subject_keeps_one_word_templates(self):
        belief = TargetBelief(
            label=TAXONOMY[3],
            anchor="John Carmack founded Armadillo Aerospace")
        queries = strategy_queries("who is John Carmack", belief,
                                   ["who is John Carmack"])
        self.assertEqual(queries[0], "John Carmack Armadillo")
        self.assertGreaterEqual(len(queries), 3)

    def test_leading_proper_noun_survives_subject_extraction(self):
        self.assertEqual(
            subject_terms("Musk builds spaceships quickly")[0], "Musk")


class UpdateBeliefTest(unittest.TestCase):
    NOTES = ("Fosowl wrote agenticSeek, a local AI agent.",
             "The repository has twenty thousand stars.")

    def test_empty_notes_skip_the_model_entirely(self):
        belief = TargetBelief()
        updated = update_belief("q", belief, NoteStore(), make_policy([]))
        self.assertIs(updated, belief)

    def test_settled_belief_skips_the_model_entirely(self):
        belief = TargetBelief(label=TAXONOMY[0], settled=True)
        updated = update_belief("q", belief, store_with(*self.NOTES),
                                make_policy([]))
        self.assertIs(updated, belief)

    def test_first_pick_sets_label_and_anchor(self):
        reply = menu_reply(TAXONOMY[0], list(TAXONOMY))
        updated = update_belief("who is Fosowl", TargetBelief(),
                                store_with(*self.NOTES),
                                make_policy([reply, " 1"]))
        self.assertEqual(updated.label, TAXONOMY[0])
        self.assertEqual(updated.confirmations, 1)
        self.assertIn("agenticSeek", updated.anchor)
        self.assertFalse(updated.settled)

    def test_repicking_the_same_label_settles(self):
        reply = menu_reply(TAXONOMY[0], list(TAXONOMY))
        store = store_with(*self.NOTES)
        first = update_belief("q", TargetBelief(), store,
                              make_policy([reply, " 1"]))
        second = update_belief("q", first, store, make_policy([reply]))
        self.assertTrue(second.settled)
        self.assertEqual(second.confirmations, 2)
        self.assertEqual(second.anchor, first.anchor)  # no re-pick

    def test_escape_keeps_the_current_belief(self):
        escape_digit = f" {len(TAXONOMY) + 1}"
        belief = TargetBelief(label=TAXONOMY[1], confirmations=1)
        updated = update_belief("q", belief, store_with(*self.NOTES),
                                make_policy([escape_digit]))
        self.assertEqual(updated, belief)

    def test_invalid_answers_keep_the_current_belief(self):
        belief = TargetBelief(label=TAXONOMY[1], confirmations=1)
        updated = update_belief("q", belief, store_with(*self.NOTES),
                                make_policy(["x", "x", "x"]))
        self.assertEqual(updated, belief)


class BeliefLineTest(unittest.TestCase):
    def test_unknown_belief_has_no_line(self):
        self.assertIsNone(TargetBelief().line())

    def test_line_carries_label_and_anchor(self):
        line = TargetBelief(label=TAXONOMY[0], anchor="the repo").line()
        self.assertIn(TAXONOMY[0], line)
        self.assertIn("the repo", line)


class BannedQueriesTest(unittest.TestCase):
    def test_banned_key_strikes_out_a_strategy_candidate(self):
        belief = TargetBelief()
        free = strategy_queries("who is Fosowl?", belief, ["who is Fosowl?"])
        banned = frozenset({normalize_query(free[0])})
        remaining = strategy_queries("who is Fosowl?", belief,
                                     ["who is Fosowl?"], banned)
        self.assertNotIn(free[0], remaining)


class NormalizeQueryTest(unittest.TestCase):
    def test_typographic_quotes_do_not_defeat_dedup(self):
        self.assertEqual(normalize_query("“Best Pizza Antibes”"),
                         "best pizza antibes")
        self.assertEqual(normalize_query("‘best pizza antibes’"),
                         "best pizza antibes")

    def test_straight_quotes_case_and_spacing_collapse(self):
        self.assertEqual(normalize_query('  "Best  Pizza"  '), "best pizza")

    def test_strip_quotes_keeps_inner_text_intact(self):
        self.assertEqual(strip_quotes("«best pizza d'Antibes»"),
                         "best pizza d'Antibes")


if __name__ == "__main__":
    unittest.main()
