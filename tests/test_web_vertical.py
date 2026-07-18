"""Offline tests for the web-research state machine (fake IO, no LLM)."""
import unittest
from dataclasses import dataclass, field

from threetoks.nodes import ESCAPE, Decision, MenuNode, PickManyNode, ShortTextNode
from threetoks.web.notes import NoteStore
from threetoks.web.vertical import (AUTO_CHUNKS_PER_PAGE,
                                   FORCE_ANSWER_AT_STEPS_LEFT,
                                   MAX_REJECTED_QUERIES, MAX_SEARCHES,
                                   NOTE_MAX_PICKS, OPT_ANSWER, OPT_BACK,
                                   OPT_NEXT_CHUNK, OPT_NEW_SEARCH,
                                   SEARCH_UNAVAILABLE_ANSWER, SITES_PREFILL,
                                   WebResearchVertical, _extract_urls)


@dataclass(frozen=True)
class FakeResult:
    title: str
    url: str
    snippet: str = "some snippet"


@dataclass
class FakePage:
    title: str
    sentences: list
    url: str = ""


@dataclass
class FakeNotes:
    items: list = field(default_factory=list)

    def add(self, text, source_url, sentence_idx):
        self.items.append((text, source_url, sentence_idx))

    def count(self):
        return len(self.items)

    def render(self, max_notes=30):
        return "\n".join(f"[{i}] {t}" for i, (t, _, _) in
                         enumerate(self.items[:max_notes], 1))

    def texts(self):
        return [t for t, _, _ in self.items]


class FakeProvider:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, max_results=8):
        self.queries.append(query)
        return self.results


def menu_decision(value):
    return Decision("menu", value, value, valid=True)


def pick_decision(indices):
    return Decision("pick_many", list(indices), "", valid=True)


def make_vertical(sentences=("Paris is the capital of France.",
                             "France has 68 million people.")):
    provider = FakeProvider([FakeResult("Result A", "http://a.example"),
                             FakeResult("Result B", "http://b.example")])
    pages = {"http://a.example": FakePage("Page A", list(sentences)),
             "http://b.example": FakePage("Page B", list(sentences))}
    vertical = WebResearchVertical("population of France?", provider,
                                   lambda url: pages[url], FakeNotes())
    return vertical, provider


class DuplicateSearchTest(unittest.TestCase):
    def test_same_query_never_runs_twice(self):
        from threetoks.web.target import QUERY_TEMPERATURE
        vertical, provider = make_vertical()
        self.assertEqual(len(provider.queries), 1)
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision("search with different words"))
        node = vertical.next_node()
        self.assertIsInstance(node, ShortTextNode)
        self.assertEqual(node.temperature, QUERY_TEMPERATURE)
        vertical.apply(node, Decision("short_text",
                                      '"Population Of France?"', "", True))
        self.assertEqual(len(provider.queries), 1)  # normalized dup skipped
        self.assertIn("already searched", vertical.episode.render_base())

    def test_fresh_query_still_runs(self):
        vertical, provider = make_vertical()
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision("search with different words"))
        vertical.apply(vertical.next_node(),
                       Decision("short_text", "france census data", "", True))
        self.assertEqual(provider.queries[-1], "france census data")

    def test_shared_seen_urls_hide_exhausted_results(self):
        provider = FakeProvider([FakeResult("Result A", "http://a.example"),
                                 FakeResult("Result B", "http://b.example")])
        vertical = WebResearchVertical(
            "q", provider, lambda url: FakePage("A", []), FakeNotes(),
            seen_urls={"http://a.example"})
        menu = vertical.next_node()
        self.assertFalse(any("Result A" in option for option in menu.options))
        self.assertTrue(any("Result B" in option for option in menu.options))


class ContextLinesTest(unittest.TestCase):
    def test_context_lines_land_in_the_prompt_history(self):
        provider = FakeProvider([FakeResult("Result A", "http://a.example")])
        vertical = WebResearchVertical(
            "q", provider, lambda url: FakePage("A", []), FakeNotes(),
            context_lines=("EARLIER ANSWERS (past sessions):\n- q -> a",
                           "TARGET SO FAR: a github user"))
        prompt = vertical.episode.render_base()
        self.assertIn("TARGET SO FAR: a github user", prompt)
        self.assertIn("EARLIER ANSWERS", prompt)

    def test_no_context_adds_nothing(self):
        vertical, _ = make_vertical()
        self.assertNotIn("TARGET SO FAR", vertical.episode.render_base())


class CurateWindowTest(unittest.TestCase):
    def test_curate_node_windows_an_overflowing_store(self):
        from threetoks.render import Episode
        from threetoks.research import CURATE_POOL
        vertical = WebResearchVertical.__new__(WebResearchVertical)
        vertical.task = "q"
        vertical.episode = Episode("S", "q")
        store = NoteStore()
        for i in range(40):
            store.add(f"note {i}", "http://x", i)
        vertical.notes = store
        node = vertical._curate_node()
        self.assertEqual(node.n_items, CURATE_POOL)
        self.assertEqual(vertical.notes.count(), CURATE_POOL)
        self.assertEqual(vertical.notes.texts()[-1], "note 39")


class HappyPathTest(unittest.TestCase):
    def test_full_walk_results_page_autonotes_answer(self):
        vertical, _ = make_vertical()
        menu = vertical.next_node()
        self.assertIsInstance(menu, MenuNode)
        self.assertTrue(menu.options[0].startswith("open result 1: Result A"))

        vertical.apply(menu, menu_decision(menu.options[0]))
        pick = vertical.next_node()  # auto-note pass, no menu in between
        self.assertIsInstance(pick, PickManyNode)
        vertical.apply(pick, Decision("pick_many", [2], "2", valid=True))
        self.assertEqual(vertical.notes.items[0][2], 2)  # provenance idx

        page_menu = vertical.next_node()
        self.assertIsInstance(page_menu, MenuNode)
        self.assertIn(OPT_BACK, page_menu.options)
        vertical.apply(page_menu, menu_decision(OPT_ANSWER))
        synth = vertical.next_node()  # synthesis: generate from the notes
        self.assertIsInstance(synth, ShortTextNode)
        self.assertTrue(synth.prefill.startswith("FINAL"))
        vertical.apply(synth, Decision("short_text",
                                       "France has about 68 million people.",
                                       "", valid=True))
        self.assertIsNone(vertical.next_node())
        self.assertEqual(vertical.result()["answer"],
                         "France has about 68 million people.")

    def test_synthesis_prompt_grounds_in_collected_notes(self):
        vertical, _ = make_vertical()
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision(menu.options[0]))
        vertical.apply(vertical.next_node(),
                       Decision("pick_many", [1], "1", valid=True))
        vertical.apply(vertical.next_node(), menu_decision(OPT_ANSWER))
        synth = vertical.next_node()
        self.assertIn("Using ONLY the notes above", synth.question)
        self.assertIn("Paris is the capital of France.",
                      vertical.episode.render_base())

    def test_note_text_is_verbatim_sentence(self):
        vertical, _ = make_vertical()
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision(menu.options[0]))
        vertical.apply(vertical.next_node(),
                       Decision("pick_many", [1], "1", valid=True))
        self.assertEqual(vertical.notes.items[0][0],
                         "Paris is the capital of France.")


class GuardTest(unittest.TestCase):
    def test_escape_at_results_asks_for_new_query(self):
        vertical, provider = make_vertical()
        menu = vertical.next_node()
        vertical.apply(menu, Decision("menu", ESCAPE, "0", valid=True))
        query_node = vertical.next_node()
        self.assertIsInstance(query_node, ShortTextNode)
        vertical.apply(query_node, Decision("short_text", "france facts", "",
                                            valid=True))
        self.assertEqual(provider.queries[-1], "france facts")

    def test_search_budget_forces_answer_on_escape(self):
        vertical, _ = make_vertical()
        vertical.searches_done = MAX_SEARCHES
        menu = vertical.next_node()
        vertical.apply(menu, Decision("menu", ESCAPE, "0", valid=True))
        self.assertIsInstance(vertical.next_node(), ShortTextNode)
        self.assertTrue(vertical.next_node().prefill.startswith("FINAL"))

    def test_step_budget_forces_answer(self):
        vertical, _ = make_vertical()
        vertical.steps_left = FORCE_ANSWER_AT_STEPS_LEFT
        node = vertical.next_node()
        self.assertIsInstance(node, ShortTextNode)
        self.assertTrue(node.prefill.startswith("FINAL"))

    def test_fetch_failure_removes_option_from_menu(self):
        provider = FakeProvider([FakeResult("Broken", "http://broken"),
                                 FakeResult("Fine", "http://fine")])

        def failing_fetch(url):
            if url == "http://broken":
                raise IOError("boom")
            return FakePage("Fine", ["A sentence."])

        vertical = WebResearchVertical("t", provider, failing_fetch,
                                       FakeNotes())
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision(menu.options[0]))
        retry_menu = vertical.next_node()
        self.assertIsInstance(retry_menu, MenuNode)
        self.assertFalse(any("Broken" in opt for opt in retry_menu.options))

    def test_invalid_menu_decision_on_page_goes_back_to_results(self):
        vertical, _ = make_vertical()
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision(menu.options[0]))
        vertical.apply(vertical.next_node(),
                       Decision("pick_many", [1], "1", valid=True))
        page_menu = vertical.next_node()
        vertical.apply(page_menu, Decision("menu", None, "junk", valid=False))
        following = vertical.next_node()
        self.assertIsInstance(following, MenuNode)

    def test_menu_bleed_answer_gets_one_worded_retry(self):
        vertical, _ = make_vertical()
        vertical.steps_left = FORCE_ANSWER_AT_STEPS_LEFT
        final = vertical.next_node()
        vertical.apply(final, Decision("short_text", "1,2,3", "", valid=True))
        retry = vertical.next_node()
        self.assertIn("plain words", retry.question)
        vertical.apply(retry, Decision("short_text", "1,2,3", "", valid=True))
        self.assertEqual(vertical.result()["answer"], "(no answer)")

    def test_persistent_single_digit_is_accepted_as_answer(self):
        vertical, _ = make_vertical()
        vertical.steps_left = FORCE_ANSWER_AT_STEPS_LEFT
        final = vertical.next_node()
        vertical.apply(final, Decision("short_text", "2", "", valid=True))
        retry = vertical.next_node()
        vertical.apply(retry, Decision("short_text", "2", "", valid=True))
        self.assertEqual(vertical.result()["answer"], "2")

    def test_numeric_year_answer_is_not_flagged_as_bleed(self):
        vertical, _ = make_vertical()
        vertical.steps_left = FORCE_ANSWER_AT_STEPS_LEFT
        final = vertical.next_node()
        vertical.apply(final, Decision("short_text", "1989", "", valid=True))
        self.assertEqual(vertical.result()["answer"], "1989")

    def test_multi_digit_menu_bleed_gets_retry(self):
        """Two-digit numbers like '10' are caught as bleed, not leaked."""
        vertical, _ = make_vertical()
        vertical.steps_left = FORCE_ANSWER_AT_STEPS_LEFT
        final = vertical.next_node()
        vertical.apply(final, Decision("short_text", "10", "", valid=True))
        retry = vertical.next_node()
        self.assertIn("plain words", retry.question)
        vertical.apply(retry, Decision("short_text", "10", "", valid=True))
        self.assertEqual(vertical.result()["answer"], "(no answer)")

    def test_forced_next_node_is_idempotent(self):
        vertical, _ = make_vertical()
        vertical.steps_left = FORCE_ANSWER_AT_STEPS_LEFT
        first = vertical.next_node()
        history_len = len(vertical.episode._history)
        second = vertical.next_node()
        self.assertIs(first, second)
        self.assertEqual(len(vertical.episode._history), history_len)

    def test_seen_url_disappears_from_results_menu(self):
        vertical, _ = make_vertical()
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision(menu.options[0]))  # open A
        vertical.apply(vertical.next_node(),
                       Decision("pick_many", [1], "1", valid=True))
        vertical.apply(vertical.next_node(), menu_decision(OPT_BACK))
        menu2 = vertical.next_node()
        open_options = [o for o in menu2.options if o.startswith("open")]
        self.assertEqual(len(open_options), 1)
        self.assertIn("Result B", open_options[0])


if __name__ == "__main__":
    unittest.main()


class RelevanceFilterTest(unittest.TestCase):
    def test_zero_keyword_junk_never_shown(self):
        junk = ["arXiv is now an independent nonprofit today."] * 15
        gold = [f"Self evolving multi-agent systems adapt autonomously v{i}."
                for i in range(25)]
        vertical, _ = make_vertical()
        vertical.task = "self evolving multi-agent systems"
        kept = vertical._relevant_sentences(junk + gold)
        junk_kept = sum("nonprofit" in s for s in kept)
        self.assertLessEqual(junk_kept, 1)  # only a context neighbor survives
        self.assertEqual(sum("multi-agent" in s for s in kept), 25)

    def test_short_pages_pass_through(self):
        vertical, _ = make_vertical()
        vertical.task = "anything"
        few = ["One sentence.", "Two sentences."]
        self.assertEqual(vertical._relevant_sentences(few), few)


class ErrorPageTest(unittest.TestCase):
    def test_error_shell_page_treated_as_failed_fetch(self):
        provider = FakeProvider([FakeResult("Repo", "http://gh"),
                                 FakeResult("Fine", "http://fine")])
        pages = {"http://gh": FakePage("Repo", [
            "There was an error while loading.", "Please reload this page.",
            "You cannot perform that action at this time."]),
            "http://fine": FakePage("Fine", ["Real content here."])}
        vertical = WebResearchVertical("what is agenticseek", provider,
                                       lambda u: pages[u], FakeNotes())
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision(menu.options[0]))
        self.assertEqual(vertical.notes.count(), 0)  # nothing noted
        retry = vertical.next_node()
        self.assertFalse(any("Repo" in o for o in retry.options))

    def test_small_clean_pages_still_open(self):
        vertical, _ = make_vertical()
        menu = vertical.next_node()
        vertical.apply(menu, menu_decision(menu.options[0]))
        self.assertIsNotNone(vertical.page)


def make_long_vertical(n_sentences, notes=None, task="apples grow on trees"):
    """A vertical over one page of ``n_sentences`` task-keyword sentences."""
    sentences = [f"Apples grow on trees in orchard number {i} each year."
                 for i in range(n_sentences)]
    provider = FakeProvider([FakeResult("Orchard", "http://a.example")])
    pages = {"http://a.example": FakePage("Orchard", sentences)}
    vertical = WebResearchVertical(task, provider, lambda url: pages[url],
                                   notes if notes is not None else FakeNotes())
    return vertical


class AutoWalkTest(unittest.TestCase):
    def test_note_pass_uses_wide_cap(self):
        vertical = make_long_vertical(60)
        vertical.apply(vertical.next_node(),
                       menu_decision("open result 1: Orchard"))
        pick = vertical.next_node()
        self.assertIsInstance(pick, PickManyNode)
        self.assertEqual(pick.max_picks, NOTE_MAX_PICKS)

    def test_three_auto_pick_passes_then_menu(self):
        vertical = make_long_vertical(60)  # exactly 3 chunks of 25/25/10
        vertical.apply(vertical.next_node(),
                       menu_decision("open result 1: Orchard"))
        for _ in range(AUTO_CHUNKS_PER_PAGE):  # 3 successive auto pick passes
            pick = vertical.next_node()
            self.assertIsInstance(pick, PickManyNode)
            vertical.apply(pick, pick_decision([1]))
        after_walk = vertical.next_node()
        self.assertIsInstance(after_walk, MenuNode)  # walk budget spent

    def test_manual_read_more_continues_beyond_auto_walk(self):
        vertical = make_long_vertical(120)  # 5 chunks: auto-walk stops at 3
        vertical.apply(vertical.next_node(),
                       menu_decision("open result 1: Orchard"))
        for _ in range(AUTO_CHUNKS_PER_PAGE):
            vertical.apply(vertical.next_node(), pick_decision([1]))
        menu = vertical.next_node()
        self.assertIn(OPT_NEXT_CHUNK, menu.options)  # manual continuation
        vertical.apply(menu, menu_decision(OPT_NEXT_CHUNK))
        self.assertIsInstance(vertical.next_node(), PickManyNode)

    def test_short_page_has_no_auto_walk(self):
        vertical = make_long_vertical(10)  # single chunk, nothing beyond
        vertical.apply(vertical.next_node(),
                       menu_decision("open result 1: Orchard"))
        vertical.apply(vertical.next_node(), pick_decision([1]))
        following = vertical.next_node()
        self.assertIsInstance(following, MenuNode)  # straight to page menu


class CurationTest(unittest.TestCase):
    def _store_with(self, count):
        store = NoteStore()
        for i in range(count):
            store.add(f"Evidence fact number {i} about the topic.",
                      "http://a.example", i)
        return store

    def test_curate_ranks_by_model_pick_order(self):
        store = self._store_with(12)
        vertical = make_long_vertical(4, notes=store)
        vertical.notes = store  # concrete store supports select()
        node = vertical._answer_flow()  # count > CURATE_KEEP -> curate first
        self.assertIsInstance(node, PickManyNode)
        self.assertEqual(node.tag, "curate")
        vertical.apply(node, pick_decision([3, 1, 7]))
        self.assertEqual(vertical.notes.count(), 3)
        picked = [n.text for n in vertical.notes.entries()]
        self.assertEqual(picked, ["Evidence fact number 2 about the topic.",
                                  "Evidence fact number 0 about the topic.",
                                  "Evidence fact number 6 about the topic."])

    def test_invalid_curation_keeps_first_curate_keep(self):
        store = self._store_with(12)
        vertical = make_long_vertical(4, notes=store)
        vertical.notes = store
        node = vertical._answer_flow()
        vertical.apply(node, Decision("pick_many", None, "junk", valid=False))
        self.assertEqual(vertical.notes.count(), 8)  # CURATE_KEEP

    def test_small_store_skips_curation(self):
        store = self._store_with(5)
        vertical = make_long_vertical(4, notes=store)
        vertical.notes = store
        node = vertical._answer_flow()  # <= CURATE_KEEP -> straight to synth
        self.assertIsInstance(node, ShortTextNode)

    def test_result_notes_render_curated_store(self):
        store = self._store_with(12)
        vertical = make_long_vertical(4, notes=store)
        vertical.notes = store
        vertical.apply(vertical._answer_flow(), pick_decision([2, 4]))
        rendered = vertical.result()["notes"]
        self.assertEqual(len(rendered.splitlines()), 2)


class DeadProvider:
    """Every hop failed: returns [] and reports the layer down."""

    def __init__(self):
        self.queries = []
        self.consecutive_failures = 0
        self.last_call_failed = False

    def search(self, query, max_results=8):
        self.queries.append(query)
        self.consecutive_failures += 1
        self.last_call_failed = True
        return []


def text_decision(text):
    return Decision("short_text", text, text, valid=True)


class RejectionBudgetTest(unittest.TestCase):
    def _write_query(self, vertical, text):
        vertical.apply(vertical.next_node(),
                       menu_decision(OPT_NEW_SEARCH))
        vertical.apply(vertical.next_node(), text_decision(text))

    def test_duplicate_query_costs_no_search_budget(self):
        vertical, provider = make_vertical()
        searches_before = vertical.searches_done
        self._write_query(vertical, "“Population Of France?”")  # curly dup
        self.assertEqual(vertical.searches_done, searches_before)
        self.assertEqual(len(provider.queries), 1)
        self.assertIn("already searched", vertical.episode.render_base())

    def test_paraphrase_query_is_rejected_in_round(self):
        vertical, provider = make_vertical()
        self._write_query(vertical, "population of France reviews")
        self.assertEqual(len(provider.queries), 1)
        self.assertIn("rehashes", vertical.episode.render_base())

    def test_rejection_budget_removes_new_search_option(self):
        vertical, _ = make_vertical()
        self._write_query(vertical, "“Population Of France?”")
        self._write_query(vertical, "population of France reviews")
        self.assertEqual(vertical._rejected_queries, MAX_REJECTED_QUERIES)
        menu = vertical.next_node()
        self.assertNotIn(OPT_NEW_SEARCH, menu.options)


class DegenerateMenuTest(unittest.TestCase):
    def test_lone_new_search_option_skips_the_menu(self):
        vertical = WebResearchVertical("t", FakeProvider([]),
                                       lambda url: FakePage("A", []),
                                       FakeNotes())
        node = vertical.next_node()  # no results, no notes: only move
        self.assertIsInstance(node, ShortTextNode)
        self.assertEqual(node.prefill, "QUERY:")

    def test_exhausted_round_ends_without_a_model_call(self):
        vertical = WebResearchVertical("t", FakeProvider([]),
                                       lambda url: FakePage("A", []),
                                       FakeNotes())
        vertical.searches_done = MAX_SEARCHES
        self.assertIsNone(vertical.next_node())
        self.assertEqual(vertical.result()["answer"], "(no answer)")

    def test_exhausted_round_names_a_dead_search_layer(self):
        provider = DeadProvider()
        vertical = WebResearchVertical("t", provider,
                                       lambda url: FakePage("A", []),
                                       FakeNotes())
        vertical.apply(vertical.next_node(), text_decision("no urls"))
        vertical.searches_done = MAX_SEARCHES
        self.assertIsNone(vertical.next_node())
        self.assertEqual(vertical.result()["answer"],
                         SEARCH_UNAVAILABLE_ANSWER)


class SiteFallbackTest(unittest.TestCase):
    def test_dead_layer_triggers_one_site_ask(self):
        provider = DeadProvider()
        pages = {"https://tripadvisor.com/antibes":
                 FakePage("TA", ["Chez Nino makes the best pizza."])}
        vertical = WebResearchVertical("best pizza in antibes", provider,
                                       lambda url: pages[url], FakeNotes())
        node = vertical.next_node()
        self.assertIsInstance(node, ShortTextNode)
        self.assertEqual(node.prefill, SITES_PREFILL)
        self.assertIn("search engines are unavailable",
                      vertical.episode.render_base())
        vertical.apply(node, text_decision(
            "try tripadvisor.com/antibes or yelp.com"))
        menu = vertical.next_node()
        self.assertIsInstance(menu, MenuNode)
        self.assertTrue(menu.options[0].startswith(
            "open result 1: https://tripadvisor.com/antibes"))
        vertical.apply(menu, menu_decision(menu.options[0]))
        self.assertIsInstance(vertical.next_node(), PickManyNode)

    def test_site_ask_happens_only_once(self):
        provider = DeadProvider()
        vertical = WebResearchVertical("t", provider,
                                       lambda url: FakePage("A", []),
                                       FakeNotes())
        sites = vertical.next_node()
        vertical.apply(sites, text_decision("no address here"))
        query_node = vertical.next_node()  # no urls parsed -> write a query
        self.assertEqual(query_node.prefill, "QUERY:")
        vertical.apply(query_node, text_decision("alpha beta"))
        self.assertNotEqual(vertical.next_node().prefill, SITES_PREFILL)

    def test_extract_urls_normalizes_and_dedupes(self):
        urls = _extract_urls("see tripadvisor.com/x, https://yelp.com; "
                             "tripadvisor.com/x again")
        self.assertEqual(urls, ["https://tripadvisor.com/x",
                                "https://yelp.com"])

    def test_extract_urls_skips_bare_file_names(self):
        self.assertEqual(
            _extract_urls("check README.md, node.js and yelp.com"),
            ["https://yelp.com"])


class SynthesisTest(unittest.TestCase):
    def test_synthesis_answer_is_generated_words(self):
        store = NoteStore()
        store.add("France has about 68 million residents.", "http://a", 1)
        vertical = make_long_vertical(4, notes=store)
        vertical.notes = store
        synth = vertical._answer_flow()  # small store -> synthesis directly
        self.assertIsInstance(synth, ShortTextNode)
        vertical.apply(synth, Decision("short_text", "About 68 million.",
                                       "", valid=True))
        self.assertEqual(vertical.result()["answer"], "About 68 million.")

    def test_double_bleed_falls_back_to_quoting_top_notes(self):
        store = NoteStore()
        store.add("France has about 68 million residents.", "http://a", 1)
        store.add("Paris is its capital city.", "http://a", 2)
        store.add("The Seine flows through it.", "http://a", 3)
        vertical = make_long_vertical(4, notes=store)
        vertical.notes = store
        synth = vertical._answer_flow()
        vertical.apply(synth, Decision("short_text", "1,2", "", valid=True))
        retry = vertical.next_node()
        vertical.apply(retry, Decision("short_text", "3,4", "", valid=True))
        self.assertEqual(vertical.result()["answer"],
                         "France has about 68 million residents.\n"
                         "Paris is its capital city.")

    def test_double_bleed_without_notes_gives_no_answer(self):
        vertical = make_long_vertical(4, notes=NoteStore())
        vertical.notes = NoteStore()
        synth = vertical._answer_flow()
        vertical.apply(synth, Decision("short_text", "1,2", "", valid=True))
        vertical.apply(vertical.next_node(),
                       Decision("short_text", "3,4", "", valid=True))
        self.assertEqual(vertical.result()["answer"], "(no answer)")
