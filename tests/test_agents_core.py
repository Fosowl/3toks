"""Tests for the agent contract, router, deep research, link following."""
import unittest

from threetoks.agents.base import AgentSpec
from threetoks.agents.router import route
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.nodes import Decision, MenuNode
from threetoks.policy import Policy, PolicyConfig
from threetoks.render import Episode
from threetoks.research import (CURATE_KEEP, OPT_WRITE_QUERY, curate_notes,
                               deep_research, judge_answer, propose_query)
from threetoks.services import Services
from threetoks.web.notes import NoteStore
from threetoks.web.target import TAXONOMY, TargetBelief, strategy_queries
from threetoks.web.vertical import OPT_LINKS, WebResearchVertical


def menu_reply(option, options, step=0, attempt=0):
    """Digit that selects ``option`` under the policy's real permutation."""
    node = MenuNode("q", list(options), escape=False)
    perm = make_policy([])._permutation(node, step, attempt)
    return f" {perm.index(options.index(option)) + 1}"


class ScriptedBackend:
    def __init__(self, texts):
        self.texts = list(texts)

    def complete(self, model, raw_prompt, opts):
        return GenResult(self.texts.pop(0), 5, 2, 0.0, "stop")


def make_policy(texts):
    return Policy(ScriptedBackend(texts),
                  PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


SPECS = [AgentSpec("casual", "small talk", lambda t, s, p: {"answer": "hey"}),
         AgentSpec("web", "internet research", lambda t, s, p: {"answer": ""})]


class RouterTest(unittest.TestCase):
    def test_valid_choice_routes_to_an_agent(self):
        chosen = route("hello!", SPECS, make_policy([" 1"]))
        self.assertIn(chosen.name, {"casual", "web"})

    def test_garbage_choice_falls_back_to_first_spec(self):
        chosen = route("hello!", SPECS, make_policy(["x", "x", "x"]))
        self.assertEqual(chosen.name, "casual")

    def test_single_spec_skips_the_model(self):
        chosen = route("hi", SPECS[:1], make_policy([]))
        self.assertEqual(chosen.name, "casual")


class RouterPrefixTest(unittest.TestCase):
    """The few-shot examples must stay internally consistent."""

    EXPECTED = {"hey there": "casual", "Ada Lovelace": "web",
                "FooBar project": "web", "reverses a string": "code",
                "notes folder": "files", "housing market": "web",
                "economy projections": "web", "Marianas trench": "web"}

    def test_every_example_answer_points_at_the_right_agent(self):
        import re
        from threetoks.agents.router import ROUTER_PREFIX
        blocks = ROUTER_PREFIX.split("Example — ")[1:]
        self.assertEqual(len(blocks), len(self.EXPECTED))
        for block in blocks:
            options = dict(re.findall(r"(\d) = (\w+)", block))
            answer = re.search(r"ANSWER: (\d)", block).group(1)
            request = re.search(r"Request: (.+)", block).group(1)
            expected = next(agent for key, agent in self.EXPECTED.items()
                            if key in request)
            self.assertEqual(options[answer], expected, request)

    def test_examples_cover_search_and_person_lookup_phrasings(self):
        from threetoks.agents.router import ROUTER_PREFIX
        self.assertIn("deep search", ROUTER_PREFIX)
        self.assertIn("who is", ROUTER_PREFIX)
        self.assertIn("search what", ROUTER_PREFIX)


class ResearchTest(unittest.TestCase):
    def test_propose_query_avoids_repeats(self):
        policy = make_policy([" population of France"])
        query = propose_query("task", ["population of France"], policy)
        self.assertNotIn(query, ["population of France"])
        self.assertTrue(query.startswith("task "))

    def test_fallback_query_never_repeats_either(self):
        tried = ["t details", "t overview"]
        query = propose_query("t", tried, make_policy([" t details"]))
        self.assertNotIn(query, tried)

    def test_banned_executed_queries_are_never_proposed(self):
        from threetoks.web.target import normalize_query
        policy = make_policy([" Fosowl github"])
        query = propose_query("who is Fosowl?", ["who is Fosowl?"], policy,
                              banned={normalize_query("Fosowl github")})
        self.assertNotEqual(normalize_query(query), "fosowl github")

    def test_numeric_answers_reach_the_judge(self):
        from threetoks.research import _obviously_bad
        for answer in ("42", "1989", "3.14"):
            self.assertFalse(_obviously_bad(
                {"answer": answer, "notes": "[1] evidence"}), answer)
        self.assertTrue(_obviously_bad(
            {"answer": "1,2,3", "notes": "[1] evidence"}))

    def test_zero_rounds_config_still_returns_full_result(self):
        class NoResults:
            def search(self, query, max_results=8):
                return []

        services = Services(provider=NoResults(), fetch_page=None,
                            max_research_rounds=0)
        result = deep_research("q", services, make_policy(
            [" 1", " some answer"] + [" 1"] * 12))
        self.assertIn("rounds", result)
        self.assertIn("queries", result)

    def test_judge_answer_returns_bool(self):
        self.assertIsInstance(
            judge_answer("q", {"answer": "a", "notes": ""},
                         make_policy([" 1"])), bool)

    def test_curate_ranks_notes_in_model_pick_order(self):
        store = NoteStore()
        for i in range(12):  # > CURATE_KEEP so curation actually runs
            store.add(f"fact number {i}", "http://x", i)
        ranked = curate_notes("task", store, make_policy([" 3,1,7"]))
        self.assertEqual(ranked.count(), 3)
        self.assertEqual([n.text for n in ranked.entries()],
                         ["fact number 2", "fact number 0", "fact number 6"])

    def test_curate_leaves_small_store_untouched(self):
        store = NoteStore()
        for i in range(CURATE_KEEP):  # == CURATE_KEEP -> no model call
            store.add(f"fact {i}", "http://x", i)
        self.assertIs(curate_notes("task", store, make_policy([])), store)

    def test_deep_research_stops_when_judge_accepts(self):
        class OneResult:
            def search(self, query, max_results=8):
                return []

        def fake_judge(task, result, policy):
            return True

        import threetoks.research as research_module
        original = research_module.judge_answer
        research_module.judge_answer = fake_judge
        try:
            services = Services(provider=OneResult(), fetch_page=None,
                                max_research_rounds=3)
            # no results -> forced empty answer path via synthesis node
            result = deep_research("q", services, make_policy(
                [" 1", " some answer"] + [" 1"] * 8))
            self.assertEqual(result["rounds"], 1)
            self.assertTrue(result["judged_good"])
        finally:
            research_module.judge_answer = original


class SearchBreakerTest(unittest.TestCase):
    def test_dead_search_layer_aborts_rounds_with_honest_answer(self):
        import threetoks.research as research_module
        from threetoks.web.vertical import SEARCH_UNAVAILABLE_ANSWER
        rounds_run = []

        class FakeVertical:
            def __init__(self, task, provider, fetch_page, notes,
                         initial_query=None, context_lines=(),
                         seen_urls=None, tried_queries=None):
                rounds_run.append(initial_query)
                self.episode = Episode("S", task)
                self.steps_left = 0
                self.notes = notes

            def next_node(self):
                return None

            def result(self):
                return {"answer": "(no answer)", "notes": ""}

        class DeadProvider:
            consecutive_failures = research_module.SEARCH_ABORT_FAILURES

        def no_requery(*args, **kwargs):
            raise AssertionError("requery must not run after the breaker")

        originals = (research_module.WebResearchVertical,
                     research_module.propose_query)
        research_module.WebResearchVertical = FakeVertical
        research_module.propose_query = no_requery
        try:
            services = Services(provider=DeadProvider(), fetch_page=None,
                                max_research_rounds=3)
            result = deep_research("q", services, make_policy([]))
        finally:
            (research_module.WebResearchVertical,
             research_module.propose_query) = originals
        self.assertTrue(result["search_unavailable"])
        self.assertEqual(result["answer"], SEARCH_UNAVAILABLE_ANSWER)
        self.assertFalse(result["judged_good"])
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(rounds_run, ["q"])

    def test_gathered_notes_keep_the_rounds_running(self):
        import threetoks.research as research_module

        class FakeVertical:
            def __init__(self, task, provider, fetch_page, notes,
                         initial_query=None, context_lines=(),
                         seen_urls=None, tried_queries=None):
                self.episode = Episode("S", task)
                self.steps_left = 0
                self.notes = notes
                notes.add("some evidence", "http://x", 1)

            def next_node(self):
                return None

            def result(self):
                return {"answer": "a", "notes": "[1] some evidence"}

        class DeadProvider:
            consecutive_failures = research_module.SEARCH_ABORT_FAILURES

        originals = (research_module.WebResearchVertical,
                     research_module.judge_answer,
                     research_module.update_belief,
                     research_module.propose_query)
        research_module.WebResearchVertical = FakeVertical
        research_module.judge_answer = lambda t, r, p: False
        research_module.update_belief = lambda t, b, n, p: b
        research_module.propose_query = lambda t, tr, p, b=None, **kw: f"q{len(tr)}"
        try:
            services = Services(provider=DeadProvider(), fetch_page=None,
                                max_research_rounds=2)
            result = deep_research("q", services, make_policy([]))
        finally:
            (research_module.WebResearchVertical,
             research_module.judge_answer,
             research_module.update_belief,
             research_module.propose_query) = originals
        self.assertEqual(result["rounds"], 2)  # notes exist: no early abort
        self.assertNotIn("search_unavailable", result)


class BeliefRequeryTest(unittest.TestCase):
    def test_written_paraphrase_falls_back_to_a_strategy(self):
        query = propose_query("find Fosowl", ["Fosowl project details"],
                              make_policy([" Fosowl project overview"]))
        self.assertEqual(query, "Fosowl github")  # composed, not a paraphrase

    def test_belief_strategy_menu_returns_composed_query(self):
        belief = TargetBelief(label=TAXONOMY[0])
        tried = ["who is Fosowl"]
        strategies = strategy_queries("who is Fosowl", belief, tried)
        options = [f"search: {q}" for q in strategies] + [OPT_WRITE_QUERY]
        reply = menu_reply(f"search: {strategies[0]}", options)
        query = propose_query("who is Fosowl", tried,
                              make_policy([reply]), belief)
        self.assertEqual(query, strategies[0])

    def test_strategy_menu_write_option_falls_through(self):
        belief = TargetBelief(label=TAXONOMY[0])
        tried = ["who is Fosowl"]
        strategies = strategy_queries("who is Fosowl", belief, tried)
        options = [f"search: {q}" for q in strategies] + [OPT_WRITE_QUERY]
        reply = menu_reply(OPT_WRITE_QUERY, options)
        query = propose_query(
            "who is Fosowl", tried,
            make_policy([reply, " Fosowl agenticSeek github"]), belief)
        self.assertEqual(query, "Fosowl agenticSeek github")

    def test_angle_hints_are_unique_and_plentiful(self):
        from threetoks.research import ANGLE_HINTS
        self.assertEqual(len(ANGLE_HINTS), len(set(ANGLE_HINTS)))
        self.assertGreaterEqual(len(ANGLE_HINTS), 10)  # outlasts any budget

    def test_written_query_runs_hot_with_rotating_angle_hint(self):
        from threetoks.research import ANGLE_HINTS
        from threetoks.web.target import QUERY_TEMPERATURE

        class RecordingBackend:
            def __init__(self):
                self.prompts, self.temps = [], []

            def complete(self, model, raw_prompt, opts):
                self.prompts.append(raw_prompt)
                self.temps.append(opts.temperature)
                return GenResult(" solid new query", 5, 2, 0.0, "stop")

        backend = RecordingBackend()
        policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        propose_query("find it", ["find it"], policy)
        self.assertEqual(backend.temps, [QUERY_TEMPERATURE])
        self.assertIn(ANGLE_HINTS[1 % len(ANGLE_HINTS)], backend.prompts[0])
        propose_query("find it", ["find it", "other query"], policy)
        self.assertIn(ANGLE_HINTS[2 % len(ANGLE_HINTS)], backend.prompts[1])

    def test_rounds_share_the_seen_url_and_query_sets(self):
        import threetoks.research as research_module
        shared = []

        class FakeVertical:
            def __init__(self, task, provider, fetch_page, notes,
                         initial_query=None, context_lines=(),
                         seen_urls=None, tried_queries=None):
                shared.append((seen_urls, tried_queries))
                self.episode = Episode("S", task)
                self.steps_left = 0
                self.notes = notes

            def next_node(self):
                return None

            def result(self):
                return {"answer": "x", "notes": "[1] x"}

        originals = (research_module.WebResearchVertical,
                     research_module.judge_answer,
                     research_module.update_belief,
                     research_module.propose_query)
        research_module.WebResearchVertical = FakeVertical
        research_module.judge_answer = lambda t, r, p: False
        research_module.update_belief = lambda t, b, n, p: b
        research_module.propose_query = lambda t, tr, p, b=None, **kw: "next"
        try:
            services = Services(provider=None, fetch_page=None,
                                max_research_rounds=2)
            deep_research("q", services, make_policy([]))
        finally:
            (research_module.WebResearchVertical,
             research_module.judge_answer,
             research_module.update_belief,
             research_module.propose_query) = originals
        self.assertIs(shared[0][0], shared[1][0])  # one seen-URL set
        self.assertIs(shared[0][1], shared[1][1])  # one tried-query set

    def test_rounds_carry_recall_and_updated_target_context(self):
        import threetoks.research as research_module
        from threetoks.memory import render_recall
        contexts = []

        class FakeVertical:
            def __init__(self, task, provider, fetch_page, notes,
                         initial_query=None, context_lines=(),
                         seen_urls=None, tried_queries=None):
                contexts.append(tuple(context_lines))
                self.episode = Episode("S", task)
                self.steps_left = 0
                self.notes = notes

            def next_node(self):
                return None

            def result(self):
                return {"answer": "x", "notes": "[1] x"}

        belief = TargetBelief(label=TAXONOMY[0], anchor="agenticSeek")
        originals = (research_module.WebResearchVertical,
                     research_module.judge_answer,
                     research_module.update_belief,
                     research_module.propose_query)
        research_module.WebResearchVertical = FakeVertical
        research_module.judge_answer = lambda t, r, p: False
        research_module.update_belief = lambda t, b, n, p: belief
        research_module.propose_query = lambda t, tr, p, b=None, **kw: "next query"
        try:
            services = Services(provider=None, fetch_page=None,
                                max_research_rounds=2,
                                recalled=[{"query": "who is Fosowl",
                                           "answer": "a github user"}])
            result = deep_research("q", services, make_policy([]))
        finally:
            (research_module.WebResearchVertical,
             research_module.judge_answer,
             research_module.update_belief,
             research_module.propose_query) = originals
        recall = render_recall(services.recalled)
        tried = "QUERIES ALREADY TRIED (do not repeat): q"
        self.assertEqual(contexts, [(recall,),
                                    (recall, belief.line(), tried)])
        self.assertEqual(result["target"],
                         {"label": TAXONOMY[0], "anchor": "agenticSeek"})
        self.assertEqual(result["queries"], ["q", "next query"])


class CurationWindowTest(unittest.TestCase):
    def test_pool_keeps_head_keepers_and_newest_notes(self):
        from threetoks.research import CURATE_POOL, curation_pool
        store = NoteStore()
        for i in range(40):
            store.add(f"note {i}", "http://x", i)
        window = curation_pool(store)
        self.assertEqual(window.count(), CURATE_POOL)
        texts = window.texts()
        self.assertEqual(texts[0], "note 0")      # prior ranked keepers
        self.assertEqual(texts[-1], "note 39")    # newest evidence
        self.assertNotIn("note 10", texts)        # stale middle dropped

    def test_pool_leaves_small_store_untouched(self):
        from threetoks.research import curation_pool
        store = NoteStore()
        store.add("only note", "http://x", 1)
        self.assertIs(curation_pool(store), store)

    def test_rounds_adopt_the_verticals_pruned_store(self):
        import threetoks.research as research_module
        counts_seen = []

        class FakeVertical:
            def __init__(self, task, provider, fetch_page, notes,
                         initial_query=None, context_lines=(),
                         seen_urls=None, tried_queries=None):
                counts_seen.append(notes.count())
                self.episode = Episode("S", task)
                self.steps_left = 0
                pruned = NoteStore()
                pruned.add("kept note", "http://x", 1)
                self.notes = pruned  # what in-episode curation left behind

            def next_node(self):
                return None

            def result(self):
                return {"answer": "x", "notes": "[1] x"}

        originals = (research_module.WebResearchVertical,
                     research_module.judge_answer,
                     research_module.update_belief,
                     research_module.propose_query)
        research_module.WebResearchVertical = FakeVertical
        research_module.judge_answer = lambda t, r, p: False
        research_module.update_belief = lambda t, b, n, p: b
        research_module.propose_query = lambda t, tr, p, b=None, **kw: "next"
        try:
            services = Services(provider=None, fetch_page=None,
                                max_research_rounds=2)
            deep_research("q", services, make_policy([]))
        finally:
            (research_module.WebResearchVertical,
             research_module.judge_answer,
             research_module.update_belief,
             research_module.propose_query) = originals
        self.assertEqual(counts_seen, [0, 1])


class LinkFollowTest(unittest.TestCase):
    def test_page_menu_offers_links_and_opens_target(self):
        from dataclasses import dataclass, field

        @dataclass(frozen=True)
        class L:
            label: str
            url: str

        @dataclass
        class Page:
            title: str
            sentences: list
            links: list = field(default_factory=list)
            url: str = ""

        @dataclass(frozen=True)
        class R:
            title: str
            url: str
            snippet: str = "s"

        class Provider:
            def search(self, query, max_results=8):
                return [R("A", "http://a")]

        pages = {"http://a": Page("A", ["One fact here."],
                                  [L("pricing page", "http://a/pricing")]),
                 "http://a/pricing": Page("Pricing", ["Costs 3 dollars."])}

        class Notes:
            def __init__(self):
                self.items = []

            def add(self, t, u, i):
                self.items.append(t)

            def count(self):
                return len(self.items)

            def render(self, max_notes=30):
                return ""

            def texts(self):
                return list(self.items)

        vertical = WebResearchVertical("price?", Provider(),
                                       lambda u: pages[u], Notes())
        menu = vertical.next_node()
        vertical.apply(menu, Decision("menu", menu.options[0], "", True))
        vertical.apply(vertical.next_node(),
                       Decision("pick_many", [1], "", True))
        page_menu = vertical.next_node()
        self.assertIn(OPT_LINKS, page_menu.options)
        vertical.apply(page_menu, Decision("menu", OPT_LINKS, "", True))
        links_menu = vertical.next_node()
        self.assertIsInstance(links_menu, MenuNode)
        self.assertTrue(links_menu.options[0].startswith("go: pricing"))
        vertical.apply(links_menu,
                       Decision("menu", links_menu.options[0], "", True))
        self.assertEqual(vertical.pages_opened, 2)
        self.assertEqual(vertical.page.title, "Pricing")


class BestChunkTest(unittest.TestCase):
    def test_page_opens_at_task_relevant_chunk(self):
        sentences = ["Filler sentence about nothing much here."] * 30 \
            + ["The exact pricing is 3 dollars per million tokens."] * 5
        vertical = WebResearchVertical.__new__(WebResearchVertical)
        vertical.task = "what is the exact pricing?"
        start = vertical._best_chunk_start(sentences)
        self.assertGreaterEqual(start, 25)


if __name__ == "__main__":
    unittest.main()


class ResolvedPickTraceTest(unittest.TestCase):
    def test_pick_events_carry_resolved_texts(self):
        from threetoks.nodes import PickManyNode
        from threetoks.render import Episode
        from threetoks.trace import Tracer
        events = []
        policy = Policy(ScriptedBackend(["2"]),
                        PolicyConfig(ModelSpec("m", FAMILY_CHATML)),
                        Tracer(None, on_event=events.append))
        node = PickManyNode("Which?", 3, items=["alpha", "beta", "gamma"])
        decision = policy.decide(Episode("S", "t"), node)
        self.assertEqual(decision.value, [2])
        self.assertEqual(events[0]["resolved"], ["beta"])

    def test_ticker_shows_resolved_text(self):
        from threetoks.tui import build_ticker_line
        line = build_ticker_line({"node": "pick_many", "value": [2, 5],
                                  "valid": True, "resolved":
                                  ["France has 68m people.", "Paris is big."],
                                  "out_tokens": 5, "wall_s": 0.4},
                                 enabled=False)
        self.assertIn("France has 68m people.", line)
        self.assertIn("Paris is big.", line)  # every pick gets its own line
        self.assertEqual(len(line.splitlines()), 3)
