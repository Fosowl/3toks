"""Offline tests for cross-session memory: store, selector, dispatch wiring."""
import json
import tempfile
import unittest
from pathlib import Path

from threetoks.agents.base import AgentSpec
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.memory import (MAX_PER_AGENT, MemoryStore, render_recall,
                             select_memories, worth_remembering)
from threetoks.policy import Policy, PolicyConfig
from threetoks.services import Services
from threetoks.tui import ReplState, _run_task


class ScriptedBackend:
    def __init__(self, texts):
        self.texts = list(texts)

    def complete(self, model, raw_prompt, opts):
        return GenResult(self.texts.pop(0) if self.texts else " 0", 5, 2, 0.0, "stop")


def make_policy(texts):
    return Policy(ScriptedBackend(texts),
                  PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


def filled_store():
    store = MemoryStore()
    store.remember("web", "population of France", "about 68 million")
    store.remember("code", "reverse a string", "def rev(s): return s[::-1]")
    return store


class MemoryStoreTest(unittest.TestCase):
    def test_remember_counts_across_agents(self):
        store = filled_store()
        self.assertEqual(store.count(), 2)

    def test_recent_is_newest_first_across_agents(self):
        store = filled_store()
        self.assertEqual([e["query"] for e in store.recent()],
                         ["reverse a string", "population of France"])

    def test_recent_respects_the_limit(self):
        store = MemoryStore()
        for i in range(5):
            store.remember("web", f"q{i}", f"a{i}")
        self.assertEqual([e["query"] for e in store.recent(limit=2)],
                         ["q4", "q3"])

    def test_dump_and_load_round_trip_preserves_recency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sub" / "mem.json")  # parent auto-created
            filled_store().dump(path)
            loaded = MemoryStore.load(path)
            self.assertEqual(loaded.count(), 2)
            self.assertEqual(loaded.recent()[0]["query"], "reverse a string")

    def test_ordinal_continues_after_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "mem.json")
            filled_store().dump(path)
            loaded = MemoryStore.load(path)
            loaded.remember("web", "capital of Japan", "Tokyo")
            self.assertEqual(loaded.recent()[0]["query"], "capital of Japan")

    def test_missing_or_broken_file_loads_empty(self):
        self.assertEqual(MemoryStore.load("/nonexistent/x.json").count(), 0)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            self.assertEqual(MemoryStore.load(str(bad)).count(), 0)

    def test_as_dict_shape_is_ordinal_and_per_agent(self):
        data = filled_store().as_dict()
        self.assertEqual(set(data), {"ordinal", "agents"})
        self.assertEqual(set(data["agents"]), {"web", "code"})

    def test_same_query_is_replaced_not_duplicated(self):
        store = filled_store()
        store.remember("web", "Population of France", "68.2 million now")
        self.assertEqual(store.count(), 2)  # case-folded match replaced
        self.assertEqual(store.recent()[0]["answer"], "68.2 million now")

    def test_per_agent_history_is_capped(self):
        store = MemoryStore()
        for i in range(MAX_PER_AGENT + 6):
            store.remember("web", f"question {i}", f"answer {i}")
        self.assertEqual(store.count(), MAX_PER_AGENT)
        queries = [e["query"] for e in store.recent(limit=MAX_PER_AGENT)]
        self.assertNotIn("question 0", queries)  # oldest evicted
        self.assertIn(f"question {MAX_PER_AGENT + 5}", queries)

    def test_relevant_ranks_by_overlap_then_recency(self):
        store = MemoryStore()
        store.remember("web", "france geography basics", "hills")
        store.remember("web", "population of France today", "68 million")
        store.remember("web", "reverse a string", "s[::-1]")
        ranked = store.relevant("france population growth")
        self.assertEqual([e["query"] for e in ranked],
                         ["population of France today",
                          "france geography basics"])  # 2 hits beat 1; no misses

    def test_relevant_matches_answer_text_too(self):
        store = MemoryStore()
        store.remember("web", "who wrote it", "agenticSeek by Fosowl")
        self.assertEqual(len(store.relevant("what is agenticSeek")), 1)

    def test_relevant_scans_the_whole_store_not_a_window(self):
        store = MemoryStore()
        store.remember("web", "population of France", "68 million")
        for i in range(40):  # newer chatter must not hide the old entry
            store.remember("casual", f"filler chatter number {i}", "ok")
        self.assertEqual(store.relevant("france population")[0]["query"],
                         "population of France")

    def test_short_word_task_falls_back_to_newest(self):
        ranked = filled_store().relevant("who is he")  # no 4+ char keywords
        self.assertEqual([e["query"] for e in ranked],
                         ["reverse a string", "population of France"])

    def test_stopwords_do_not_create_overlap(self):
        store = MemoryStore()
        store.remember("web", "what does this mean", "it means that thing")
        self.assertEqual(store.relevant("what makes rust fast"), [])

    def test_load_coerces_malformed_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.json"
            path.write_text(json.dumps({"ordinal": "7", "agents": {
                "web": [{"n": 1, "query": 123, "answer": None},
                        {"query": "ok question", "answer": "fine"},
                        "garbage"],
                "bad": "not a list"}}), encoding="utf-8")
            store = MemoryStore.load(str(path))
            self.assertEqual(store.count(), 2)  # coerced + valid, junk gone
            store.remember("web", "OK QUESTION", "updated")  # no crash
            self.assertEqual(store.count(), 2)  # replaced, not duplicated
            self.assertEqual(store.recent()[0]["answer"], "updated")


class WorthRememberingTest(unittest.TestCase):
    def test_real_answers_pass(self):
        self.assertTrue(worth_remembering({"answer": "Paris"}))
        self.assertTrue(worth_remembering({"answer": "42",
                                           "judged_good": True}))

    def test_empty_and_placeholder_answers_fail(self):
        self.assertFalse(worth_remembering({"answer": ""}))
        self.assertFalse(worth_remembering({}))
        self.assertFalse(worth_remembering({"answer": "(no answer)"}))

    def test_judge_rejected_answers_fail(self):
        self.assertFalse(worth_remembering({"answer": "plausible junk",
                                            "judged_good": False}))


class RenderRecallTest(unittest.TestCase):
    def test_block_lists_query_and_clipped_answer(self):
        block = render_recall([{"query": "population of France",
                                "answer": "about 68 million"}])
        self.assertTrue(block.startswith("EARLIER ANSWERS"))
        self.assertIn("population of France -> about 68 million", block)

    def test_empty_recall_renders_none(self):
        self.assertIsNone(render_recall([]))

    def test_multiline_answers_are_flattened(self):
        block = render_recall([{"query": "show code",
                                "answer": "def f():\n    return 1"}])
        self.assertEqual(len(block.splitlines()), 2)  # header + one entry
        self.assertIn("def f(): return 1", block)

    def test_selector_observation_keeps_one_line_per_candidate(self):
        from threetoks.memory import _selector_episode
        episode = _selector_episode("t", [{"query": "q one",
                                           "answer": "line one\nline two"}])
        self.assertIn("[1] q one -> line one line two",
                      episode.render_base())


class SelectMemoriesTest(unittest.TestCase):
    def test_picks_the_indexed_memory(self):
        chosen = select_memories("reverse text", filled_store(), make_policy([" 1"]))
        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0]["query"], "reverse a string")  # newest = 1

    def test_zero_means_none_selected(self):
        chosen = select_memories("population data", filled_store(),
                                 make_policy([" 0"]))
        self.assertEqual(chosen, [])

    def test_caps_at_three_picks(self):
        store = MemoryStore()
        for i in range(6):
            store.remember("web", f"france fact {i}", f"answer {i}")
        chosen = select_memories("france", store, make_policy([" 1,2,3,4,5"]))
        self.assertEqual(len(chosen), 3)

    def test_out_of_range_indices_are_dropped(self):
        chosen = select_memories("population info", filled_store(),
                                 make_policy([" 9"]))
        self.assertEqual(chosen, [])

    def test_empty_store_makes_no_model_call(self):
        backend = ScriptedBackend([])  # would raise IndexError if popped... it won't
        policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        self.assertEqual(select_memories("x", MemoryStore(), policy), [])
        self.assertEqual(len(backend.texts), 0)  # untouched, no call

    def test_no_keyword_overlap_makes_no_model_call(self):
        class ExplodingBackend:
            def complete(self, model, raw_prompt, opts):
                raise AssertionError("selector must not run")

        policy = Policy(ExplodingBackend(),
                        PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        self.assertEqual(
            select_memories("zebra migration", filled_store(), policy), [])

    def test_selector_only_sees_overlapping_candidates(self):
        # store has 2 entries but only one overlaps; index 1 = that one
        chosen = select_memories("reverse text", filled_store(),
                                 make_policy([" 1"]))
        self.assertEqual(chosen[0]["query"], "reverse a string")
        self.assertEqual(len(chosen), 1)


class DispatchWiringTest(unittest.TestCase):
    def _state(self, store, policy):
        answered = AgentSpec("web", "research", lambda t, s, p: {"answer": "42"})
        return ReplState(Services(), [answered], policy, "m", None,
                         enabled=False, memory=store, memory_path=""), answered

    def test_run_task_recalls_then_remembers(self):
        store = filled_store()
        policy = make_policy([" 1"])  # the recall selector picks memory 1
        state, spec = self._state(store, policy)
        _run_task(state, "please reverse a word", lambda t, s, p: spec, sink=[].append)
        self.assertEqual(len(state.services.recalled), 1)
        self.assertEqual(state.services.recalled[0]["query"], "reverse a string")
        self.assertEqual(store.count(), 3)  # the new turn was saved
        newest = store.recent()[0]
        self.assertEqual((newest["query"], newest["answer"]),
                         ("please reverse a word", "42"))

    def test_disabled_memory_neither_recalls_nor_saves(self):
        policy = make_policy([])  # no recall call expected
        state, spec = self._state(None, policy)
        _run_task(state, "hi", lambda t, s, p: spec, sink=[].append)
        self.assertEqual(state.services.recalled, [])  # untouched default

    def test_task_report_is_the_last_line_of_a_run(self):
        lines = []
        state, spec = self._state(filled_store(), make_policy([" 1"]))
        _run_task(state, "please reverse a word", lambda t, s, p: spec,
                  sink=lines.append)
        self.assertIn("task report", lines[-1])  # rendered below the panel

    def test_junk_outcomes_are_not_remembered(self):
        for outcome in ({"answer": "(no answer)"},
                        {"answer": ""},
                        {"answer": "junk", "judged_good": False}):
            store = filled_store()
            spec = AgentSpec("web", "research", lambda t, s, p, o=outcome: o)
            state = ReplState(Services(), [spec], make_policy([]), "m", None,
                              enabled=False, memory=store, memory_path="")
            _run_task(state, "zz", lambda t, s, p: spec, sink=[].append)
            self.assertEqual(store.count(), 2, outcome)  # nothing added


if __name__ == "__main__":
    unittest.main()
