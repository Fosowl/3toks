"""Offline tests for the live progress tree: rows, branches, /debug dumps."""
import unittest
from types import SimpleNamespace

from threetoks import progress
from threetoks.theme import AMBER, GREEN, PINK


def make_tree(debug=False, debug_full=False):
    """A plain-text tree that keeps every shown row in ``tree.rows``."""
    rows: list[str] = []
    view = SimpleNamespace(enabled=False, debug=debug, debug_full=debug_full)
    tree = progress.ProgressTree(view, rows.append)
    tree.rows = rows
    return tree


def decision(value="go", **extra):
    """A valid decision event; a plain menu choice unless overridden."""
    return {"node": "menu", "value": value, "valid": True,
            "out_tokens": 2, "wall_s": 0.1, **extra}


def joint_column(row: str) -> int:
    """Column of the row's own joint: deeper rows have it further right."""
    return max(row.rfind(progress.JOINT), row.rfind(progress.END))


class DecisionRowTest(unittest.TestCase):
    def test_decision_shows_kind_value_and_stats(self):
        line = progress.build_decision_row(decision("open result 2",
                                                   wall_s=0.31),
                                          enabled=False)
        self.assertIn("choose", line)
        self.assertIn("open result 2", line)
        self.assertIn("2t·0.31s", line)

    def test_stats_column_aligns_across_values_and_depths(self):
        short = progress.build_decision_row(decision("go"), enabled=False)
        deep = progress.build_decision_row(
            decision("open result 2: pricing", out_tokens=3), enabled=False,
            depth=3)
        self.assertEqual(short.index("2t·"), deep.index("3t·"))

    def test_token_bar_grows_with_cost_and_saturates(self):
        self.assertEqual(progress._token_bar(0), "·")
        self.assertEqual(progress._token_bar(1), progress.TOKEN_BARS[0])
        self.assertEqual(progress._token_bar(3), progress.TOKEN_BARS[2])
        self.assertEqual(progress._token_bar(99), progress.TOKEN_BARS[-1])

    def test_pace_color_maps_speed_to_green_amber_pink(self):
        self.assertEqual(progress._pace_color(0.1), GREEN)
        self.assertEqual(progress._pace_color(0.8), AMBER)
        self.assertEqual(progress._pace_color(2.5), PINK)

    def test_malformed_numeric_fields_do_not_crash(self):
        event = decision("x", out_tokens="abc", wall_s=None)
        self.assertIn("0t·0.00s",
                      progress.build_decision_row(event, enabled=False))

    def test_pick_and_plan_show_their_indices(self):
        pick = {"node": "pick_many", "value": [3, 7], "valid": True}
        plan = {"node": "sequence", "value": [2, 3, 2], "valid": True}
        self.assertIn("3,7", progress.build_decision_row(pick, enabled=False))
        self.assertIn("2,3,2",
                      progress.build_decision_row(plan, enabled=False))

    def test_long_value_is_clipped(self):
        event = {"node": "short_text", "value": "a" * 80, "valid": True}
        line = progress.build_decision_row(event, enabled=False)
        self.assertIn("…", line)
        self.assertNotIn("a" * (progress.VALUE_CLIP + 1), line)

    def test_invalid_event_renders_dash(self):
        line = progress.build_decision_row({"node": "menu", "valid": False},
                                          enabled=False)
        self.assertIn("—", line)

    def test_resolved_picks_follow_under_the_row(self):
        event = {"node": "pick_many", "value": [2, 5], "valid": True,
                 "resolved": ["France has 68m people.", "Paris is big."]}
        rows = progress.build_decision_row(event, enabled=False,
                                          depth=2).splitlines()
        self.assertEqual(len(rows), 3)
        self.assertIn("· France has 68m people.", rows[1])
        self.assertTrue(rows[1].startswith("  │ │"))

    def test_no_escape_when_disabled(self):
        self.assertNotIn("\x1b", progress.build_decision_row(
            decision(), enabled=False, depth=2))


class TreeShapeTest(unittest.TestCase):
    """Branches nest by span; decisions nest under the action before them."""

    def test_rows_nest_under_the_open_branch(self):
        tree = make_tree()
        tree({"span": "open", "label": "round 1"})
        tree(decision())
        self.assertEqual(tree.rows[0], "  ◆ round 1")
        self.assertGreater(joint_column(tree.rows[1]), -1)

    def test_decisions_after_a_plain_menu_nest_under_it(self):
        tree = make_tree()
        tree({"span": "open", "label": "round 1"})
        tree(decision("open result 1"))
        tree({"node": "pick_many", "value": [1], "valid": True})
        tree(decision("answer the task now"))
        action, child, sibling = tree.rows[1:]
        self.assertGreater(joint_column(child), joint_column(action))
        self.assertEqual(joint_column(sibling), joint_column(action))

    def test_tagged_menu_does_not_open_an_action(self):
        tree = make_tree()
        tree({"span": "open", "label": "check"})
        tree(decision("bad", tag="judge"))
        tree(decision("good", tag="judge"))
        self.assertEqual(joint_column(tree.rows[1]),
                         joint_column(tree.rows[2]))

    def test_a_new_branch_ends_the_open_action(self):
        tree = make_tree()
        tree({"span": "open", "label": "web agent"})
        tree(decision("answer the task now"))
        tree({"span": "open", "label": "check the answer"})
        self.assertEqual(joint_column(tree.rows[2]),
                         joint_column(tree.rows[1]))

    def test_end_row_gives_result_and_counts_nested_calls(self):
        tree = make_tree()
        tree({"span": "open", "label": "web agent"})
        tree({"span": "open", "label": "round 1"})
        tree(decision())
        tree({"span": "close", "result": None})
        tree(decision(tag="judge"))
        tree({"span": "close", "result": "answered"})
        self.assertIn("╰ done · 1 call · 2t", tree.rows[3])
        self.assertTrue(tree.rows[-1].startswith("  ╰ answered · 2 calls"))
        self.assertEqual(len(tree.branches), 1)

    def test_close_without_open_is_ignored(self):
        tree = make_tree()
        tree({"span": "close", "result": "x"})
        self.assertEqual(tree.rows, [])

    def test_note_is_a_leaf_at_branch_depth(self):
        tree = make_tree()
        tree({"span": "open", "label": "planner agent"})
        tree(decision("open result 1"))
        tree.note("plan ›")
        self.assertIn("⌁ plan ›", tree.rows[2])
        self.assertEqual(joint_column(tree.rows[2]),
                         joint_column(tree.rows[1]))

    def test_fetch_failures_show_without_model_stats(self):
        tree = make_tree()
        tree({"node": "fetch", "value": "failed to open 'x'", "valid": True})
        tree(decision())
        self.assertIn("✖ fetch   failed to open 'x'", tree.rows[0])
        self.assertNotIn("0t·", tree.rows[0])
        self.assertEqual(tree.stats(1.0)["calls"], 1)

    def test_retries_and_tokens_are_tallied(self):
        tree = make_tree()
        tree(decision(attempt=1, out_tokens=4))
        stats = tree.stats(2.0)
        self.assertEqual((stats["calls"], stats["tokens"], stats["retries"]),
                         (1, 4, 1))


FAILED_EVENT = {"node": "menu", "valid": False, "out_tokens": 3,
                "wall_s": 0.2, "attempt": 0, "done_reason": "length",
                "raw_text": "Okay, the user wants"}
DUMP_EVENT = {"debug": True, "system_text": "be terse",
              "prompt_text": "Request: hi\nACTIONS:\n1 = a\n2 = b\n\nANSWER:",
              "result_text": " 1", "prompt_tokens": 5, "out_tokens": 1,
              "done_reason": "stop"}


class DebugRowsTest(unittest.TestCase):
    """/debug adds the raw answer and the prompt dump under the rows."""

    FAILED = FAILED_EVENT
    DUMP = DUMP_EVENT

    def test_debug_off_shows_only_the_decision_row(self):
        tree = make_tree()
        tree(self.FAILED)
        tree(self.DUMP)
        self.assertEqual(len(tree.rows), 1)
        self.assertNotIn("length", tree.rows[0])

    def test_debug_on_adds_the_raw_answer_under_the_row(self):
        tree = make_tree(debug=True)
        tree(self.FAILED)
        self.assertEqual(len(tree.rows), 2)
        self.assertIn("—", tree.rows[0])
        self.assertIn("length", tree.rows[1])
        self.assertIn("Okay, the user wants", tree.rows[1])

    def test_fetch_row_gets_no_raw_answer_line(self):
        tree = make_tree(debug=True)
        tree({"node": "fetch", "value": "failed to open 'x'", "valid": True})
        self.assertEqual(len(tree.rows), 1)

    def test_prompt_dump_waits_and_shows_under_its_own_decision(self):
        # Policy records the dump BEFORE the decision it belongs to; the
        # decision here is a new action, so it sits shallower than the
        # pick that came first. The dump must follow the action, not the pick.
        tree = make_tree(debug=True)
        tree({"span": "open", "label": "round 1"})
        tree(decision("open result 1"))
        tree({"node": "pick_many", "value": [1], "valid": True})
        tree(self.DUMP)
        tree(decision("answer the task now"))
        action_row = tree.rows.index(next(r for r in tree.rows
                                          if "answer the task now" in r))
        dump_row = tree.rows[action_row + 2]
        self.assertIn("5pt/1ot", dump_row)
        self.assertEqual(dump_row.index("·"),
                         tree.rows[action_row + 1].index("·"))

    def test_debug_full_streams_every_prompt_line_unclipped(self):
        tree = make_tree(debug=True, debug_full=True)
        tree({**self.DUMP, "prompt_text": "x" * 5000})
        tree(self.FAILED)
        joined = "\n".join(tree.rows)
        self.assertIn("· system:", joined)
        self.assertIn("x" * 5000, joined)
        self.assertEqual(tree.stats(1.0)["calls"], 1)


class DebugLineTest(unittest.TestCase):
    """The /debug line that makes an unparseable decision explain itself."""

    def _event(self, **overrides):
        event = {"node": "menu", "valid": False, "raw_text": "",
                 "done_reason": "length", "attempt": 0}
        return {**event, **overrides}

    def test_empty_completion_is_visible_as_quotes(self):
        # The user's failure shape: the cap was spent before any digit.
        self.assertIn("''", progress.build_debug_line(self._event()))

    def test_shows_done_reason_and_one_based_attempt(self):
        line = progress.build_debug_line(self._event(attempt=2))
        self.assertIn("length", line)
        self.assertIn("attempt 3", line)

    def test_raw_text_is_shown_and_stays_one_line(self):
        line = progress.build_debug_line(self._event(raw_text="Sure!\nLet me"))
        self.assertIn("Sure!", line)
        self.assertIn("\\n", line)  # repr keeps the newline visible
        self.assertEqual(len(line.splitlines()), 1)

    def test_long_raw_text_is_clipped_with_an_ellipsis(self):
        line = progress.build_debug_line(self._event(raw_text="z" * 400))
        self.assertIn("…", line)
        self.assertNotIn("z" * (progress.RAW_CLIP + 1), line)

    def test_empty_stop_completion_gets_a_hint(self):
        # raw '' + done: stop = the first generated token was already a
        # stop token (end-of-turn or the newline stop), not the token cap.
        line = progress.build_debug_line(
            self._event(raw_text="", done_reason="stop"))
        self.assertIn("stopped before answering", line)

    def test_empty_length_completion_gets_no_hint(self):
        # done: length already names its own cause (the token cap).
        self.assertNotIn("stopped before answering",
                         progress.build_debug_line(self._event()))

    def test_missing_keys_do_not_crash(self):
        line = progress.build_debug_line({})
        self.assertIn("''", line)
        self.assertIn("attempt 1", line)


class DebugTokensLineTest(unittest.TestCase):
    """The token-level dump line under every LLM call while /debug is on."""

    def _event(self, **overrides):
        event = {"debug": True,
                 "prompt_text": "Pick one\n1 = go\nReply with ONE digit.",
                 "result_text": " 1",
                 "prompt_tokens": 42, "out_tokens": 2,
                 "done_reason": "stop"}
        return {**event, **overrides}

    def test_shows_prompt_result_and_token_counts(self):
        line = progress.build_debug_tokens_line(self._event())
        self.assertIn("Pick one", line)
        self.assertIn("' 1'", line)
        self.assertIn("42pt/2ot", line)
        self.assertIn("stop", line)

    def test_clips_long_prompt_and_result(self):
        line = progress.build_debug_tokens_line(
            self._event(prompt_text="x" * 500, result_text="y" * 300))
        self.assertIn("…", line)
        self.assertNotIn("x" * (progress.DEBUG_PROMPT_HEAD
                                + progress.DEBUG_PROMPT_TAIL + 1), line)
        self.assertNotIn("y" * (progress.DEBUG_RESULT_CLIP + 1), line)

    def test_long_prompt_shows_the_tail_not_just_the_head(self):
        # The routing-debug failure shape: a head-only clip showed nothing
        # but the system block; the request and menu live at the tail.
        tail = ("Request: do a web search for forecast on the economy of "
                "japan\n\nWhich agent should handle this request?\n"
                "ACTIONS:\n2 = code — write or debug Python code\n\n"
                "ANSWER:")
        prompt = "<|im_start|>system\nbe terse\n" + "pad " * 100 + "\n" + tail
        line = progress.build_debug_tokens_line(
            self._event(prompt_text=prompt))
        self.assertIn("economy of japan", line)
        self.assertIn("ANSWER:", line)
        self.assertIn("<|im_start|>system", line)  # head still identifies it

    def test_short_prompt_stays_a_single_plain_repr(self):
        line = progress.build_debug_tokens_line(self._event())
        self.assertEqual(line.count("…"), 0)

    def test_missing_keys_do_not_crash(self):
        line = progress.build_debug_tokens_line({})
        self.assertIn("''", line)
        self.assertIn("0pt/0ot", line)


class DebugFullLinesTest(unittest.TestCase):
    """The /debug full dump: complete prompt and result, nothing clipped."""

    def _event(self, **overrides):
        event = {"debug": True, "system_text": "be terse",
                 "prompt_text": "Request: hi\nACTIONS:\n1 = a\n2 = b",
                 "result_text": " 1", "prompt_tokens": 9, "out_tokens": 2,
                 "done_reason": "stop"}
        return {**event, **overrides}

    def test_shows_system_prompt_and_result_with_counts(self):
        joined = "\n".join(progress.build_debug_full_lines(self._event()))
        self.assertIn("system:", joined)
        self.assertIn("be terse", joined)
        self.assertIn("2 = b", joined)
        self.assertIn("' 1'", joined)
        self.assertIn("9pt/2ot", joined)
        self.assertIn("stop", joined)

    def test_raw_template_prompt_needs_no_system_block(self):
        # Raw mode embeds the system block in the template already.
        joined = "\n".join(progress.build_debug_full_lines(
            self._event(system_text="")))
        self.assertNotIn("system:", joined)
        self.assertIn("Request: hi", joined)

    def test_missing_keys_do_not_crash(self):
        self.assertTrue(progress.build_debug_full_lines({}))


if __name__ == "__main__":
    unittest.main()
