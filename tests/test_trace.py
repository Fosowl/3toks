"""Offline tests for trace spans: the open/close events the TUI tree draws."""
import unittest

from threetoks.trace import Tracer, span


def recording_tracer():
    """A file-less tracer and the list that receives its events."""
    events: list[dict] = []
    return Tracer(None, on_event=events.append), events


class SpanTest(unittest.TestCase):
    def test_none_tracer_records_nothing_and_still_yields_an_outcome(self):
        with span(None, "part") as outcome:
            outcome["result"] = "ok"
        self.assertEqual(outcome, {"result": "ok"})

    def test_nested_spans_close_inner_first_even_on_an_error(self):
        tracer, events = recording_tracer()
        with self.assertRaises(KeyboardInterrupt), span(tracer, "outer"), \
                span(tracer, "inner"):
            raise KeyboardInterrupt
        edges = [(e["span"], e["label"], e.get("result")) for e in events]
        self.assertEqual(edges, [("open", "outer", None),
                                 ("open", "inner", None),
                                 ("close", "inner", "stopped"),
                                 ("close", "outer", "stopped")])

    def test_a_result_set_before_the_error_is_kept(self):
        tracer, events = recording_tracer()
        with self.assertRaises(RuntimeError), span(tracer, "part") as outcome:
            outcome["result"] = "web"
            raise RuntimeError("boom")
        self.assertEqual(events[-1]["result"], "web")


if __name__ == "__main__":
    unittest.main()
