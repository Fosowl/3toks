"""JSONL decision tracing.

Every model consultation is recorded: it feeds the eval suite, debugging,
and (later) the prompt-template bandit. A Tracer built with path=None is
a no-op, so callers never branch. :func:`span` adds open/close events
around a named part of a task, so the TUI can draw a progress tree.
"""
import json
from contextlib import contextmanager
from pathlib import Path


class Tracer:
    """Append-only JSONL event writer."""

    def __init__(self, path: Path | str | None, on_event=None):
        self.path = Path(path) if path else None
        self.on_event = on_event
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict) -> None:
        """Append one event; write only when a path is configured."""
        if self.on_event:
            self.on_event(event)
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


@contextmanager
def span(tracer, label: str):
    """Record one named part of a task as an open and a close event.

    The TUI, the research loop and the planner use this so the TUI can
    show the task as a tree. The caller gives the policy tracer and a
    short label. The caller can set ``outcome["result"]`` in the block;
    the close event carries it. An exception sets the result "stopped"
    when the block did not set one yet, and goes on to the caller.

    Args:
        tracer (Tracer | None): the policy tracer; None records nothing.
        label (str): what this part does, e.g. "round 1 · search and read".

    Yields:
        dict: the outcome; its "result" (str) goes into the close event.
    """
    outcome: dict = {}
    if tracer is None:
        yield outcome
        return
    tracer.record({"span": "open", "label": label})
    try:
        yield outcome
    except BaseException:
        outcome.setdefault("result", "stopped")
        raise
    finally:
        tracer.record({"span": "close", "label": label,
                       "result": outcome.get("result")})


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tracer = Tracer(Path(tmp) / "trace.jsonl")
        tracer.record({"step": 0, "valid": True})
        tracer.record({"step": 1, "valid": False})
        lines = (Path(tmp) / "trace.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2 and json.loads(lines[0])["step"] == 0
    Tracer(None).record({"ignored": True})
    seen: list[dict] = []
    with span(Tracer(None, on_event=seen.append), "part") as outcome:
        outcome["result"] = "ok"
    assert [e["span"] for e in seen] == ["open", "close"], seen
    assert seen[1]["result"] == "ok", seen
    print("smoke OK")
