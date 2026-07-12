"""JSONL decision tracing.

Every model consultation is recorded: it feeds the eval suite, debugging,
and (later) the prompt-template bandit. A Tracer built with path=None is
a no-op, so callers never branch.
"""
import json
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


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tracer = Tracer(Path(tmp) / "trace.jsonl")
        tracer.record({"step": 0, "valid": True})
        tracer.record({"step": 1, "valid": False})
        lines = (Path(tmp) / "trace.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2 and json.loads(lines[0])["step"] == 0
    Tracer(None).record({"ignored": True})
    print("smoke OK")
