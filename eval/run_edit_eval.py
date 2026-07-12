"""Live edit-vertical eval: 12 planted-bug scenarios against the real model.

Usage (needs Ollama running with the settled policy model pulled):

    python3 eval/run_edit_eval.py [--runs N] [--out results.json] [keys...]

Per scenario it reports success/verified, decision count, retries, total
output tokens across every attempt (honest accounting: discarded retries
included), and the delta-economics ratio against the edited file's own
token count (measured with the model's real tokenizer via one throwaway
prompt-eval call).
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edit_scenarios import ALL_SCENARIOS, materialize  # noqa: E402

from threetoks.backend.base import FAMILY_CHATML, GenOpts, ModelSpec  # noqa: E402
from threetoks.backend.ollama import OllamaBackend  # noqa: E402
from threetoks.code.edit.vertical import EditVertical  # noqa: E402
from threetoks.engine import run_episode  # noqa: E402
from threetoks.policy import Policy, PolicyConfig  # noqa: E402
from threetoks.trace import Tracer  # noqa: E402

MODEL_NAME = "qwen2.5:1.5b-instruct"
MAX_STEPS = 40


def whole_file_tokens(backend: OllamaBackend, text: str) -> int:
    """The file's token count per the policy model's own tokenizer."""
    result = backend.complete(MODEL_NAME, text, GenOpts(max_tokens=1))
    return result.prompt_tokens or max(1, len(text) // 4)


def run_scenario(scenario, backend: OllamaBackend) -> dict:
    """One live episode; returns the outcome plus measurements."""
    events: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        materialize(scenario, root)
        vertical = EditVertical(scenario.request, root, scenario.test_code)
        policy = Policy(backend, PolicyConfig(ModelSpec(MODEL_NAME,
                                                        FAMILY_CHATML)),
                        Tracer(None, on_event=events.append))
        started = time.perf_counter()
        outcome = run_episode(vertical, policy, max_steps=MAX_STEPS)
        elapsed = time.perf_counter() - started

        edited = outcome["target_file"] or max(
            scenario.files, key=lambda rel: len(scenario.files[rel]))
        file_tokens = whole_file_tokens(backend, (root / edited).read_text())

    out_tokens = sum(int(e.get("out_tokens") or 0) for e in events)
    decisions = sum(1 for e in events if e.get("attempt") == 0)
    return {
        "key": scenario.key,
        "expected_operation": scenario.expected_operation,
        **outcome,
        "decisions": decisions,
        "retries": len(events) - decisions,
        "output_tokens": out_tokens,
        "whole_file_tokens": file_tokens,
        "ratio": round(out_tokens / file_tokens, 3) if file_tokens else 0.0,
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*",
                        help="scenario keys or prefixes to run (default all)")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", default=None)
    options = parser.parse_args()

    picked = [s for s in ALL_SCENARIOS if not options.keys
              or any(s.key.startswith(k) for k in options.keys)]
    backend = OllamaBackend()
    rows = []
    for run in range(options.runs):
        for scenario in picked:
            row = run_scenario(scenario, backend)
            row["run"] = run
            rows.append(row)
            print(f"{row['key']:32s} success={row['success']!s:5s} "
                  f"op={row['operation'] or '-':12s} "
                  f"repairs={row['repairs']} decisions={row['decisions']} "
                  f"retries={row['retries']} out_tok={row['output_tokens']:4d} "
                  f"ratio={row['ratio']:.2f} {row['seconds']}s")

    wins = sum(1 for r in rows if r["success"])
    right_op = sum(1 for r in rows
                   if r["operation"] == r["expected_operation"])
    print(f"\ntotal: {wins}/{len(rows)} succeeded, "
          f"{right_op}/{len(rows)} picked the expected operation")
    win_ratios = [r["ratio"] for r in rows if r["success"]]
    if win_ratios:
        print(f"mean ratio on successes: "
              f"{sum(win_ratios) / len(win_ratios):.2f} "
              f"(1.0 = whole-file rewrite cost)")
    if options.out:
        Path(options.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {options.out}")
    return 0 if wins == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
