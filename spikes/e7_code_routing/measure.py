"""Measurement harness for the e7 code-mode routing pipeline.

Runs every benchmark.jsonl item through pre_router.pre_classify(); items it
can't resolve for free fall through to router_menu.classify_with_menu(),
the one-token model decision. Reports pre-router coverage/precision, live
menu accuracy (overall + per true mode), a confusion table, end-to-end
accuracy, average model output tokens, and retry-ladder usage.

Usage (run from the repo root so ``threetoks`` is importable):
    PYTHONPATH="$PWD:$PWD/spikes/e7_code_routing" \\
        python3 spikes/e7_code_routing/measure.py [--backend ollama|scripted]

``--backend scripted`` exercises the exact same code path with a canned
offline backend (no Ollama, no network) — useful for iterating on the
harness itself; the real numbers in REPORT.md come from ``--backend ollama``
against the live qwen2.5:1.5b-instruct server.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pre_router import MODES, pre_classify
from router_menu import classify_with_menu

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.policy import Policy, PolicyConfig
from threetoks.trace import Tracer

SPIKE_DIR = Path(__file__).parent
BENCHMARK_PATH = SPIKE_DIR / "benchmark.jsonl"
FIXTURES_ROOT = SPIKE_DIR / "fixtures"
MODEL_NAME = "qwen2.5:1.5b-instruct"


class ScriptedBackend:
    """Cycles through canned completions; for offline plumbing checks only."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, model, raw_prompt, opts):
        text = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        return GenResult(text, 20, 2, 0.01, "stop")


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict]:
    """Read {request, label} records from the JSONL benchmark file."""
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


def route_all(backend, items: list[dict], roots: list[Path]) -> list[dict]:
    """Route every benchmark item; return one result record each.

    Free pre-router hits never touch ``policy``/``backend``. Everything
    else spends exactly one ``classify_with_menu`` call, whose retry-ladder
    attempts are all captured via the tracer for the token/retry stats.
    """
    events: list[dict] = []
    policy = Policy(backend, PolicyConfig(ModelSpec(MODEL_NAME, FAMILY_CHATML)),
                    Tracer(None, on_event=events.append))
    records = []
    for item in items:
        request, true_label = item["request"], item["label"]
        pre_mode, reason = pre_classify(request, roots)
        if pre_mode is not None:
            records.append({"request": request, "true": true_label,
                            "pred": pre_mode, "source": "pre_router",
                            "reason": reason, "attempts": 0, "out_tokens": 0})
            continue
        before = len(events)
        mode, decision = classify_with_menu(request, policy)
        attempt_events = events[before:]
        records.append({
            "request": request, "true": true_label, "pred": mode,
            "source": "menu", "attempts": len(attempt_events),
            "out_tokens": sum(event["out_tokens"] for event in attempt_events),
            "raw_text": decision.raw_text, "valid": decision.valid,
        })
    return records


def summarize(records: list[dict]) -> dict:
    """Aggregate per-item records into the numbers REPORT.md quotes."""
    total = len(records)
    pre = [r for r in records if r["source"] == "pre_router"]
    menu = [r for r in records if r["source"] == "menu"]
    pre_correct = sum(r["pred"] == r["true"] for r in pre)
    menu_correct = sum(r["pred"] == r["true"] for r in menu)

    per_mode: dict[str, dict] = {mode: {"n": 0, "correct": 0} for mode in MODES}
    for r in menu:
        per_mode[r["true"]]["n"] += 1
        per_mode[r["true"]]["correct"] += int(r["pred"] == r["true"])

    confusion = Counter((r["true"], r["pred"]) for r in records)
    retried = [r for r in menu if r["attempts"] > 1]

    return {
        "total": total,
        "pre_router": {
            "resolved": len(pre), "correct": pre_correct,
            "coverage": len(pre) / total if total else 0.0,
            "precision": pre_correct / len(pre) if pre else None,
        },
        "menu": {
            "n": len(menu), "correct": menu_correct,
            "accuracy": menu_correct / len(menu) if menu else None,
            "per_true_mode": {
                mode: {**d, "accuracy": d["correct"] / d["n"] if d["n"] else None}
                for mode, d in per_mode.items()},
            "avg_out_tokens": (sum(r["out_tokens"] for r in menu) / len(menu)
                              if menu else 0.0),
            "retried": len(retried),
            "retry_rate": len(retried) / len(menu) if menu else 0.0,
        },
        "confusion": {f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
        "overall_accuracy": ((pre_correct + menu_correct) / total
                             if total else 0.0),
    }


def print_report(summary: dict, records: list[dict]) -> None:
    """Human-readable console report; the harness's own smoke check reads
    this same summary dict, so the printed numbers and the asserted ones
    can never drift apart."""
    pre, menu = summary["pre_router"], summary["menu"]
    print(f"total items:            {summary['total']}")
    print(f"pre-router coverage:    {pre['resolved']}/{summary['total']} "
         f"({pre['coverage']:.1%})")
    print(f"pre-router precision:   {pre['correct']}/{pre['resolved']} "
         f"({pre['precision']:.1%})" if pre['resolved'] else
         "pre-router precision:   n/a (nothing resolved)")
    print(f"live menu accuracy:     {menu['correct']}/{menu['n']} "
         f"({menu['accuracy']:.1%})" if menu['n'] else
         "live menu accuracy:     n/a (nothing fell through)")
    for mode in MODES:
        d = menu["per_true_mode"][mode]
        acc = f"{d['accuracy']:.1%}" if d["accuracy"] is not None else "n/a"
        print(f"  {mode:>8}: {d['correct']}/{d['n']} ({acc})")
    print(f"overall accuracy:       {summary['overall_accuracy']:.1%}")
    print(f"avg out tokens (menu):  {menu['avg_out_tokens']:.1f}")
    print(f"retry-ladder usage:     {menu['retried']}/{menu['n']} "
         f"({menu['retry_rate']:.1%})" if menu['n'] else
         "retry-ladder usage:     n/a")
    print("confusion (true->pred): " + json.dumps(summary["confusion"]))
    wrong = [r for r in records if r["pred"] != r["true"]]
    print(f"\n{len(wrong)} misclassified item(s):")
    for r in wrong:
        print(f"  [{r['source']}] true={r['true']} pred={r['pred']!r} "
             f"raw={r.get('raw_text', r.get('reason'))!r} :: {r['request']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ollama", "scripted"),
                        default="ollama")
    parser.add_argument("--out", type=Path, default=SPIKE_DIR / "results.json")
    args = parser.parse_args()

    if args.backend == "ollama":
        from threetoks.backend.ollama import OllamaBackend
        backend = OllamaBackend()
    else:
        backend = ScriptedBackend([" 1", " 2", " 3", " 4"])

    items = load_benchmark()
    records = route_all(backend, items, roots=[FIXTURES_ROOT])
    summary = summarize(records)
    print_report(summary, records)
    args.out.write_text(json.dumps({"summary": summary, "records": records},
                                   indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
