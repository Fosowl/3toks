"""Run scenarios against the real settled policy model (qwen2.5:1.5b-instruct
via Ollama) -- every menu pick and every span regeneration is a real model
decision, no scripting.

Requires Ollama running at http://localhost:11434 with qwen2.5:1.5b-instruct
pulled. Does not start/stop/manage the server.

    python3 spikes/e8_edit_vertical/run_live.py             # all 5 scenarios
    python3 spikes/e8_edit_vertical/run_live.py 1 2 3        # by index (1-based)
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_DIR.parent.parent
for _p in (str(SPIKE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure  # noqa: E402
import scenarios as scen  # noqa: E402
import vertical as vert  # noqa: E402

from threetoks.backend.base import FAMILY_CHATML, ModelSpec  # noqa: E402
from threetoks.backend.ollama import OllamaBackend  # noqa: E402
from threetoks.engine import run_episode  # noqa: E402
from threetoks.policy import Policy, PolicyConfig  # noqa: E402
from threetoks.trace import Tracer  # noqa: E402

MODEL_NAME = "qwen2.5:1.5b-instruct"
# E1's winning menu-selector system prompt (threetoks/cli.py SYSTEM_CHATML),
# reused verbatim -- this spike does not invent its own menu-steering prompt.
SYSTEM_CHATML = (
    "You are a menu selector controlling a research agent. When shown a "
    "numbered list of actions, reply with exactly one digit: the number of "
    "the best action. When asked for sentence numbers, reply with only the "
    "fewest numbers separated by commas. Otherwise reply with the shortest "
    "correct answer. Do not explain.")


def run_one_live(scenario: scen.Scenario, work_root: Path) -> dict:
    root = work_root / scenario.key
    if root.exists():
        shutil.rmtree(root)
    scen.materialize(scenario, root)

    events: list[dict] = []
    backend = OllamaBackend()
    config = PolicyConfig(ModelSpec(MODEL_NAME, FAMILY_CHATML), system=SYSTEM_CHATML)
    policy = Policy(backend, config, Tracer(None, on_event=events.append))
    edit_vertical = vert.EditVertical(scenario.request, root, scenario.test_code)
    outcome = run_episode(edit_vertical, policy, max_steps=30)

    summary = measure.summarize_events(events)
    file_text = "\n\n".join(scenario.files.values())
    whole_tokens, method = measure.whole_file_tokens(file_text)
    econ = measure.delta_economics(summary["output_tokens"], whole_tokens)

    return {"scenario": scenario.key, "title": scenario.title,
           "outcome": outcome, "measurement": summary,
           "delta_economics": {**econ, "token_method": method},
           "trace": events}


def main() -> None:
    indices = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else \
        list(range(1, len(scen.ALL_SCENARIOS) + 1))
    chosen = [scen.ALL_SCENARIOS[i - 1] for i in indices]

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work_root = Path(tmp)
        for scenario in chosen:
            print(f"--- running live: {scenario.key} ---", file=sys.stderr)
            result = run_one_live(scenario, work_root)
            results.append(result)
            o = result["outcome"]
            print(f"    success={o['success']} reason={o['reason']!r}",
                 file=sys.stderr)

    trace_path = SPIKE_DIR / "live_trace.jsonl"
    with trace_path.open("w") as fh:
        for r in results:
            for event in r["trace"]:
                fh.write(json.dumps({"scenario": r["scenario"], **event}) + "\n")

    for r in results:
        r.pop("trace")
    print(json.dumps(results, indent=2))
    print()
    ok = sum(1 for r in results if r["outcome"]["success"])
    print(f"live: {ok}/{len(results)} scenarios succeeded")
    for r in results:
        o, m, e = r["outcome"], r["measurement"], r["delta_economics"]
        print(f"  {r['scenario']:24s} success={o['success']!s:5} "
             f"path={o['path_taken']:12s} op={o['operation']:12s} "
             f"decisions={m['decision_points']} retries={m['retries']} "
             f"out_tok={m['output_tokens']:4d} ratio={e['ratio']:.3f} "
             f"reason={o['reason']}")


if __name__ == "__main__":
    main()
