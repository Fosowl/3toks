"""Run all 5 scenarios with a scripted policy -- proves the state machine.

The scripted backend never sees permutation math: MenuNode always renders
its options as literal "N = text" lines, so a menu decision is answered by
grepping the CURRENT prompt for a scenario-specific keyword and returning
the digit in front of it. A generation decision is answered with the
scenario's pre-written correct completion. Every scenario should therefore
reach success in this offline run -- what's being proven here is that the
state machine (locate/navigate/disambiguate -> operation -> generate/delete
-> oracle) wires together correctly, not that a real model can pick right.

    python3 spikes/e8_edit_vertical/run_offline.py
"""
import json
import re
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

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec  # noqa: E402
from threetoks.engine import run_episode  # noqa: E402
from threetoks.policy import Policy, PolicyConfig  # noqa: E402
from threetoks.trace import Tracer  # noqa: E402

ACTION_LINE = re.compile(r"^(\d+) = (.*)$")


class KeywordScriptedBackend:
    """Answers menus by keyword search, generations with a canned body.

    This is the offline analogue of tests/test_agents_core.py's
    ScriptedBackend, generalized to "find the right numbered option" so it
    does not need to replicate Policy's per-attempt permutation math.
    """

    def __init__(self, menu_keywords: list[str], completion: str | None):
        self.menu_keywords = menu_keywords
        self.completion = completion
        self.calls = 0

    def complete(self, model, raw_prompt, opts):
        self.calls += 1
        if "ACTIONS:" in raw_prompt:
            return self._answer_menu(raw_prompt)
        text = self.completion if self.completion is not None else "pass"
        return GenResult(text, len(raw_prompt) // 4, len(text.split()), 0.0, "stop")

    def _answer_menu(self, raw_prompt: str) -> GenResult:
        lines = raw_prompt.splitlines()
        for keyword in self.menu_keywords:
            for line in lines:
                match = ACTION_LINE.match(line.strip())
                if match and keyword in match.group(2):
                    return GenResult(f" {match.group(1)}", len(raw_prompt) // 4,
                                     1, 0.0, "stop")
        # defensive fallback: first option (should not be hit if keywords
        # are set up correctly for the scenario)
        for line in lines:
            match = ACTION_LINE.match(line.strip())
            if match:
                return GenResult(f" {match.group(1)}", len(raw_prompt) // 4,
                                 1, 0.0, "stop")
        return GenResult(" 1", len(raw_prompt) // 4, 1, 0.0, "stop")


def menu_keywords_for(scenario: scen.Scenario) -> list[str]:
    keywords = list(scenario.nav_keywords)
    if scenario.disambiguate_keyword:
        keywords.append(scenario.disambiguate_keyword)
    keywords.append(vert.OPERATION_LABELS[scenario.expected_operation])
    if scenario.delete_keyword:
        keywords.append(scenario.delete_keyword)
    return keywords


def run_one(scenario: scen.Scenario, work_root: Path) -> dict:
    root = work_root / scenario.key
    if root.exists():
        shutil.rmtree(root)
    scen.materialize(scenario, root)

    events: list[dict] = []
    backend = KeywordScriptedBackend(menu_keywords_for(scenario),
                                     scenario.good_completion)
    policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)),
                    Tracer(None, on_event=events.append))
    edit_vertical = vert.EditVertical(scenario.request, root, scenario.test_code)
    outcome = run_episode(edit_vertical, policy, max_steps=30)

    summary = measure.summarize_events(events)
    file_text = "\n\n".join(scenario.files.values())
    whole_tokens, method = measure.whole_file_tokens(file_text)
    econ = measure.delta_economics(summary["output_tokens"], whole_tokens)

    return {"scenario": scenario.key, "title": scenario.title,
           "outcome": outcome, "measurement": summary,
           "delta_economics": {**econ, "token_method": method}}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work_root = Path(tmp)
        results = [run_one(scenario, work_root) for scenario in scen.ALL_SCENARIOS]

    print(json.dumps(results, indent=2))
    print()
    ok = sum(1 for r in results if r["outcome"]["success"])
    print(f"offline: {ok}/{len(results)} scenarios succeeded")
    for r in results:
        o, m, e = r["outcome"], r["measurement"], r["delta_economics"]
        print(f"  {r['scenario']:24s} success={o['success']!s:5} "
             f"path={o['path_taken']:12s} op={o['operation']:12s} "
             f"decisions={m['decision_points']} out_tok={m['output_tokens']:4d} "
             f"ratio={e['ratio']:.3f} ({e['token_method']})")


if __name__ == "__main__":
    main()
