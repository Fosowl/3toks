"""Drive the 12-spec E9 benchmark against live search + the real internet.

Network required; no model calls (Ollama is untouched — the subprocess in
judge.py is the only judge). Writes:
  - results.json           raw per-task funnel + attempt log
  - accepted_snippets/*.py  verbatim accepted candidate + provenance header

Run from the repo root: `python3 spikes/e9_code_retrieval/run_benchmark.py`
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetchers  # noqa: E402
import pipeline  # noqa: E402
from specs import ALL_SPECS  # noqa: E402

RESULTS_PATH = HERE / "results.json"
SNIPPETS_DIR = HERE / "accepted_snippets"


def main():
    provider, hop_note = fetchers.build_provider_chain()
    print(f"[providers] {hop_note}")
    github_cache = {}
    results = []
    t_start = time.perf_counter()

    for i, spec in enumerate(ALL_SPECS, 1):
        kind = "classic" if spec.classic else "NON-classic (expect miss)"
        print(f"\n=== [{i}/{len(ALL_SPECS)}] {spec.name} ({kind}) ===")
        result = pipeline.run_task(spec, provider, github_cache, verbose=True)
        status = "HIT" if result.hit else "miss"
        print(f"  -> {status} in {result.wall_clock_s:.1f}s "
              f"(searched {result.search_results_count}, "
              f"fetched {result.files_fetched_count} files, "
              f"matched {result.name_matched_defs_count} defs, "
              f"gated-through {result.gate_survivors_count})")
        results.append(result)

    total_s = time.perf_counter() - t_start
    _write_results(results, hop_note, total_s)
    _write_snippets(results)
    _print_summary(results, total_s)


def _write_results(results, hop_note, total_s):
    payload = {
        "provider_hop_note": hop_note,
        "total_wall_clock_s": total_s,
        "tasks": [dataclasses.asdict(r) for r in results],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[written] {RESULTS_PATH}")


def _write_snippets(results):
    SNIPPETS_DIR.mkdir(exist_ok=True)
    for r in results:
        if not r.hit:
            continue
        header = (f'"""Accepted candidate for {r.spec_name}\n'
                  f"Source URL: {r.accepted_url}\n"
                  f"License detection: NOT PERFORMED (future work — see "
                  f'REPORT.md safety section)\n"""\n')
        (SNIPPETS_DIR / f"{r.spec_name}.py").write_text(
            header + r.accepted_snippet)
    print(f"[written] {SNIPPETS_DIR}/*.py "
          f"({sum(1 for r in results if r.hit)} accepted)")


def _print_summary(results, total_s):
    classic = [r for r in results if r.classic]
    nonclassic = [r for r in results if not r.classic]
    classic_hits = [r for r in classic if r.hit]
    nonclassic_hits = [r for r in nonclassic if r.hit]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"classic hit rate:      {len(classic_hits)}/{len(classic)}")
    print(f"non-classic hit rate:  {len(nonclassic_hits)}/{len(nonclassic)} "
          f"(expected near-0 — a hit here is a FALSE ACCEPT to investigate)")
    if classic_hits:
        avg_hit = sum(r.wall_clock_s for r in classic_hits) / len(classic_hits)
        print(f"avg seconds per classic HIT:  {avg_hit:.1f}s")
    classic_misses = [r for r in classic if not r.hit]
    if classic_misses:
        avg_miss = sum(r.wall_clock_s for r in classic_misses) \
            / len(classic_misses)
        print(f"avg seconds per classic MISS: {avg_miss:.1f}s")
    if nonclassic:
        avg_nc = sum(r.wall_clock_s for r in nonclassic) / len(nonclassic)
        print(f"avg seconds per non-classic task: {avg_nc:.1f}s")
    print(f"total wall clock: {total_s:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
