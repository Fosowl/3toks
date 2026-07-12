"""Per-spec funnel: search -> fetch -> name-match -> gate -> anchor-judge.

Logs every stage (task instructions §c: "Log the full funnel per task:
search results -> files fetched -> name-matched defs -> gate survivors ->
anchor passes"). The model never appears anywhere in this file — the
subprocess in judge.py is the only judge, matching the spike's whole
point (retrieval-as-repair does not need a policy call at all).
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract  # noqa: E402
import fetchers  # noqa: E402
import judge  # noqa: E402

MAX_URLS_PER_SPEC = 6      # bound wall-clock: don't fetch every search hit
MAX_CANDIDATES_TRIED = 8   # bound wall-clock: stop judging after this many


@dataclass
class Attempt:
    """One candidate's full trip through the gates, for the funnel log."""

    source_url: str
    original_name: str
    tier: str
    gate_result: str        # "ok" or a rejection reason
    anchor_result: str = "not_run"   # "pass" / "fail:<err>" / "not_run"
    judge_seconds: float = 0.0


@dataclass
class TaskResult:
    spec_name: str
    classic: bool
    hit: bool
    wall_clock_s: float
    search_results_count: int
    urls_fetched_count: int
    files_fetched_count: int
    name_matched_defs_count: int
    gate_survivors_count: int
    attempts: list = field(default_factory=list)
    accepted_url: str | None = None
    accepted_snippet: str | None = None
    search_hop_used: str = ""


def _dedupe(seq):
    seen, out = set(), []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def run_task(spec, provider, github_cache: dict, verbose: bool = True
             ) -> TaskResult:
    """Run one spec end to end, first anchor-passing candidate wins."""
    t0 = time.perf_counter()
    all_urls = []
    search_results_count = 0
    for q in spec.queries:
        try:
            hits = provider.search(q, max_results=8)
        except Exception:
            hits = []
        search_results_count += len(hits)
        all_urls.extend(h.url for h in hits)
        if verbose:
            print(f"    query {q!r} -> {len(hits)} hits")

    urls = _dedupe(all_urls)[:MAX_URLS_PER_SPEC]

    files_fetched = []          # (source_url, file_text)
    for url in urls:
        fetched = fetchers.classify_and_fetch(url, github_cache)
        files_fetched.extend(fetched)
        if verbose:
            print(f"    fetch {url} -> {len(fetched)} file(s)")

    name_matched = []           # extract.Candidate, across all files
    for source_url, text in files_fetched:
        cands, parsed_ok = extract.find_matches(text, spec, source_url)
        if verbose and not parsed_ok:
            print(f"    {source_url}: did not parse standalone")
        name_matched.extend(cands)

    result = TaskResult(
        spec_name=spec.name, classic=spec.classic, hit=False,
        wall_clock_s=0.0, search_results_count=search_results_count,
        urls_fetched_count=len(urls), files_fetched_count=len(files_fetched),
        name_matched_defs_count=len(name_matched), gate_survivors_count=0,
    )

    gate_survivors = 0
    for cand in name_matched[:MAX_CANDIDATES_TRIED]:
        file_tree = None
        try:
            import ast
            file_tree = ast.parse(
                "\n".join(cand.file_lines))
        except (SyntaxError, ValueError):
            attempt = Attempt(cand.source_url, cand.original_name, cand.tier,
                              "file_does_not_reparse")
            result.attempts.append(attempt)
            continue
        extract.assemble(cand, spec.name, file_tree)
        if cand.rejected:
            result.attempts.append(Attempt(
                cand.source_url, cand.original_name, cand.tier,
                cand.rejected))
            continue
        gate_survivors += 1
        ok, err, elapsed = judge.judge(cand.snippet, spec.anchor)
        attempt = Attempt(
            cand.source_url, cand.original_name, cand.tier, "ok",
            anchor_result=("pass" if ok else f"fail:{err}"),
            judge_seconds=elapsed)
        result.attempts.append(attempt)
        if verbose:
            print(f"    candidate {cand.original_name} ({cand.tier}) from "
                  f"{cand.source_url} -> anchor {attempt.anchor_result}")
        if ok:
            result.hit = True
            result.accepted_url = cand.source_url
            result.accepted_snippet = cand.snippet
            break

    result.gate_survivors_count = gate_survivors
    result.wall_clock_s = time.perf_counter() - t0
    return result


if __name__ == "__main__":
    # Offline-only smoke: a fake "provider" so this never touches the
    # network (the real thing is exercised by run_benchmark.py).
    class _FakeHit:
        def __init__(self, url):
            self.url = url

    class _FakeProvider:
        def search(self, query, max_results=8):
            return [_FakeHit("fixture://roman.py")]

    import types
    real_classify = fetchers.classify_and_fetch
    fixture_src = (
        "def roman_to_integer(s):\n"
        "    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, "
        "'M': 1000}\n"
        "    total = 0\n"
        "    for i, ch in enumerate(s):\n"
        "        v = vals[ch]\n"
        "        if i + 1 < len(s) and vals[s[i + 1]] > v:\n"
        "            total -= v\n"
        "        else:\n"
        "            total += v\n"
        "    return total\n"
    )
    fetchers.classify_and_fetch = lambda url, cache: (
        [("fixture://roman.py", fixture_src)] if "roman" in url else [])
    try:
        from specs import Spec
        spec = Spec(name="roman_to_int", aliases=("romanToInt",),
                    anchor='assert roman_to_int("MCMXCIV") == 1994',
                    queries=("q1",), classic=True)
        res = run_task(spec, _FakeProvider(), {}, verbose=False)
        assert res.hit, res
        assert res.name_matched_defs_count == 1
        assert res.gate_survivors_count == 1
    finally:
        fetchers.classify_and_fetch = real_classify
    print("smoke OK")
