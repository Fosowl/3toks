# E7 — code-mode routing

Spike question: before a coding request enters a code vertical, can we
cheaply decide which of 4 modes it needs — **navigate** (answer a question
about existing code), **edit** (modify existing code), **compute** (the
user wants an answer code produces), **author** (the user wants a code
artifact delivered) — using (a) free deterministic pre-checks for the
high-precision cases, and (b) a one-token `MenuNode` model decision for
everything else?

This report states what was measured. It does not conclude whether the
approach is worth shipping — that call belongs upstream.

## What was built

- `fixtures/` — a tiny fake project (`parser.py` with a real off-by-one
  bug, `pkg/utils.py` with a `slugify` function and a `Cache` class,
  `data/access.log`, `data/sales.csv`, `notes.md`) so the pre-router's
  "does this path actually exist" checks have something real to hit.
- `pre_router.py` — the free pre-router. Zero model calls, ever. Two
  signal families, both required to be high-precision (fall through on
  any doubt):
  1. **path-mention**: a token in the request names a file that really
     exists under `fixtures/` (`os.path`/`Path.rglob`, no guessing),
     combined with a verb cue → `edit` / `navigate` / `compute`. A path
     named with no matching verb cue is left ambiguous on purpose.
  2. **phrasing-only**: an explicit "write/create/build a script/
     function/tool" is `author` even if a file is mentioned only as an
     illustrative example (`"write a script that parses log files like
     access.log"` must not get steered into `compute` — this is checked
     as one of the offline tests). An unambiguous "where is X defined" /
     "what does X do" interrogative with a code-entity noun (function,
     method, class, defined, ...) is `navigate`. A compute verb (parse,
     sum, count, calculate, ...) plus a data noun (csv, log, column, ...)
     is `compute`.
- `router_menu.py` — the one-token fallback for anything the pre-router
  won't touch: a real `threetoks.nodes.MenuNode` over the four mode
  descriptions, run through the real `threetoks.policy.Policy` +
  `threetoks.render.Episode`, with a `CODE_ROUTER_PREFIX` few-shot block
  in the style of `threetoks/agents/router.py`'s `ROUTER_PREFIX` (digit
  position shuffled across examples so the model can't just memorize
  "the answer is always 3"). No escape option — the four modes are
  treated as exhaustive, same choice `route()` makes for the top-level
  agent menu.
- `make_benchmark.py` → `benchmark.jsonl` — 69 hand-written `{request,
  label}` pairs, labels assigned by my own judgment of user intent
  *before* tuning the pre-router against them (see "how the benchmark was
  built" below), spread over the 4 modes (19 navigate / 17 edit / 16
  compute / 17 author), including several that mention the fixture paths
  and several deliberately ambiguous ones (marked in comments in
  `make_benchmark.py`).
- `measure.py` — runs every benchmark item through `pre_classify()` then
  `classify_with_menu()` for whatever falls through, and reports
  coverage/precision/accuracy/tokens/retries. `--backend scripted` runs
  the identical code path against a canned offline backend (no Ollama);
  `--backend ollama` (the default) hits the real
  `threetoks.backend.ollama.OllamaBackend`.
- `test_offline.py` — 14 unittest cases, all offline, no network: the
  pre-router's path/phrasing branches, the menu router against a scripted
  backend (including a garbage-reply fallback), and `measure.py`'s
  aggregation math against a hand-checked mini confusion table.

### How to run it

From the repo root (needs `threetoks` importable, hence the `PYTHONPATH`):

```bash
# offline, no Ollama, no network
cd spikes/e7_code_routing
PYTHONPATH="<repo_root>:." python3 -m unittest test_offline -v

# live measurement (needs Ollama + qwen3.5:2b pulled)
cd <repo_root>
PYTHONPATH="$PWD:$PWD/spikes/e7_code_routing" \
    python3 spikes/e7_code_routing/measure.py --backend ollama \
    --out spikes/e7_code_routing/results.json
```

Every module also ends with its own `if __name__ == "__main__":` smoke
check (`pre_router.py`, `router_menu.py`, `make_benchmark.py`, and the two
fixture modules that have real bugs/logic in them).

### How the benchmark was built (so the numbers below can be trusted)

`benchmark.jsonl` was written first, labels chosen by reading each request
as a person would, independent of what any regex would do with it. Only
*after* the full 69-item set was fixed did I run the pre-router against it
and look at the misses. One genuine bug turned up this way (`pre_router.py`
was matching the bare word "top" as a compute cue, which false-positived on
"the top IP **computed** in parser.py" — a navigate question that happens
to contain the word "top"); I tightened that to require `"top <number>"`
(a real top-K signal) since a bare common English word is too weak a cue
in general, not because it fixed one benchmark line. One deliberately
ambiguous item was *not* patched around and is reported as a real
pre-router miss below (see "Concrete failures").

## Measured numbers

Live run: `qwen3.5:2b` via Ollama at `localhost:11434`,
`threetoks.backend.base.FAMILY_CHATML`, default `PolicyConfig` (temperature
ladder 0.0/0.0/0.4, `MenuNode` cap of 3 output tokens). 69 benchmark items,
about 9 seconds wall time total for the whole run (all 21 model calls).

### Pre-router (deterministic, zero model calls)

| | value |
|---|---|
| coverage | 48 / 69 = **69.6%** |
| precision (of the 48 resolved) | 47 / 48 = **97.9%** |

The one wrong pre-router call is the deliberately ambiguous item described
above — see "Concrete failures".

### Live one-token menu accuracy (the 21 items the pre-router wouldn't touch)

| true mode | n | correct | accuracy |
|---|---|---|---|
| navigate | 8 | 5 | 62.5% |
| edit | 7 | 6 | 85.7% |
| compute | 3 | 2 | 66.7% |
| author | 3 | 3 | 100.0% |
| **overall** | **21** | **16** | **76.2%** |

### End-to-end (pre-router + menu combined, all 69 items)

**63 / 69 = 91.3%** overall routing accuracy.

### Confusion table (true row → predicted column, all 69 items, both sources combined)

| true \ pred | navigate | edit | compute | author | n |
|---|---|---|---|---|---|
| navigate | 16 | 1 | 2 | 0 | 19 |
| edit | 0 | 15 | 2 | 0 | 17 |
| compute | 0 | 0 | 15 | 1 | 16 |
| author | 0 | 0 | 0 | 17 | 17 |

Every error in the whole run is a mode being mistaken for **compute** or,
once, **author** — nothing is ever wrongly routed *into* navigate or edit
from a different true mode. `compute` and `author` are "attractor" modes
here: once the model or the pre-router is unsure, it leans toward "this
must want a computed fact" or "this must want a script."

### Cost per model-routed request

- average model **output tokens**: **3.0** per routed request (the
  `MenuNode` cap is 3; the model used the full budget every time, mostly
  a leading space + one digit).
- **retry-ladder usage: 0 / 21 (0%)** — every first attempt at temperature
  0.0 parsed as a valid digit, so the reshuffle-retry (attempt 2) and the
  temperature-bump retry (attempt 3) never fired in this run. This says
  the *format* was never the problem; every wrong answer here is a
  confidently wrong first-attempt pick, not a garbled one — the ladder
  measured in E4 for other decisions targets format failures, and this
  decision didn't have any.

## Concrete failure examples

1. **True: navigate, predicted: compute** (model picked option 1 of 4)
   > "Can you check why parser.py gives the wrong method field?"

   Reads as a diagnostic question about existing code (navigate), but the
   model treated "check why ... gives the wrong X" as wanting a computed
   fact.

2. **True: navigate, predicted: compute** (model picked option 1 of 4)
   > "How is the top IP computed in parser.py?"

   Asks about the *logic* ("how is X computed"), not a fresh computation —
   the word "computed" itself seems to have pulled the model toward the
   compute option.

3. **True: navigate, predicted: edit** (model picked option 3 of 4)
   > "Walk me through what happens when Cache.put is called twice past
   > capacity."

   A pure explain-the-code request; the model treated "past capacity" /
   "Cache.put" as something that needs changing rather than explaining.

4. **True: edit, predicted: compute** (model picked option 1 of 4)
   > "Debug why top_ip returns the wrong IP sometimes."

   "Debug" here means "find and fix the bug" (edit), but the model read
   the IP-counting subject matter as a computation request.

5. **True: compute, predicted: author** (model picked option 4 of 4)
   > "Group the sales by region and tell me which region made the most
   > revenue."

   No script is requested and no file is mentioned by name — but the
   imperative "Group the sales by region" reads enough like a task
   description that the model picked "write something" over "give me the
   answer."

6. **Pre-router failure** (source: deterministic pre-check, not the model)
   > "This script is supposed to sum a column but it just prints zero
   > every time — what's going on?" — true: edit, pre-router said:
   > compute (`"compute verb + data noun, no path"` — matched on "sum" +
   > "column").

   This is the one pre-router miss, and it's the deliberately ambiguous
   item described above: the sentence *contains* a compute verb and a
   data noun (which is exactly the phrasing-only compute signal), but the
   actual ask is "fix my broken script," not "compute this for me." It
   was left in rather than patched around, per the brief's instruction
   not to massage the benchmark to flatter the result.

## Open problems

- **The four-way model menu tops out around 76% on the fall-through set**,
  with `navigate` the weakest (62.5%, n=8 — a small slice, so treat that
  number as noisy, not settled). All observed model errors funnel toward
  `compute` or `author`; `edit` and `navigate` never get invented as false
  positives. If this pattern held at scale, a cheap mitigation would be
  biasing the few-shot prefix with more navigate/edit-vs-compute contrast
  examples specifically, since that's where every miss in this run lives.
- **Coverage (69.6%) is currently what the phrasing lists happen to catch,
  not a principled target.** No attempt was made to raise coverage by
  loosening the verb/entity-noun lists — the brief explicitly asked for
  high precision over high recall, and 21 items still fell through in a
  handful of seconds of live model time, so there's no evidence coverage
  needs to be higher for this to be cheap.
- **Sample size is small (21 fall-through items, 3–8 per mode).** A single
  flipped answer moves a per-mode accuracy number by 12–33 percentage
  points. These per-mode breakdowns should be read as a first look, not a
  stable estimate — a follow-up spike would want several hundred
  fall-through items per mode before trusting the per-mode split.
- **The benchmark is self-authored.** All 69 requests (and their ground-
  truth labels) were written by the same person who built the pre-router
  and the few-shot prefix — there is no independent labeler, and phrasing
  habits (mine) could accidentally match phrasing habits baked into the
  few-shot examples (also mine). A second author writing an independent
  benchmark would be the natural next check.
- **The path-mention signal only ever looked inside one small fixture
  folder.** A real deployment would need to decide what "roots" means
  (the whole repo? the open editor buffers? a workspace root?) and that
  choice changes coverage a lot — a big real repo has far more chances for
  an incidental filename match (e.g., a request that merely *mentions* a
  common filename like `utils.py` in passing, with no intent to touch it).
- **No interaction with the actual code vertical was tested.** This spike
  only measures the routing decision in isolation; it says nothing about
  whether getting the mode right/wrong actually matters much downstream
  (e.g., if `edit` and `navigate` verticals share most of their machinery,
  a navigate/edit confusion may be cheap to recover from, while a
  compute/author confusion — running code vs. handing back a script —
  is a much more visible user-facing mistake).
