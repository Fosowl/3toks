# E8 -- Edit Vertical: editing existing code, selection-dominated

Spike for a sibling of the coding vertical (`threetoks/code/`,
`docs/DESIGN-coding-agent.md`): instead of writing a new module from
scratch, apply a bug fix to an *existing* corpus of `.py` files. The
thesis under test: editing is selection-dominated -- navigate to the exact
span (folder -> file -> def, or a free index lookup), then generate **only
the replacement tokens**, never the whole file.

This is exploratory spike code, not shipped. It does not decide whether
the edit vertical is worth building into `threetoks/` -- that call belongs
upstream. It reports what was built and what happened when it ran, honestly,
including the live model's failures.

## What was built

| Piece | File | Maps to task item |
|---|---|---|
| Symbol index: one `ast` pass over a corpus, free lookup by name, call sites | `symbol_index.py` | (a) |
| Folder -> file -> (def \| class -> method) navigation, paged `MenuNode`s, free-take on 0/1-option levels | `nav.py` | (b) |
| Splice primitives (replace/insert-after/delete-span), whole-file re-parse gate, `GenerateSpanNode` (stateless micro-prompt, blind resample) | `edit_ops.py` | (c) |
| Trusted-oracle gate: run a scenario's test before (must fail) and after (must pass) an edit | `oracle.py` | (d) |
| Five fixture scenarios + the state machine that drives one episode | `scenarios.py`, `vertical.py` | (e) |
| Offline scripted run (all 5) | `run_offline.py` | proves the state machine |
| Live run against `qwen3.5:2b` | `run_live.py` | measures the real model |
| Token/decision measurement, delta-economics | `measure.py` | MEASURE |
| Offline unittest for splice+gate machinery | `tests/test_edit_machinery.py` | required deliverable |

Everything is built directly on the real harness: `threetoks.nodes.MenuNode`,
`threetoks.render.Episode`, `threetoks.policy.Policy` (unmodified -- its
existing retry ladder, temperature schedule, and permutation are what make
generation resampling and menu selection work here), and
`threetoks.engine.run_episode`. `edit_ops.py` reuses
`threetoks.code.gates.function_source` / `is_placeholder` / `parses` /
`undefined_names` directly rather than re-implementing them. Nothing
outside `spikes/e8_edit_vertical/` was read as anything but a library import,
and nothing outside it was modified.

### (a) Symbol index

`SymbolIndex.__init__` walks the corpus with one `ast` pass, collecting
every top-level function/class, every method, and every `ast.Call` callee
name with its (file, line). `find_by_name(name)` is the free lookup: 0
hits -> navigate, 1 hit -> **zero model calls**, >1 hit -> one
disambiguation `MenuNode`.

### (b) Navigation

`nav.LevelChoice` wraps a `MenuNode` with paging (`PAGE_SIZE=8`, "show more
options" occupies the last non-escape slot, mirroring how `web/search.py`'s
results pages already page rather than widen) and a **free-take rule**: a
level with 0 or 1 real option never asks the model (the same rule
`research.py` already applies to a single-result menu). `EditVertical`
drives folder -> file -> def, and, only if a class is picked, one more level
for its methods.

### (c) Edit machinery

`replace_span` / `insert_after_span` / `delete_span` are pure line-splice
functions over 1-based `ast` spans. `GenerateSpanNode` is duck-typed
exactly like `threetoks/code/nodes.py`'s `ImplementNode` (`kind`, `system`,
`prefill`, `max_tokens`, `stop`, `render`, `parse`) so it runs through
`Policy`'s existing retry ladder (temps 0.0 / 0.0 / 0.4) untouched -- this
spike adds **no new retry code**. Its `parse()` calls a `splice_fn` that:
reconstructs the candidate function, rejects a placeholder dodge
(`gates.is_placeholder`), splices it into the file, and re-parses the
**whole file** (`gates.parses`) -- any break is rejected for free and the
next ladder attempt (or repair round) resamples blind.

`insert_after_span`'s helper name and arity are found **deterministically**,
zero model calls: `find_missing_helper` reuses
`threetoks.code.gates.undefined_names` (the code vertical's own gate) to
name the undefined callee, then counts the first call site's arguments to
synthesize a parameter list. The model only writes the missing body.

`delete_span`'s target line is **not** hardcoded ground truth -- after
"delete" is chosen, the vertical shows one more `MenuNode` listing every
statement in the target function (via `edit_ops.statement_spans`) and the
model picks which one is wrong. Scenario 3 below shows this menu being
answered wrong once and self-correcting on repair.

### (d) Trusted-oracle gate

`oracle.run_test` execs a scenario's test in a fresh `python3` subprocess
(reusing `threetoks.code.gates.execute`) with the corpus root on
`sys.path`. `EditVertical` runs it once before any decision (must fail --
this is the deterministic pre-check gating everything downstream) and once
after every candidate edit (must pass, or the vertical restores the
original file and, budget permitting, retries).

**Bug found and fixed while building this:** the oracle rewrites the same
file path before and after an edit, sometimes within the same wall-clock
second. A stale `__pycache__/*.pyc` keyed on mtime then silently re-ran the
*pre-edit* bytecode, making an "after" check look identical to "before".
Fixed by setting `PYTHONDONTWRITEBYTECODE=1` for oracle subprocesses
(`oracle.py`). This would have quietly produced false-negative "the fix
didn't work" results, so it is called out explicitly here.

### (e) Five scenarios

All fixtures are invented for this spike (`scenarios.py`); none touch real
repo files. Each plants one bug + a test that fails on the buggy source and
passes on the intended fix (verified independently -- see Offline results).

| # | Scenario | Path to target | Operation | Generation calls |
|---|---|---|---|---|
| 1 | `slice_page` off-by-one | direct lookup (unique name) | replace-span | 1 |
| 2 | `average()` calls undefined `_safe_add` | direct lookup | insert-after-span | 1 (signature found for free) |
| 3 | `to_snake_case` has a stray undo line | direct lookup | delete-span | **0** |
| 4 | `normalize_whitespace` doesn't collapse spaces | no symbol named -> folder->file->def navigation | replace-span | 1 (+ repairs) |
| 5 | `validate()` defined in two files | ambiguous name -> disambiguation menu | replace-span | 1 (+ repairs) |

## How to run

```
# offline: proves the state machine end to end (scripted decisions)
python3 spikes/e8_edit_vertical/run_offline.py

# offline unittest for the splice+gate machinery
python3 -m unittest spikes.e8_edit_vertical.tests.test_edit_machinery -v

# live: real qwen3.5:2b via Ollama, every decision is real
python3 spikes/e8_edit_vertical/run_live.py            # all 5
python3 spikes/e8_edit_vertical/run_live.py 1 3         # just scenarios 1 and 3
```

Each module also has its own `if __name__ == "__main__":` smoke block
(`python3 spikes/e8_edit_vertical/<module>.py`), per repo convention.

## Measured numbers

"Decisions" = distinct decision points (menus + generations), not counting
ladder retries separately. "Output tokens" is summed across **every**
attempt actually made (including retries that were later discarded) -- an
honest accounting, not just the tokens that were finally kept. Whole-file
token counts are the real `qwen3.5:2b` tokenizer's
`prompt_eval_count` for that file's text (one throwaway Ollama call per
file, not a chars/4 guess) -- labelled `ollama-tokenizer` in the raw output.

### Offline (scripted decisions -- proves the state machine)

All 5/5 succeeded, as expected: the scripted backend answers every menu by
grepping the rendered options for a scenario-specific keyword and returns
a hand-written correct completion, so this run is a wiring proof, not an
accuracy measurement.

| Scenario | Decisions | Output tokens | Whole-file tokens | Ratio (output/whole-file) | Savings |
|---|---:|---:|---:|---:|---:|
| 1 slice_page (replace) | 2 | 15 | 111 | 0.135 | 86.5% |
| 2 average (insert-after) | 2 | 5 | 94 | 0.053 | 94.7% |
| 3 to_snake_case (delete) | 2 | 2 | 105 | 0.019 | 98.1% |
| 4 normalize_whitespace (navigate+replace) | 5 | 14 | 144 | 0.097 | 90.3% |
| 5 validate (disambiguate+replace) | 3 | 10 | 87 | 0.115 | 88.5% |

### Live (qwen3.5:2b via Ollama, every decision real)

**2 / 5 scenarios succeeded live.** Reproduced across two full runs (same
pattern both times -- see failure narratives below).

| Scenario | Success | Decisions | Retries | Output tokens | Whole-file tokens | Ratio | Repair rounds | Path taken |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 slice_page | **yes** | 2 | 0 | 52 | 111 | 0.469 | 0 | direct |
| 2 average | no | 4 | 0 | 123 | 94 | 1.308 | 2 | direct |
| 3 to_snake_case | **yes** | 4 | 0 | 12 | 105 | 0.114 | 1 | direct |
| 4 normalize_whitespace | no | 7 | 0 | 112 | 144 | 0.778 | 2 | navigate |
| 5 validate | no | 5 | 0 | 120 | 87 | 1.379 | 2 | disambiguate |

Zero *ladder* retries occurred in any live scenario (every attempt parsed
on the first try, post-indent-fix) -- the failures are all correctness
failures caught by the oracle, not syntax failures caught by `ast.parse`.
"Repair rounds" counts *outer* regenerate-and-recheck cycles after the
oracle rejected a syntactically-valid candidate, capped at
`MAX_REPAIR_ROUNDS = 2`.

### Delta-economics: does selection beat whole-file rewrite?

| Scenario | Live output tokens | Whole-file tokens | Ratio | Reading |
|---|---:|---:|---:|---|
| 1 (succeeded) | 52 | 111 | 0.47 | selection saved ~53% even though it took the ladder's full path |
| 2 (failed) | 123 | 94 | 1.31 | **failure costs MORE than a rewrite** -- 2 repair rounds regenerating the same wrong function |
| 3 (succeeded) | 12 | 105 | 0.11 | delete-only edit: cheapest scenario, ~89% cheaper than rewrite |
| 4 (failed) | 112 | 144 | 0.78 | still cheaper than rewrite, but 3 generations spent on nothing usable |
| 5 (failed) | 120 | 87 | 1.38 | **failure costs MORE than a rewrite** -- wrong file entirely, 3 wasted generations |

The honest reading: **when the edit succeeds, selection is dramatically
cheaper than a whole-file rewrite** (11%-47% of whole-file tokens across
the two live successes and all five offline runs). **When it fails, the
bounded repair loop can make a failed edit cost more tokens than a rewrite
would have** -- because every repair round pays for a full regeneration
without ever being cheap. Any accounting of "does selection win" has to be
paired with the success rate; on this five-scenario, single-run live
sample it's 2/5, so the *expected* token cost of this pipeline (weighting
failures by their real cost, not zero) is not obviously better than
rewriting on this small sample -- see Open problems.

## Concrete failure narratives (live model)

**Scenario 2 -- wrong operation chosen, never recovered.** The request was
"average() crashes with a NameError -- it calls a helper function that was
never written... Add the missing helper." The operation menu's three
options were: replace the target function's body / insert new helper code
after it / delete a line. The model answered digit `1`
("replace the target function's body") instead of the insert option, then
rewrote `average()`'s own body three times (adding `+=`, tweaking nothing
structural) while still calling the undefined `_safe_add` -- the oracle
kept failing with the same `NameError`. **Root cause in the harness, not
just the model:** `EditVertical._apply_and_check`'s repair loop only
reroutes back to the *operation* menu when the chosen operation was
`delete`; for `replace`/`insert_after` it just regenerates the body with
the *same* (here, wrong) operation. A wrong high-level menu pick is
unrecoverable by the current repair loop unless it happens to be `delete`.

**Scenario 3 -- the repair loop working exactly as designed.** The model
correctly chose "delete" both times. On the first pass it picked the wrong
statement (line `s2 = re.sub(r"([a-z0-9])([A-Z])"...)` -- real casing logic,
not the bug); deleting it broke the fix, the oracle failed, the harness
restored the file and rerouted to the operation menu; second time it
correctly picked `result = result.upper()` (the actual stray undo line)
and the oracle passed. This is the one scenario where "wrong menu pick,
then self-correct" is visible end to end.

**Scenario 4 -- navigation was flawless; generation degraded on repair.**
All three navigation menus (folder -> file -> def) and the operation menu
were answered correctly (`textutils` -> `strings.py` ->
`normalize_whitespace` -> replace). The first generated body used
`.replace("  ", " ")` (collapses only literal double-spaces, not the
triple space in the test input `"a   b\tc"`) -- a plausible but incomplete
fix. The **third** attempt regenerated `.replace("\t", " ")` only, i.e.
**reproduced the original buggy line verbatim**. `edit_ops.py`/`vertical.py`
has no hash-based oscillation guard (unlike
`threetoks/code/store.py`'s `MethodRecord.seen_hashes`), so a repair round
can spend a full generation re-deriving the exact bug it was supposed to
fix.

**Scenario 5 -- a real disambiguation failure.** Two files define
`validate`; the request says "it's letting zero-total orders through" --
a human reader maps "orders" to `orders.py` easily, but the 1.5B model's
disambiguation menu (`validate -- duplicates/orders.py` vs.
`validate -- duplicates/users.py`) picked `users.py`. Every subsequent
generation edited the wrong file's `validate` (the email-format checker),
so the oracle (which specifically checks `orders.validate`) failed every
time -- the edit never had a chance once the wrong file was picked, and
(as in scenario 2) the repair loop cannot revisit a wrong `replace`-path
locate decision.

**A bug found and fixed, not just observed:** the very first live attempt
at scenario 1 failed with a genuine `IndentationError`-class break -- the
model's multi-line completion indented every continuation line at 5
spaces against the prefill's 4 (a docstring line, then the body), the same
failure family `docs/DESIGN-coding-agent.md` sec.7 already documents for a
*single* stray leading space (fixed upstream in `gates.reconstruct`) but
one level deeper: **every** line, not just the first. Added
`edit_ops.renormalize_indent` (dedent-and-reanchor at a 4-space level,
preserving deeper relative nesting) as a whitespace-consistency gate ahead
of `gates.function_source`; scenario 1 then succeeded first-try with zero
retries. This is reported transparently rather than silently patched
because it changes the "0 ladder retries" reading above -- without the fix,
every multi-line replace/insert body from this model would likely have
burned its full 3-attempt ladder on indentation alone, which would have
inflated `output_tokens` further and probably converted scenario 4's
already-marginal ratio into a straightforward retry-exhaustion failure.

## Open problems

- **Repair asymmetry.** Only the `delete` path reroutes to the operation
  menu on oracle failure; `replace`/`insert_after` only ever regenerate the
  body. Scenarios 2 and 5 show this concretely: a wrong operation or a
  wrong disambiguation pick is never revisited once locked in. A real
  vertical would need the repair loop to widen its scope (re-ask operation,
  or even re-ask locate) after N body-only repairs fail, not just for
  delete.
- **No oscillation guard.** `threetoks/code/store.py` hashes accepted
  bodies (`seen_hashes`) to stop a repair loop reproducing the same wrong
  answer; this spike's `GenerateSpanNode`/`vertical.py` has none. Scenario
  4's third attempt regenerating the *original bug* verbatim is exactly
  the failure this guard exists to catch upstream.
- **Multi-span edits.** Every operation here touches exactly one
  contiguous span. A fix that needs two non-adjacent changes (e.g. a
  helper signature change plus updating every call site) has no home in
  this design; `call_sites` is collected by the index but never used to
  drive a multi-site edit.
- **Imports.** No scenario needed a new `import` line. `find_missing_helper`
  assumes the missing piece is a same-file function; a fix that requires
  adding `import re` (scenario 4's own candidate fixes happened to inline
  `import re` *inside* the function body, dodging the issue) or reaching
  into another module is unhandled.
- **Non-Python.** The entire index, splice, and gate stack is `ast`-based;
  none of it generalizes to any other language without a different parser
  and a different "does this still parse" gate.
- **Disambiguation is asked as a name-only menu.** The menu label included
  the file path (`validate -- duplicates/orders.py`) but not any content
  cue (e.g. the docstring or a matching request keyword highlighted); a
  1.5B model may need the harness to score/pre-rank candidates by keyword
  overlap with the request rather than presenting a bare file-path choice.
- **Single-run live sample.** Live numbers here are one (reproduced twice)
  run per scenario, not pass@k. `docs/DESIGN-coding-agent.md`'s own E5/E6
  evidence found 83-92% pass rates on the *generation* half of the coding
  vertical with multiple samples; this spike's much smaller live sample
  (5 scenarios, 1-2 runs each) is nowhere near enough to estimate a real
  success rate for the edit vertical, only to surface failure modes.
- **Delta-economics ignores failure cost asymmetry.** The ratio table
  above shows failed scenarios can cost *more* tokens than a whole-file
  rewrite (2 and 5, ratio > 1). A fair economic comparison needs an
  expected-cost model (success-rate-weighted, including the cost of
  detecting failure) rather than a per-success ratio.

## What this does and doesn't prove

This spike demonstrates the mechanics: a real, working folder->file->def and
free-lookup navigation stack, a splice-and-reparse-gate loop reusing the
code vertical's own gates, a trusted before/after oracle, and a state
machine that composes them -- all runnable end to end, offline and live,
with reproducible measurements. It does **not** demonstrate that
`qwen3.5:2b` can reliably pick the right operation or
disambiguate ambiguous names; on this sample it picked correctly for the
two most direct scenarios (unique-name lookup + a syntactically simple
delete-vs-replace choice) and failed on the one requiring an
operation-vocabulary judgment (insert-after), the one requiring
integrating a request-text hint with a file-path choice (disambiguation),
and produced an incomplete fix on the one requiring exact regex/string
logic (navigation scenario, whitespace collapsing). Whether that pattern
holds up, and whether the repair-loop and oscillation-guard fixes above
close the gap, is exactly the kind of question a next iteration -- not this
report -- should decide.
