# ThreeToks — Coding Agent Design (CodeVertical)

Design note for a code-writing vertical, grounded in spike **E5**
(`spikes/e5_coding/`) and a two-persona design debate (tiny-LLM realist,
harness architect). Status: **v1 shipped** (`threetoks/code/`, agent `code`,
282 tests) and validated by spike **E6** (`spikes/e6_whole_module/`) —
**4/6 goals produce a working module on a live 1.5b, 6/6 always parse**.

## 1. Thesis fit

The coding domain is a *better* fit for the ThreeToks thesis than web research,
for one reason: **the judge is free.** Web research needs a model to judge
whether an answer is good (a fallible 1-token decision). Code has an oracle the
harness can run for zero tokens — `ast.parse`, a signature diff, an
undefined-name scan, a subprocess that runs asserts. Every check we move from
"ask the model" to "ask `ast`" is pure profit on 8 GB hardware.

So CodeVertical consults the model only through bounded **generations** (plan
lines, one method body at a time) and verifies each one deterministically. In
v1 there are almost **no menus** in the core loop — a departure from the
menu-centric web/files verticals, and the correct one given the evidence below.

## 2. What the experiments settled (E5)

| Question | Result | Design consequence |
|---|---|---|
| Can it write an atomic method? | **83% first-try, 92% pass@3** (12 tasks) | Yes — generation is the strong part. Keep stateless micro-prompts. |
| Does it honour provided helper contracts? | **3/3** tier-3 tasks called shown helpers by name | Contracts propagate *if handed structurally*. Use example-value contracts. |
| Can it plan in a rigid format? | one-shot parse 0.20–0.40, iterative **0.00** | No. Force the shape at the harness; loose-parse; cap 3–4 methods. |
| Can it grade its own code? | ~75% correct asserts → **~25% wrong expectations** | No gating on model tests. Gate on harness anchor asserts only. |
| Does error-feedback repair beat blind retry? | **0/2 vs 1/2** (small N) | Resample-first ladder (like E4), not a feedback loop. |
| instruct vs coder (1.5b each) | tie on every axis | Use `qwen2.5:1.5b-instruct`; drop the coder model. |

The naive fear ("can a 1.5b write code?") is answered *yes*. The real risk
concentrates in **PLAN quality** and **self-testing** — so the design spends
its complexity there.

## 3. State machine

The harness owns a `MethodStore`: an ordered list of records
`{name, args, contract, status, body, attempts}` where `status ∈
{planned, implemented, stubbed, tested}`. The `.py` file on disk is **always**
rendered deterministically from the store, and a body is accepted only if
`ast.parse` succeeds — so **the file always parses**, at every step, by
construction. The model never sees or rewrites the whole file.

```
PLAN ──▶ (per method, in declaration order)
           IMPLEMENT ──▶ [free gates] ──▶ TEST ──▶ tested
                              │                │
                              ▼ fail           ▼ fail
                           REPAIR ◀────────────┘   (resample-first, max 3)
                              │
                              ▼ budget exhausted
                           STUB (NotImplementedError, spec in docstring)
         episode done when every method ∈ {tested, stubbed}
```

- **PLAN** — one bounded generation, 3–4 method signatures + example-value
  contracts. Validated deterministically (unique valid identifiers, arg lists
  parse, example `ast.literal_eval`s). On failure → **blind retry the plan**
  (no menu — "is this plan syntactically sane" has one computable answer). A
  free back-reference reorder pass sets implementation order (declaration order,
  reordered if method N's purpose names method N+k).
- **IMPLEMENT** — stateless micro-prompt per method: overall goal + the
  signature-and-contract lines of *only the methods this one references* + this
  method's spec, prefill `def name(args):`, ~200 tok, temp 0, stop at next
  top-level `def`. (Scoping context to referenced methods keeps the prompt flat
  as the plan grows — no KV-cache mercy here, each call pays full prefill.)
- **Free gates** (zero model calls, run in order before TEST counts):
  `ast.parse` → a `FunctionDef`/`AsyncFunctionDef` with the right name exists →
  signature matches the plan (arg names/count) → **undefined-name scan** (every
  `ast.Name` load resolves to a MethodStore name, an arg, a local, or an allowed
  builtin) → import-time `exec` of the rendered module in a subprocess.
- **TEST** — run the **anchor asserts** (harness-templated from each method's
  example-value contract) in a 5 s subprocess. Model-written asserts are
  generated and *logged as advisory* but do not gate. A failing anchor assert is
  unambiguous (the anchor is trusted) → straight to REPAIR, no blame menu.
- **REPAIR** — resample the body (reshuffle temps 0.0/0.0/0.4, per E4), max 3
  attempts. Hash each accepted body; if an attempt reproduces a prior one,
  stop early (oscillation) rather than burn the budget. **No error-feedback**
  (E5b showed it underperforms blind resampling).
- **STUB** — on exhaustion, write `raise NotImplementedError` with the spec in
  the docstring. File stays valid; episode continues; the method is a terminal
  `stubbed`.

**Budgets are nested.** An episode-level step budget (forces "stub everything
remaining" when steps run low, mirroring `FORCE_ANSWER_AT_STEPS_LEFT` in the
web/files verticals) plus a per-method attempt cap. This stops a repair loop on
method 2 from starving methods 3–4 of any attempt.

## 4. The one contract mechanism: example values

Each plan line carries a concrete example of its return value:

```
word_frequencies(text)  # e.g. {'dog': 2, 'cat': 1}
```

This single artifact does three jobs, which is why it beats typed stubs or
prose contracts:

1. **Interface contract** — `{'dog': 2}` cannot be misread as a list, a tuple,
   or a differently-keyed dict, so a downstream stateless micro-prompt consuming
   this method's output is unambiguous (prose like "returns the counts" is not).
2. **Trusted anchor test** — the harness templates it into
   `assert word_frequencies(<example input>) == {'dog': 2, 'cat': 1}` with zero
   model involvement, so it is trusted absolutely and gates the TEST step.
3. **Plan validity check** — `ast.literal_eval` of the example must succeed, or
   the plan is rejected and retried.

The harness copies the signature+contract line **verbatim** into every
downstream micro-prompt — never paraphrased, never regenerated (the same
discipline as `Episode.open_observation` rendering prior state).

## 5. v1 scope (deliberate cuts)

Cut, with the reason each cut removes a whole failure class:

- **Free functions only, no classes** — no `self`, no shared mutable state, no
  constructor ordering. Every method's contract is fully expressible in one
  signature+example line.
- **Single file** — matches "harness owns the file"; no cross-file import
  resolution.
- **stdlib-only imports, whitelisted** — a bad import parses fine and fails at
  `exec` time; keep that failure class small and detectable.
- **No plan amendment** — PLAN is append-only and fixed after generation, so
  "declaration order = implementation order" holds for the whole episode. Mid-run
  helper insertion is the most state-machine-expanding feature; defer to v2.
- **No separate docstring-generation step** — pull the docstring text from the
  contract already in the MethodStore; do not spend a generation re-deriving it.
- **No blame menu, no model-gated tests, no error-feedback repair** — all three
  removed by the E5 evidence above.

## 6. From the decomposition literature (what transfers at 1.5b)

- **Transfers:** skeleton-first then fill (skeleton-of-thought), per-hole
  independent testing (Parsel), an explicit externalised plan artifact that
  survives across calls (CodePlan → our MethodStore).
- **Breaks:** any step that asks the model to *self-verify* correctness (the
  whole premise of ThreeToks is that a 1.5b can't); Parsel's search over many
  candidate implementations per hole (unaffordable — one shot + 3 bounded
  repairs, not a candidate pool); SoT's parallel fill (our methods are
  dependent, so fill is strictly sequential).

## 7. What E6 measured, and what changed

E6 ran the shipped vertical on six single-entry goals (easy→hard) against the
live model and graded each assembled module with hidden asserts. Result:
**4/6 whole-module pass, 6/6 parse**. The passes were the easy tier plus
fizzbuzz; the two failures were the predicted ceiling — a decomposition that
needs shared state (`roman_to_int`, honestly stubbed) and a single-token logic
slip (`caesar_encode`'s trailing `.lower()`). Running real code against the
real model surfaced fixes now in v1:

- **Strip the model's stray leading space** in `reconstruct` — it was landing
  the first body line at five spaces and corrupting the module (and cascading
  to every later method's import gate).
- **Guarantee the goal's entry function** (`_ensure_entry`) — the model
  decomposed `word_count` into helpers but omitted the entry; the harness now
  injects the `name(params)` the goal names. This one fix moved 3/6 → 4/6.
- **Trim each body to its own function** — a body ran past its `def` into
  top-level globals that the undefined-name gate then trusted; trimming makes
  that gate sound and keeps methods self-contained.
- **Self-tests OFF by default in the agent** — E6 showed model asserts don't
  change whole-module correctness on a 1.5b (it resamples the same wrong logic)
  and only add latency; the deterministic gates carry the quality. The
  mechanism stays for trusted-example runs, and `verified` now reads True only
  when real asserts actually passed (never vacuously).

## 7a. Known limitations (v1, reviewed and accepted)

An adversarial review confirmed the state machine sound on termination, the
repair/oscillation timing, and the cardinal rule (a wrong advisory assert can
never stub correct code). It found and we fixed two invariant breaks (a `"""`
in a goal/contract, a null byte in model output) and one prose-injection bug in
the entry guard. Three findings are left as accepted v1 limitations because
each fails *safe* (a conservative stub, never wrong code):

- `signature_matches` compares only positional args, so a legitimately planned
  `*args` / `**kwargs` / keyword-only / richly-annotated signature is rejected
  and that method stubs. v1 targets simple free functions; widen to the full
  arg spec when variadic support is wanted.
- `undefined_names` is intentionally conservative (a name bound in a nested
  scope or used-before-assignment is treated as defined) so it never rejects
  correct code; `import_ok` and the asserts catch what it lets through.
- `_ensure_entry` does not reconcile args when the plan already used the entry
  name with a different signature — the plan's version wins.

## 7b. The runner phase (v1.1): execute, blame, repair

v1 gated each method in isolation and never executed the finished module,
so two failure classes shipped silently: a body that gates clean but
crashes at runtime, and a cross-method break (a caller gated while its
helper was still a stub, or a helper whose crash only surfaces through its
caller). v1.1 adds a deterministic runner phase (`code/runner.py`) that
closes both without violating the settled rules — every new decision is
harness code, and the model still only ever sees the same blind
ImplementNode.

- **Widened placeholder gate.** `return NotImplementedError` (returning the
  class instead of raising), bare `return`, `return None`, and `return ...`
  now count as cheats alongside `pass`/`...`/`raise NotImplementedError`.
  All of them parse, import, and "run" — only this gate stops them.
- **Plan lines are validated before they reach the store.** A live run
  produced a planned method `print(i * i)` — a builtin-shadowing name with
  an expression for a parameter list. Its stub rendered as a SyntaxError
  (breaking the always-parses invariant) and every sibling then failed the
  candidate-import gate, cascading the whole module to stubs. `_plannable`
  now drops any plan line whose `def name(args):` does not parse or whose
  name shadows a builtin; an all-junk plan falls back to `main()` as
  before.
- **Checks are deterministic only.** Trusted example asserts, plus a smoke
  call for zero-argument functions (the `main()` fallback path — the common
  TUI case has no examples, and a smoke call still catches runtime
  crashes). Model-written advisory asserts never run here: they stay
  advisory (E6). A check that dies on `EOFError`/`KeyboardInterrupt` is
  *skipped*, not failed — `input()`-driven code is interactive, not broken.
- **Blame is a traceback lookup, not a model call.** Each check runs in its
  own subprocess against the rendered module; on failure the deepest
  `File "<stdin>", line N` frame that lands inside a top-level def names
  the culprit (a stub's `raise NotImplementedError` blames the stub, a
  crashing helper blames the helper). No in-span frame — a plain assert
  mismatch — blames the checked method itself. This replaces the rejected
  blame *menu* with something free and deterministic.
- **Repair re-enters the existing flow.** An unlocked method goes back to
  `planned` with fresh attempts at `REPAIR_TEMPERATURE` — still blind
  resampling (E5b: error feedback underperforms). `seen_hashes` survives
  the unlock, so a resample that reproduces the same wrong body trips the
  oscillation guard instead of looping. Caps: `MAX_REPAIR_ROUNDS = 2`
  run→unlock cycles, `MAX_METHOD_REPAIRS = 2` per method, and no unlock at
  all when the step budget is nearly spent (report-only run).
- **Implement-time triage.** When a method's trusted example fails and the
  blame lands on a *different* method, the innocent body is kept
  (unverified) and the culprit is unlocked — or, if the culprit simply
  hasn't been written yet, the check defers to the runner phase. This
  fixes the v1 ordering flaw where a caller burned its three attempts and
  stubbed because its helper didn't exist yet.
- **Honest limitation.** Blame needs an exception frame. A helper that
  returns a *wrong value* (no crash) fails its caller's assert with no
  frame inside the helper, so the caller takes the blame and the repair
  budget. Fixing that would need value-level attribution — model judgment
  the blame-menu measurement already rejected.

The result payload now carries the run report (`run.ok`, per-failure
error + blamed method, rounds used) and per-method `repairs` counts.

- **The delivered module is a script.** The *answer* render appends
  `if __name__ == "__main__":` calling a deterministically chosen entry —
  a bodied zero-arg `main`, else the goal-named entry when zero-arg, else
  the module's sole zero-arg function; never a stub, never an arbitrary
  helper, no guard when nothing qualifies. The guard is harness-rendered
  (the implement node's stop sequences already cut the model off at
  `\nif __name__`) and appears **only** in the deliverable: gates and the
  runner keep using the guard-free render, otherwise every candidate
  import would execute the entry while siblings are still stubs. The
  runner's zero-arg smoke call executes exactly what the guard will, so a
  clean run report means `python3 module.py` runs.

## 8. What's left for v2 (historical; §9 shipped part of it)

The plan step remains the highest-leverage lever (the one fix targeting it
bought a whole point): plan **coherence checks** (reject a plan whose later
contract names a noun absent from earlier ones) and **plan amendment** (add a
missing helper mid-run — deliberately cut from v1 because it makes the method
list non-append-only). Beyond that: a repair pass that feeds a *negative
example* rather than resampling blind, and promoting model asserts from
advisory to gating only once a trusted oracle is present.

## 9. v2: modes, editing, and retrieval (E7 / E8 / E9)

Three spikes (spikes/e7_code_routing, e8_edit_vertical, e9_code_retrieval)
measured the coding agent's expansion beyond greenfield generation. All
three shipped; the evidence and the settled rules:

### 9a. Mode routing (`threetoks/code/route.py`, E7)

Every coding request is first routed into one of four modes — navigate /
edit / compute / author — by deterministic pre-checks (path-mention +
verb cues, phrasing shapes), falling through to one one-token menu.
Measured live on a 69-request benchmark: **69.6% routed free at 97.9%
precision, 76.2% menu accuracy on the remainder, 91.3% end-to-end at ~3
output tokens per model-routed request**. The top-level agent router
stays heuristic-free by design; this sub-router earned its pre-checks
with that measurement. Every observed menu error funnels into
compute/author (the least damaging confusions — nothing is wrongly
routed *into* a mutating mode). The modes map to: free symbol-index
lookup (navigate), the edit vertical (edit), CodeVertical + executing
the script and answering with its output (compute), CodeVertical
delivering the module (author).

### 9b. The edit vertical (`threetoks/code/edit/`, E8)

Editing existing code is selection-dominated — the content already
exists, so the model points and only the delta is generated. One ast
pass indexes every def/class/method + call sites; "where is X" resolves
free when unique. Navigation (folder → file → def → method) and
disambiguation are keyword-ranked with request-token overlap (path,
qualname, docstring, AND body identifiers — body words navigated E8's
scenario 12 for free); a strict ranking winner is taken for FREE, ties
show enriched labels, an escape backtracks one level (bounded) instead
of aborting. Operations are pre-filtered to what can actually run:
insert-after only when the target provably calls an undefined name
(then it is *inferred* free, no menu — the "known" set covers imports
and module constants, not just defs), delete only when a deletable
statement exists (never the def line itself). Splices are gated free:
placeholder dodge, oscillation (seed hashes cover the original in
candidate-canonical form, with and without docstring), whole-file
re-parse. The trusted oracle is the caller's test (fail-before /
pass-after, subprocess, bytecode writes disabled); without one, the
edited module must import and the result reports `verified=False`.
Repair escalates per target — regenerate once, re-ask the operation
(inference suppressed), relocate to the runner-up — the 6th failed
check ends the episode.

**Load-bearing boundary (E5b):** menu decisions read the growing
session log (what was located, what failed) — that memory is what makes
a re-asked menu meaningful. Generation decisions get a fresh stateless
episode every time and never see failure text: repair is blind
resampling, because error feedback measurably underperforms it.

Measured on 14 planted-bug scenarios live (eval/run_edit_eval.py):
**10-11/14 across runs, 13/14 expected-operation choice, successful
edits cost 3-92% of a whole-file rewrite** (failures can cost more than
a rewrite — the expected-cost argument rests on the success rate, which
is why the escalation ladder exists). The spike's original 5 scenarios
went from 2/5 live (E8) to 4-5/5.

### 9c. Two-anchor contracts and retrieval-as-repair (E9)

Trusted contracts now carry a *list* of anchor examples. E9's proof:
one anchor accepted an `is_prime` that returns True for 0 and 1 —
boundary cases are exactly where classic implementations diverge, so
pair a normal case with a boundary case.

`threetoks/code/retrieve.py` is the repair tail for the E6 failure
class (classic textbook functions): when an anchor-bearing method
exhausts its generation attempts and would stub, search the public web,
extract candidate defs from `<pre>`/`<code>` blocks (E9: every accepted
snippet came from explained-code pages, none from raw GitHub — and
unauthenticated GitHub code search is dead), gate with a TIGHT
whitelist (pure-computation stdlib only, forbidden-builtins scan,
dunder trip wire), and accept only what passes every anchor in a
subprocess. **No model call anywhere — execution is the judge** (7/8
classics, 0/4 false accepts on invented names in E9). Accepted snippets
are provenance-stamped and cached (`[code] snippet_cache`) so repeat
misses are free. Mojeek leads the zero-infrastructure search hops
(E9: it answered plain HTTP while Bing/DDG were bot-walled).

**Off by default** (`[code] retrieval` in config.ini): it executes and
embeds internet code; the whitelist + subprocess timeout are a speed
bump, not a sandbox, and licensing is recorded (source URL) but not
resolved. Turning it on is an explicit, informed choice.
