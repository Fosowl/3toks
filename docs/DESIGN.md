# 3 toks — Run agents in a cave with a box of scrap.

**Goal:** a useful agent running on a <8b model on 8GB-class hardware, with sub-second decision steps.

**Secondary Goal:** [Colibri](https://github.com/JustVugg/colibri) Compatibility for fast GLM5.2 Agentic performance on SDD+decent GPU. Run Fable level agent on a Potato?

**Name:** the agent walks a *tree* of decisions, spending near-minimal
*tokens* at each node.

---

## 1. Principles

1. **Classification over generation.** Small models are poor generators but
   decent classifiers. Every decision the harness can phrase as a numbered
   menu, it does. Free-text generation is the exception, tightly bounded.
2. **Constrained, not requested.** Output format is enforced by decoding
   constraints (prefill steering today, grammar/logit-mask when the backend
   supports it), never by politely asking. Probes showed the 1.5b ignores
   "reply with one digit" but complies when the decode path is forced.
3. **KV-cache-first prompts.** With ~1 output token, prefill is the entire
   cost. Prompts are append-only within an episode: stable prefix (system +
   few-shot), then a strictly growing log. Nothing near the top of the prompt
   may mutate between steps. Compaction happens only at episode boundaries.
4. **The harness owns everything deterministic.** Parsing, page rendering,
   note storage, plan bookkeeping, undo — all code. The model is a policy
   oracle consulted at decision nodes.
5. **Compute on demand.** Thinking is not banned; it is a *purchasable
   action* with a bounded token budget, triggered by uncertainty or chosen
   from the menu. Cheap by default.
6. **Recoverable by design.** A 1.5b will misclick. Every menu carries an
   escape option (rendered as an ordinary numbered LAST option — E4
   showed models never emit an out-of-distribution `0`), dead options
   disappear from menus after failure, and step/search budgets force
   termination. True undo/checkpointing is future work, required before
   any vertical that *mutates* state (web reads are idempotent).
7. **Measure everything.** Every decision is traced (prompt, options, choice,
   tokens, latency, template id). Traces feed the eval suite and the
   template bandit.

## 2. Why this can work: the token math

A ReAct-style agent emits 200–500 tokens per step; on an M1 at ~15 tok/s
that is 15–30s per step. A ThreeToks menu step emits 1–3 tokens, and with KV
reuse only the new prompt suffix is ingested → ~0.3s per step (measured).

The threat is compounding error: a 20-step task at 90% per-step accuracy
succeeds 12% of the time; at 98% it is 67%; at 99.5%, 90%. Therefore:
per-step accuracy is the metric that matters (E1), and cheap
detection/undo/voting is a first-class mechanism, not a nice-to-have.

## 3. Decision nodes

Every interaction with the model is one of five node types:

| Node | Model output | Enforcement |
|---|---|---|
| `MENU` | one digit `1..N` (escape = numbered LAST option; `0` always invalid) | prefill `ANSWER:` + `num_predict≤3`; grammar when available |
| `PICK_MANY` | comma-separated indices, e.g. `2,5,9` (capped at max_picks) | prefill `RELEVANT:` + `num_predict≤24` + newline stop |
| `SHORT_TEXT` | bounded free text (search query, final answer) | `num_predict` cap + stop tokens |
| `CONFIRM` | `1` (yes) / `2` (no) | as MENU (helper over MenuNode) |

Bounded thinking is a *policy parameter* (`think_budget`, R1-family only,
off by default — E1 showed it doesn't pay), not a node type.

Node contract: ≤9 numbered lines per menu including the escape, every
option is a short verb phrase, option order is shuffle-able (templates must
not depend on position), gold answers in evals are option *texts* not
positions.

## 4. Prompt layout (append-only)

```
[S] system + answer-format few-shots     — static per vertical, cached forever
[T] task statement                       — static per episode
[L] rolling log: compact action history + notes taken   — append-only
[O] current observation (numbered menu / numbered sentences)
[A] "ANSWER:" prefill
```

Rules: no step counters or timestamps above [O]; [L] entries are one-line
(`> searched "france population"`, `> noted s3,s7 from page#2`); raw page
text lives only in [O] for the current step and is replaced — never
accumulated — because notes ([L]) carry the useful content forward.
Episode-boundary compaction rewrites [L] into notes-only form (accepting one
full re-ingest).

## 5. Model strategy (8GB M1) — settled by E1/E3

- **Policy model for ALL nodes: `qwen3.5:2b`** (ChatML +
  `menu_picker` system prompt + prefill): 76.7% menu accuracy, 97.5%
  format compliance, 0.285s/decision (E1); 85% note-pick recall (E3).
  Single-model design — no model swapping on 8GB.
- **`deepseek-r1:1.5b` is disqualified for decisions** (E1: 23-26% in all
  variants, strong 0-escape bias; E3: 35% recall, intro-anchoring).
  R1 raw-mode think suppression stays supported in the backend, but the
  distill's menu-following is broken. Bounded think (80 tok) did not
  rescue it (25.8% at 12x latency).
- **Structured-output JSON hurts** both models vs prefill (E1: -9pts,
  ~3x latency) — constrained decoding via grammar remains interesting for
  the llama.cpp backend, but Ollama-schema is not the path.
- **Escalation tier:** `deepseek-r1:7b` fails to load at 2048 ctx
  (HTTP 500; 4.7GB + KV > 5.3 GiB Metal budget) and at 512 ctx runs at
  ~0.8 tok/s with 9s prompt ingest (measured) — unusable. Config-gated off
  by default on ≤8GB; the ladder ends at voting on such machines.
- Contexts capped at 2048 tokens (also Ollama's default) — small context is
  a feature: prefill stays fast.

## 6. Escalation ladder (cost-ordered, calibrated by E4)

1. **reshuffle option order, temp 0** — recovers 29% of failures for
   +1 call (temp-0 failures are *confident*, so same-prompt retries with
   temperature do nothing: 3.6%)
2. **rephrase the menu frame** (template bandit, Phase 2) — 39% recovery
3. temperature bump (last resort, attempt 3)
4. ~~self-consistency vote~~ — rejected as default: +0.8pt overall and it
   *entrenches* consistent errors (navigation regressed). Available via
   `PolicyConfig.vote_k` for irreversible actions only.
5. bigger model (config-gated off on ≤8GB) / surface to user

Escape blindness (E4's unrecoverable class — models never emit "0"):
fixed structurally by rendering the escape as an ordinary numbered
option in last position. Live probe: 3/3 correct incl. two none-fit
cases that scored ~0% as "0".

## 7. Backend abstraction

`LLMBackend` protocol: `complete_raw(model, prompt, opts) -> GenResult` plus
typed helpers `choose / pick_many / short_text`. Adapters:

- **OllamaAdapter** (now): raw-mode templating done by us (R1 and ChatML
  templates in code), prefill steering, `num_predict` caps. KV reuse is
  implicit via Ollama's prompt cache (E2 verifies).
- **LlamaCppAdapter** (later): GBNF grammars for hard constraints, explicit
  `cache_prompt` slots, logit bias. The serious target for guarantees.

## 8. Web-research vertical (first vertical)

Informed by agenticSeek recon (see `docs/recon-agenticseek.md`):

- **Search:** `SearchProvider` protocol. `SearxngProvider` POSTs to a local
  searxng docker container (upstream image, `formats:[html]`,
  `limiter:false` — reuse agenticSeek's `searxng/` compose almost as-is)
  and parses `article.result` blocks with bs4. `DdgsProvider` as a
  zero-infrastructure fallback. Auto-detect at startup.
- **Fetch:** plain HTTP (`requests`) + bs4 + markdownify, adapting
  agenticSeek's `Browser.get_text()` pipeline (strip script/style →
  markdown → `is_sentence()` noise filter). No Selenium/Chrome in the
  prototype: ~1GB RAM and seconds of wall-clock per page don't fit the
  budget; the Fetcher is a protocol so a browser backend can slot in later.
- **Render:** numbered sentences `[1] … [2] …` (capped per page, paginated
  `MENU: next chunk / back / answer`); every freshly shown chunk gets an
  automatic PICK_MANY note pass (opening a page implies noting).
- **Notes:** `PICK_MANY` over sentence indices → verbatim quotes stored with
  provenance (url, sentence idx). Notes are hallucination-free by
  construction; the model never rewrites content, it points at it.
- **Answer:** final synthesis is the only free-generation step (SHORT_TEXT
  from notes), or for extractive tasks, a `PICK_MANY` over notes.

## 9. Repository layout

```
docs/               design + recon + experiment reports
spikes/common/      shared minimal ollama client (stdlib only)
spikes/e1..e4/      Phase-0 experiments (kept for provenance)
threetoks/           the framework package (Phase 1)
  backend/  nodes.py  engine.py  render.py  policy.py  trace.py
  (escalation ladder lives in policy.py; the rephrase rung is Phase 2)
  web/      (vertical)
eval/               benchmark tasks + runners
tests/
```

## 10. Phases

- **Phase 0 (now):** spikes. E1 menu accuracy (go/no-go: need ≳95% on the
  policy model), E2 KV-cache latency, E3 sentence-pick note quality,
  E4 reformulation recovery. Numbers decide defaults.
- **Phase 1:** core engine + web vertical + tracer; end-to-end research
  tasks on 1.5b.
- **Phase 2:** eval suite vs a ReAct baseline (same model), template bandit
  learning loop, llama.cpp backend with grammars.

## 11. Phase-0 results (2026-07-05)

| Spike | Headline |
|---|---|
| E1 menu accuracy | qwen-prefill 76.7% / 0.285s / 3 tok; r1 disqualified (23-26%); JSON-schema hurts both. Weak spots: navigation 46%, escape under-used. |
| E2 KV-cache | Append-only prompts: prefill flat ~300ms as prompt grows; identical re-send 931ms→27ms; mutation at top forces full re-ingest. Signal = `prompt_eval_duration`, not `prompt_eval_count`. |
| E3 note-picking | qwen picks: 0.85 recall at ~12 tok; freetext notes cost 3.7x tokens, land answer less (65%); >40-sentence pages crater recall (recommended ≤35; shipped chunk = 25). |
| E4 reformulation | 50% of failures recoverable (union): reshuffle 29%/+1 call, rephrase 39%/+2; temp-retry 3.6%, helper 11%, vote3 21%. Vote3 global: 76.7→77.5 only, navigation regresses. All escape failures resisted prompting → fixed by numbered-escape rendering instead. |

Per-step accuracy at 76.7% is far below the §2 target — the escalation
ladder (E4) and voting must close the gap, and the vertical's guards
(reversible actions, forced-answer budget) absorb what remains.

## 12. Open questions

- Does bounded-think beat think-suppressed on hard menus enough to justify
  its latency? (E1 C2 vs C1)
- Is qwen2.5-instruct a better no-think policy than the R1 distill? (E1)
- Does Ollama's prompt cache survive our append-only layout in practice, and
  what does a cache miss cost at 2k context? (E2)
- Per-step accuracy → maximum viable task depth; do we need voting by
  default on irreversible actions? (E1 + math in §2)

## 13. Phase 2 (2026-07-05)

Phase 2 turned the single web-research episode into a small mesh of agents
around a judged multi-round research loop, and added an interactive TUI.

**What shipped:**

- **Interactive TUI** (`threetoks/tui.py`, `python3 -m threetoks`): a REPL
  that streams every decision live as a ticker line (node kind, chosen
  value, tokens, latency) so the model's ~1-token-per-step behavior is
  visible as it happens, not just in a trace file afterward. Slash
  commands: `/help`, `/agents`, `/deep N`, `/browser on|off`, `/model NAME`,
  `/quit`. Pure formatting/dispatch functions are separated from the input
  loop so the TUI is unit-testable without an LLM or network
  (`tests/test_tui.py`).
- **Pluggable agents** (`threetoks/agents/`): an `AgentSpec` contract
  (`name`, `description`, `run(task, services, policy) -> dict`), a
  one-token router (`route()`), and three built-ins — `casual` (canned
  replies ranked by keyword overlap, zero-model-call exact-hint shortcut,
  bounded free-text fallback), `files` (read-only sandboxed file explorer,
  same menu/notes/answer shape as the web vertical), `web` (thin wrapper
  over the new deep-research loop). Registry (`default_agents()`) is a flat
  list; adding an agent is one module + one line — see `docs/AGENTS.md`.
- **Deep research** (`threetoks/research.py`): judged rounds. Each round runs
  a full `WebResearchVertical` episode, then a judge decides accept/reject;
  on reject, the model proposes a new search query that must differ from
  every query tried so far, and the next round reuses the *same*
  `NoteStore` so evidence accumulates instead of resetting per round.
  `threetoks.cli` now calls `deep_research` directly (`--rounds N`).
- **Selenium browser fetcher** (`threetoks/web/browser.py`): drives a real
  Chrome (visible by default, `visible=False` for headless) via plain
  selenium — no stealth, no undetected-chromedriver, no anti-bot tricks —
  waits for `document.readyState == "complete"` plus a fixed settle delay,
  then hands `page_source` to the same `html_to_page` pipeline the HTTP
  fetcher uses. Wired into the TUI as `/browser on|off`; lazy driver
  creation so importing the module never launches Chrome.
- **Web vertical upgrades** (`threetoks/web/vertical.py`,
  `threetoks/web/search.py`): link following (`follow a link on this page`
  menu option, backed by `PageText.links` from `textify.py`); keyword-
  guided chunk entry (`_best_chunk_start` scores each chunk by task-keyword
  overlap so a long page opens near the relevant section instead of always
  at sentence 1); `FallbackProvider` now falls back from searxng to DDG on
  an **empty result list**, not only on a request failure.

**Decisions worth recording:**

- **Routing-as-menu.** Agent selection is the same `MenuNode` decision as
  everything else in the harness — one digit over agent
  name+description — rather than a separate classifier or keyword match.
  Consistent with §1 (classification over generation): the router is just
  another node type consumer, gets the same escape/retry handling for
  free, and needs no new infrastructure. Fallback on an invalid pick is the
  first registered spec (`casual`), chosen because it is the cheapest and
  safest default action.
- **Judge: deterministic pre-checks before a semantic model decision.**
  `research._obviously_bad` rejects empty, digits-only, or note-less
  answers in code, with zero model calls, before anything reaches the
  judge. Only genuinely ambiguous cases go to the model, and even then not
  as an abstract yes/no: the judge is a two-option `MenuNode` with option
  text describing what a good vs. bad answer looks like for the task. This
  mirrors the router: menus over meaningful text, not raw booleans.
- **Yes/no confirm was a coin flip for the 1.5b.** Early judge iterations
  used `confirm_node` (plain "yes"/"no"). Behavior was close to random —
  consistent with E1's broader finding that the model's accuracy comes
  from picking among *described* options, not from parsing an abstract
  binary. Rephrasing the same decision as `JUDGE_GOOD` /
  `JUDGE_BAD` — full sentences describing what each verdict means — fixed
  it. Lesson generalized into a design rule (`docs/AGENTS.md`): prefer
  semantic option pairs over `confirm_node` for any decision that requires
  judgment rather than a mechanical toggle.
- **Search fallback on empty results, not just unreachability.**
  `FallbackProvider` originally only fell back to DDG when the primary
  provider raised (network error, non-200). In practice, both searxng's
  upstream engines and DDG rate-limit or return a technically-successful
  page with zero parsed results under heavy use — a failure mode that
  looks identical to "no results exist" from the caller's side. Falling
  back on `not results` as well as on exceptions catches this; it does not
  catch a rate limit that returns junk-but-nonempty markup, which is why
  the README calls out search-provider quality as an honest limitation
  rather than something the fallback fully absorbs.

## 14. Phase 3 (2026-07-07): coding vertical

A fourth agent, `code` (`threetoks/code/`), applies the thesis to program
synthesis. Full design and evidence in `docs/DESIGN-coding-agent.md`;
spikes E5 (per-method generation) and E6 (whole-module) in `spikes/`.

**Why coding fits the thesis better than research:** the judge is free.
Web research needs a fallible 1-token model judge; code has an oracle the
harness runs for zero tokens (`ast.parse`, signature diff, undefined-name
scan, a subprocess). So the vertical consults the model only through
bounded *generations* (a plan, then one method body at a time) and verifies
each deterministically. In v1 the core loop has almost no menus.

**Measured (live qwen3.5:2b, M1 8GB):** per-method generation
83% first-try / 92% pass@3 (E5a); whole-module 4/6 goals working, 6/6
always parse (E6). The file on disk always parses by construction — a body
is stored only after ast checks; unfilled methods render as
`NotImplementedError` stubs.

**Decisions worth recording:**

- **The MethodStore owns the file; the model never round-trips whole code.**
  An ordered list of method records renders deterministically to the module.
  This is Phase-2's "harness owns state" (the NoteStore, the Episode)
  applied to source.
- **Model self-tests are advisory, never gating.** E5d showed ~25% of
  model-written asserts encode a wrong expected value. A body that fails
  only advisory asserts keeps its runnable body; only a deterministic gate
  or a trusted example (harness-templated from a known input→output) can
  stub it. E6 confirmed this live — the model scored its own correct code
  "wrong" and the harness kept the correct code.
- **Blind resample beats error-feedback repair (E5b), so the repair ladder
  reuses the E4 temperature bumps** rather than a feedback loop; self-tests
  are off by default in the agent because they don't move whole-module
  correctness on a 1.5b (E6).
- **Structure is forced at the harness, not asked of the model.** E5c showed
  the model won't emit a rigid `name(args): purpose` plan; the parser is
  lenient and the harness guarantees the goal's named entry function exists
  (`_ensure_entry`) — that single fix moved E6 from 3/6 to 4/6, confirming
  the debate's call that plan quality is the top lever.

## 15. Target belief and strategy requery (2026-07-10)

A live trace ("search what Fosowl has built") exposed the deep-research
loop's blind spot: `propose_query` saw only the task and the tried-query
strings — never the notes — so the 1.5b paraphrased the task for ten
rounds ("Fosowl project details" → "… overview" → "… specifications")
while the one real discovery (the target's GitHub repo, opened in round
2) never influenced a single later query. Junk notes (© footers,
`curl | bash` lines) also outlived every round: per-episode curation
rebound only the vertical's store reference, and the curate pool
silently showed the FIRST 30 notes of an append-only store, so
late-round evidence was invisible.

**What shipped (`threetoks/web/target.py` + wiring):**

- **`TargetBelief`** — what the target appears to be: one label from a
  five-entry taxonomy (github user / project / company / person / topic)
  picked by a 1-token menu after each rejected round, plus a verbatim
  note fragment as *anchor* (one PICK over the top notes). Re-picking
  the same label confirms it; two confirmations settle the belief and
  stop all further update calls. Escape or an invalid answer changes
  nothing.
- **Strategy requery** — `propose_query` now offers a menu of
  harness-composed queries (`site:github.com` filters, profile/social
  angles, subject + anchor keyword) built from the belief;
  already-tried and paraphrase strings are struck out in code, so the
  requery loop can no longer degenerate. "Write a different query
  yourself" stays as an ordinary option; a written query survives only
  if it passes `too_similar` (content-word coverage ≥ 0.6 against any
  tried query falls back deterministically).
- **Belief in the episode** — the next round's `WebResearchVertical`
  logs `TARGET SO FAR: <label> — best note: <anchor>` into its history
  before the first search, so in-round decisions know what the target
  turned out to be. Append-only discipline holds: each round is a
  fresh episode.
- **Curation persists** — `deep_research` adopts `vertical.notes` after
  each round, so a curation prune removes junk for good; and
  `curation_pool` windows an overflowing store to the head keepers plus
  the newest notes instead of the oldest 30.
- **Junk never displayed** — textify additionally drops © footers,
  copyright-year lines, and shell-command lines (`$ …`, `curl -…`,
  `… | bash`), so they can never become notes.

**Info flow (what each new node sees):**

- belief menu: `BELIEF_PREFIX` + task + top-8 notes → 5 taxonomy
  options (+ escape); tag `belief`.
- anchor pick: `ANCHOR_PREFIX` + task + top-8 notes → pick ONE note;
  tag `anchor`.
- strategy menu: `STRATEGY_PREFIX` + task + belief line + tried queries
  → ≤5 composed queries + the write-your-own option, which doubles as
  the escape hatch (no separate escape row); tag `requery`.
- written requery: `QUERY_PREFIX` + task + belief line + tried queries.

**Token cost:** ~3t belief + ~5t anchor (usually once) + 3t strategy
menu per rejected round, replacing the 8–14t written requery — roughly
neutral, often cheaper.

**Decisions worth recording:**

- **Belief as a menu, not free text.** The motivating trace shows the
  model's free text degenerating under pressure; a fixed taxonomy costs
  1 token and cannot hallucinate. The specific anchor stays extractive
  (a verbatim note), consistent with §8's notes-are-verbatim rule.
- **Settle on confirmation, not on a round count.** A hard freeze risks
  locking in a wrong belief formed from junk pages; requiring the same
  label twice makes settling an earned state, and until then an update
  costs ~3 tokens.
- **Compose queries in the harness; let the model pick.** The same
  shape as every other lesson here: the model chooses among described
  options (site: filters, social angles) it would never write itself.

## 16. Memory that pays for itself (2026-07-11)

The recall step existed but its output went nowhere: after routing, the
selector spent ~10 tokens picking `services.recalled` entries and no
agent ever read them. Memory also kept everything — including
"(no answer)" and judge-rejected junk — so the Fosowl trace recalled
three failed past attempts as "context". Fixed at every edge,
deterministically:

- **Recall now reaches the model.** `render_recall` renders the picked
  (query -> answer) pairs as an `EARLIER ANSWERS` block;
  `deep_research` logs it into every round's episode through the
  vertical's generalized `context_lines` — the same fold-once-into-
  history mechanism as the `TARGET SO FAR:` line (§15).
- **Junk never enters memory.** `worth_remembering` (code, zero
  tokens) refuses empty answers, the "(no answer)" placeholder, and
  results the judge panel rejected. What cannot be recalled usefully
  is never stored.
- **The store stays bounded.** A same-query (case-folded) entry is
  replaced instead of duplicated, and each agent keeps only its newest
  `MAX_PER_AGENT` (64) exchanges, so the JSON file cannot grow without
  limit.
- **The selector is shortlisted for free.** `MemoryStore.relevant`
  scores the whole store by 4+ character keyword overlap (query plus
  answer text, function words excluded) against the task; zero overlap
  means zero candidates and zero model calls, while a task with no
  usable keywords at all ("who is he?") falls back to the newest
  entries — recency is the only signal an underspecified follow-up
  gives. The model sees at most 8 candidates — it used to see the 32
  newest regardless of relevance — each with a flattened, clipped
  answer, so it judges usefulness rather than query similarity alone.

Token cost: the selector call is now skipped outright on unrelated
tasks, and when it does run its observation is roughly 4× smaller.

## 17. Strict search dedup and diversified requeries (2026-07-12)

A live trace showed judged rounds burning their whole budget on
repetition: the same written requery proposed round after round, the
same pages reopened, the same sentences re-picked (and silently dropped
by note dedup — a full round of model calls for zero new evidence).
Root causes: a temp-0 sampler given a near-identical prompt writes the
identical query, and every round got a fresh `seen_urls`, so exhausted
pages looked new again. Three changes:

- **A query never runs twice.** `normalize_query` (case, quotes,
  spacing) keys a `tried_queries` set that `deep_research` shares
  across every round's vertical; `_run_search` skips a duplicate before
  the provider is called — zero cost — and logs
  `> already searched '…' — use different words` into the session so
  the model is told to move. The same normalized comparison now guards
  the strategy menu, the written-requery check, and the fallback.
- **A page never reopens.** The vertical accepts a shared, mutable
  `seen_urls`; a URL harvested in round 1 is excluded from result
  menus and link menus in every later round. New rounds must produce
  new evidence or answer.
- **The written requery cannot repeat itself.** The node runs at
  `QUERY_TEMPERATURE` (0.7) instead of the temp-0 ladder — deliberate
  diversity, same lever as the coding vertical's blind resample — and a
  rotating `ANGLE_HINTS` nudge ("aim at official or primary sources",
  "name a specific person, organisation, or place", …) indexed by the
  round count makes the prompt itself differ from every earlier
  requery. Rotation is deterministic (no RNG), so traces stay
  reproducible up to sampling; junk or repeats still fall back to the
  composed strategy candidates.

The E4 rule "temperature only enters at attempt 3" is about *menu
accuracy* and still holds; query *writing* wants diversity, not
determinism, and its failure modes stay caught by the free checks
(`too_similar`, the normalized dedup gate).

## 18. Search-layer circuit breaker and site fallback (2026-07-12)

A live trace ("best pizza in antibes") burned 85 decisions, 458 tokens
and 147 s across 10 rounds to answer "no results found". The postmortem
found the model blameless: the whole IO layer was down — the local
SearXNG had every upstream engine suspended ("Suspended: too many
requests / CAPTCHA"), Bing served a captcha shell (HTTP 200, zero
organic results), and DDG's HTML endpoint refused TCP — and nothing in
the loop could tell "no hits for this query" from "search is down".
Worse, every result that *was* opened failed: Bing wraps organic hrefs
in `bing.com/ck/a` JavaScript redirects that a plain fetch cannot
follow, so even Wikipedia results produced zero notes. With zero notes,
`update_belief` never labels, the judge auto-rejects every round, and
the loop degenerates into query-writing theater until the round budget
dies. Fixes, all deterministic harness code:

- **Failure is distinct from emptiness.** A provider that is blocked
  raises `SearchUnavailable` (SearXNG "Suspended:" boxes, Bing's
  `/challenge/verify` shell, DDG's anomaly page) instead of returning
  `[]`. The now-flat `FallbackProvider(*hops)` counts calls where every
  hop failed (`consecutive_failures`, `last_call_failed`); any hop that
  answers — even with zero hits — resets the count.
- **Circuit breaker.** After `SEARCH_ABORT_FAILURES` all-hops-failed
  searches and with zero notes gathered, `deep_research` returns the
  honest `(search unavailable: …)` answer instead of running further
  rounds. Inside a round, a dead end (nothing to open, no searches
  left, no notes) ends the episode without the synthesis/judge theater
  — code already knows the verdict.
- **Site fallback (LLM + fetch as last-resort search).** When every
  hop failed, one `SITES:` node asks the model to name up to 3 website
  URLs from its own knowledge; the harness parses, dedupes against
  `seen_urls`, and renders them as ordinary openable results — the rest
  of the pipeline (fetch, error page detection, notes) is unchanged.
  One ask per round, capped at 3 sites.
- **Bing redirect decode.** `_decode_bing_href` unwraps the base64url
  `u=a1…` payload, mirroring the DDG `uddg=` decode, so Bing-sourced
  pages actually open in http mode.
- **Dedup hardening.** Typographic quotes (`“…”`) no longer evade
  `normalize_query`; rejected duplicates/paraphrases cost no search
  budget but burn a 2-strike rejection budget so paraphrase loops end
  the round; the in-round written query faces the same `too_similar`
  check as the round-level requery, and each round's episode opens with
  a `QUERIES ALREADY TRIED` line so the model can see what failed.
  Requery proposals (menu, written, fallback) are additionally struck
  against every query the verticals *executed* — in-round writes
  included — so a proposal can never hand the next round a query its
  own dedup gate would refuse on arrival. The dead-end label keys on
  `last_call_failed` (this round's last live search), not the
  cross-round counter, so a cache-served round ends "(no answer)"
  rather than falsely claiming the layer is down.
- **Degenerate menus never reach the model.** A results menu whose one
  real option leads where its escape also leads (lone "search with
  different words", lone "answer the task now") is taken by the
  harness for free.

Live re-run of the failing trace with the search layer still dead: 4
model calls and an honest unavailability answer when the model names no
usable site, or 11 calls and a judged-good answer when the site-ask
lands on a fetchable page — versus 85 calls of theater before.

## 19. Multi-provider transports and hardware extras (2026-07-14)

Integrated (adapted, not copied) from the reference voice-assistant
project: chat-API providers, voice in/out, camera vision, Pi relay.

- **Chat transports** (`backend/providers.py`, stdlib urllib like the
  Ollama backend — zero new required deps). `Policy._call` dispatches on
  a callable `chat` attribute: chat backends get un-templated
  `ChatPrompt(system, user, prefill)` parts and the node's own stops;
  the raw path keeps family templates + family stops. One
  `OpenAIChatBackend` covers openai/openrouter/together/deepseek/
  google-compat/lm-studio/custom via a registry; `AnthropicBackend` is
  separate because its native trailing-assistant prefill preserves the
  measured prefill discipline. OpenAI-shaped prefill is a documented
  degraded mode: hint line in the user turn + echo strip on the reply;
  decisions over chat carry `mode: chat` in traces. Supported, not
  recommended — the measured token math remains the local raw path.
- **Voice** (`voice/`, extras `stt`/`tts`/`voice`). Transcripts pass
  three gates, cheapest first: hallucination list and both-direction
  n-gram echo filter (free), then a two-semantic-option pertinence menu
  (~1 token) — the reference's COHERENT/GIBBERISH free-text prompt and
  its persona-specific "fix the text" second call were replaced/dropped.
  TTS pre-synthesizes all sentences, then hard-mutes the mic for
  playback + reverb decay via instance callbacks (the reference's global
  singleton and swallowed callback exceptions were deliberately not
  ported).
- **Camera + relay agents** (extras `camera`/`relay`). `node.images`
  rides `GenOpts.images` (raw base64; each backend adds its own wire
  framing; Ollama uses the `images` field). `look` registers only for
  vision-capable model names; `light` only on ARM Linux with RPi.GPIO —
  both via `optional_agents`, re-derived on `/model`. The light agent
  settles clear phrasings by regex with zero model calls; the reference
  tools.py bug (boolean-negating an "on"/"off" string) was fixed, and
  8-bit PCM decode underflow in the reference TTS was corrected.
