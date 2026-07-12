# ThreeToks — Handoff Summary (updated 2026-07-13; original 2026-07-07)

## 0. Coding-agent v2 update (2026-07-13)

Three research spikes (spikes/e7_code_routing, e8_edit_vertical,
e9_code_retrieval — each with a REPORT.md) were promoted into the tree:

- **Mode routing** (`threetoks/code/route.py`): navigate/edit/compute/
  author, ~70% routed free at ~98% precision, 91.3% end-to-end live.
- **Edit vertical** (`threetoks/code/edit/`): selection-dominated editing
  of existing code — ast symbol index, ranked free-take navigation with
  bounded backtracking, span-only generation, caller-test oracle,
  per-target escalating repair. Live: 10-11/14 planted-bug scenarios
  (`python3 eval/run_edit_eval.py`), successful edits at 3-92% of
  whole-file rewrite cost. Settled boundary: menus read the session log;
  generations get fresh stateless episodes (E5b — never error feedback).
- **Two-anchor contracts + retrieval-as-repair** (`threetoks/code/
  retrieve.py`, `[code] retrieval`, OFF by default): anchors are lists
  (boundary case + normal case); exhausted anchor-bearing methods can be
  repaired from web-retrieved snippets judged purely by subprocess
  execution (E9: 7/8 classics, 0 false accepts). Mojeek added as the
  leading zero-infra search hop.

Full design + evidence: docs/DESIGN-coding-agent.md §9. Deferred
follow-ups (verification-panel advisories, all logged, none blocking):
split `code/edit/vertical.py`'s navigation walk into a Navigator in
nav.py; consolidate the now-4 keyword-overlap ranking implementations
(memory.py, web/target.py, web/vertical.py, code/edit/vertical.py); the
prefix-fuzzy match rule in edit ranking is deterministic but unmeasured;
a real license pipeline + OS-level sandbox before retrieval defaults on;
a compute/REPL vertical proper (compute mode currently reuses
CodeVertical + script execution).

**Known broken on this branch (pre-existing, not from this work): the
web layer is behind its own tests** — `SearchUnavailable`/challenge
detection missing from web/search.py, `MAX_REJECTED_QUERIES`/
`SEARCH_UNAVAILABLE_ANSWER` from web/vertical.py, `strip_quotes` from
web/target.py; 7 tests fail at import/assert (test_web_search,
test_web_vertical, test_target, 4 in test_agents_core). A separate
restoration task was spawned for this; everything else in the suite is
green.

Self-contained briefing for a new instance/agent taking over. Repo:
`/Users/mlg/Documents/A-project/AI/ThreeToks`, git `main`, ~36 commits,
210 tests green (`python3 -m unittest discover -s tests`).

## 1. Mission & thesis

A token-frugal agentic framework that makes a **1.5b local model useful on
8GB hardware**. Core thesis: **classification over generation** — the
harness (deterministic code) walks a tree of decisions and renders each as
a numbered menu; the model answers with ~1 token. Notes are taken by
*pointing* at numbered sentences (verbatim + provenance = hallucination-free
by construction). A ReAct agent spends 200-500 tokens / 15-30s per step on
this hardware; a ThreeToks step is 1-3 tokens / ~0.3s warm.

Reference machine: Apple M1, 8GB (Metal budget ~5.3GB). Ollama 0.6.7 at
localhost:11434. Inspired by agenticSeek (reference clone at `agenticSeek/`,
gitignored; recon in docs/recon-agenticseek.md).

## 2. Experimentally validated facts (Phase 0 — spikes/*/REPORT.md)

- **Policy model = `qwen2.5:1.5b-instruct` for ALL nodes.** E1: 76.7% menu
  accuracy / 0.285s / 3 tok. `deepseek-r1:1.5b` DISQUALIFIED (23-26% all
  variants, escape-biased; also bad at picks in E3). R1 raw-mode think
  suppression (`<think>\n\n</think>\n\n` + prefill) stays supported in the
  backend but non-default. 7b models: unusable on 8GB (0.8 tok/s).
- **Prefill beats constraints**: Ollama JSON-schema output HURTS accuracy
  (-9pts, 3x latency). Format is forced via assistant prefill ("ANSWER:",
  "RELEVANT:") + tiny num_predict + raw-mode templates built by us.
- **KV-cache-first prompts** (E2): append-only prompts keep prefill flat
  (~300-500ms); any mutation near the top forces full re-ingest (2-5x
  slower). Reuse signal = `prompt_eval_duration`, NOT `prompt_eval_count`.
  Keep prompts well under num_ctx=2048.
- **Escape must be an ordinary numbered LAST option** — tiny models almost
  never emit "0" (out-of-distribution). Digit 0 always parses invalid.
- **Retry ladder** (E4): temp-retry useless (3.6%); reshuffle recovers 29%
  (+1 call); rephrase 39% (template bandit still unbuilt — Phase 2 item).
  **Majority voting REJECTED as default**: entrenches consistent errors
  (navigation regressed). `PolicyConfig.vote_k` exists, defaults 1.
- E3: sentence-picking recall 0.85 at ~12 tok vs freetext notes 3.7x cost
  and worse; >40-sentence views crater recall → 25-sentence chunks.

## 3. Architecture (all in `threetoks/`)

Core (orchestrator-written, panel-reviewed):
- `nodes.py` — MenuNode (1 digit, permuted options, numbered escape),
  PickManyNode (comma indices, max_picks, `items=` for trace resolution),
  ShortTextNode (bounded free text). Nodes may carry `.tag` ("judge",
  "curate", "requery") shown in the TUI ticker.
- `render.py` — Episode: append-only prompt = [static prefix | task |
  history log (1-line entries) | observation (replaced at page boundaries)
  | session]. THE cache discipline — never mutate above the observation.
- `policy.py` — retry ladder (temps 0.0/0.0/0.4 = reshuffle-first), option
  permutation per attempt (position-bias defense), optional vote_k,
  bounded-think phase for R1 family, trace events via `trace.py` Tracer
  (`on_event` callback feeds the TUI ticker; pick indices resolved to
  texts in events).
- `engine.py` — run_episode(vertical, policy, max_steps); Vertical protocol
  (episode / next_node / apply / result / steps_left).
- `backend/` — ModelSpec families "r1"/"chatml", raw-prompt builders,
  OllamaBackend (stdlib urllib).
- `config.py` + `config.ini` — [llm] model/host/vote_k, [browser]
  mode=http|plain|stealth + visible, [search], [research], [files].
  Type-safe fallbacks, THREETOKS_CONFIG env override.

Web vertical (`web/`): `search.py` (SearxngProvider with per-query engine
rotation, BingHtmlProvider, DdgHtmlProvider, FallbackProvider chain that
retries on EMPTY results, 5s pacing, bounded cache that never caches empty),
`fetch.py` (HTTP), `browser.py` (plain Selenium), `stealth_browser.py`
(agenticSeek port: undetected-chromedriver + selenium_stealth, one reused
driver, visible option — pulls GitHub READMEs where HTTP gets bot walls),
`textify.py` (HTML→sentences: markdownify, junk deny-list incl. forum
chrome/timestamps/"signed in", glued-camelCase soup filter, table-pipe
cleanup, dedupe), `notes.py` (NoteStore: verbatim + provenance, near-dup
drop, `select()` preserves pick order = ranking), `vertical.py` (the state
machine: results menu → open page → relevance-filtered sentences →
keyword-best entry chunk → AUTO-WALK 3 chunks x 8-pick note passes →
links/read-more/re-search → curation → synthesis; error-page detection;
menu-bleed guard with persistent-single-digit acceptance; quote fallback).

Research loop (`research.py`): deep_research = judged rounds. Deterministic
junk pre-checks (empty/comma-list/note-less) veto free; then a **unanimous
3-lens judge panel** (directness / junk-detection / usefulness — semantic
options, NOT yes/no which was a coin flip). Rejected → propose_query (sees
all tried queries, unique fallback) → new episode over the SAME NoteStore.
Curation: >8 notes → one PICK_MANY ranks best-first (pick order = ranking).
Final answer = bounded generation (80 tok) grounded ONLY in curated notes.

Agents (`agents/`): AgentSpec(name, description, run(task, services,
policy)->dict with "answer"). Router = one MenuNode over agent descriptions
(fallback = first spec = casual). `casual` (canned bank, exact-hint
zero-call shortcut, menu pick, free-text fallback), `web` (deep_research
wrapper), `files` (sandboxed read-only explorer vertical; resolve()+
is_relative_to, symlink-escape verified safe), `code` (writes a small
single-file module). Registry: `agents/__init__.py: default_agents(services)`.
Guide: docs/AGENTS.md.

Code vertical (`code/`): the harness owns the file via `store.py`
(MethodStore -> deterministic module render, always parses; unfilled methods
render as NotImplementedError stubs). `plan.py` (lenient parse of the model's
function list), `gates.py` (free deterministic judge: reconstruct+trim body to
one function, signature/undefined-name checks, subprocess import+assert runs),
`nodes.py` (PlanNode/ImplementNode/TestNode — bounded generations carrying
their own `system`/`temperature`, duck-typed through the policy ladder),
`vertical.py` (state machine: PLAN once -> per method IMPLEMENT -> gates ->
optional test asserts -> resample-repair or stub; `_ensure_entry` guarantees
the goal's named function). Model asserts are advisory (never veto correct
code); trusted example contracts gate. Self-tests off by default in the agent.
Evidence: spikes E5 (per-method 83%/92%) and E6 (whole-module 4/6, 6/6 parse),
design in docs/DESIGN-coding-agent.md.

TUI (`tui.py`, `python3 -m threetoks`): cyberpunk REPL, live ticker (one
line per decision incl. per-pick resolved sentences, judge/curate/requery
tags), wrapped no-trim result panels, `WEB · UNVERIFIED` badge when
judges exhausted, /help /agents /deep N /browser http|plain|stealth
[on-screen] /model /quit. One-shot: `python3 -m threetoks.cli "q" [--rounds N]`.
Eval: `python3 eval/run_eval.py` (10 tasks, scores retrieval vs answer
separately; historical: 6-7/10 answers, ~11s/task).

## 4. Working conventions (keep these)

- User (mlg.fcu@gmail.com) wants the orchestrator as PM/architect: heavy
  thinking + core program flow yourself; delegate implementation to Opus
  subagents with self-contained briefs (scope, style rules ≤20-line
  functions/docstrings/named constants/no TODOs, no-commit contract,
  report format). Load `anthropic-skills:engineering-standards` each
  session. Verification panel (bug hunter/architecture/docs/direction
  personas) before milestones; full-coverage review found real bugs the
  triaged review missed — don't skip it.
- Commits: `type(scope): one line`, no footer, one type per commit; commit
  every green change without asking.
- Tests: unittest, offline with scripted fake backends (see
  tests/test_agents_core.py ScriptedBackend); every behavior change ships
  a test. Modules end with `if __name__ == "__main__":` smokes.
- Memory files live at ~/.claude/.../memory/ (threetoks-framework.md).

## 5. Known issues & environment caveats

- **Search fragility is the #1 practical limiter**: searxng engine
  suspensions now 5min (was 24h) and rotation+Bing+DDG fallbacks exist,
  but heavy automated use gets the residential IP CAPTCHA'd everywhere;
  HTTPS_PROXY env is honored by all requests paths. Stealth browser is the
  reliable fetch path; a BrowserSearchProvider (search THROUGH stealth
  Chrome) is designed-but-unbuilt and would close the class.
- searxng runs via infra/searxng (docker compose, container left running).
  Machine: stale homebrew chromedriver v139 (uc bypasses it); Chrome BETA
  151 installed, not stable.
- Advisory nits (reviewed, logged, unfixed): TUI float() strictness on
  malformed events; eval trace_totals subscripts; files agent offers
  answer on empty folders; note provenance indexes the relevance-filtered
  list, not the raw page.
- Eval numbers vary run-to-run with live search; multi-run eval unbuilt.

## 6. Next-step queue (in rough priority)

1. BrowserSearchProvider — search via stealth Chrome (kills the IP-block
   class). Then re-run the 10-task eval + the user's live questions
   ("what is agenticSeek", pizza-in-Antibes, model comparisons).
2. Digest view — top-K scored sentences page-wide instead of
   filter-then-chunk (removes the 120-sentence cap + forward-only walk).
3. Template-bandit rephrase rung (39% recovery, measured, unbuilt);
   per-node-type prompt phrasings learned from traces.
4. REPL conversation memory (1-line rolling summary prepended per task).
5. WebResearchVertical split (PageCursor/ResultsBoard) before vertical #4;
   llama.cpp backend for GBNF grammar-constrained decoding.
6. Multi-run eval + eval tasks for deep research specifically.
