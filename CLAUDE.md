# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ThreeToks is a token-frugal agent framework for tiny local LLMs (settled policy model: `qwen2.5:1.5b-instruct` via Ollama, target hardware 8GB M1). The harness — deterministic Python, zero-dependency core — walks a tree of decisions and renders each as a numbered menu; the model answers with ~1 token. The model is only a policy oracle. Everything deterministic (parsing, page rendering, note storage, judging code output) is harness code, never a model call.

## Commands

- Interactive TUI: `python3 -m threetoks` (needs Ollama running and `ollama pull qwen2.5:1.5b-instruct`)
- One-shot research: `python3 -m threetoks.cli "question" [--rounds N] [--trace out.jsonl]`
- All tests — offline, no LLM or network needed: `python3 -m unittest discover -s tests`
- Single test: `python3 -m unittest tests.test_tui.TickerFormatTest.test_menu_event_shows_kind_value_and_stats` (pytest works too: `python3 -m pytest tests/test_tui.py -q`)
- Module smoke check: every module ends with an `if __name__ == "__main__":` block — run `python3 -m threetoks.<module>` after editing it
- Eval (live: needs Ollama + network, results vary with search quality): `python3 eval/run_eval.py`
- Config: resolved `$THREETOKS_CONFIG` > `./config.ini` (repo root, dev) > per-user file (`~/.config/threetoks/config.ini`, `%APPDATA%\threetoks` on Windows) — sections [llm], [browser], [search], [research], [files], [memory]; bad values silently fall back to defaults. First TUI start with no config runs the step-by-step wizard in `threetoks/onboard.py` (re-run with `/setup`), which saves via `config.save_config`
- No linter is configured. Core imports are stdlib-only; `web`/`browser` extras (requests, bs4, markdownify, selenium) are optional and imported lazily — `tests/test_lazy_extras.py` enforces this, and a bare install runs the TUI with the web agent disabled
- Install: `uv tool install "threetoks[web,browser] @ ."` gives a global `threetoks` executable (entry point in pyproject.toml); dev setup is `uv pip install -e ".[web,browser]"`

## Architecture

### One model call (the decision pipeline)

`nodes.py` defines the node types: `MenuNode` (one digit), `PickManyNode` (comma-separated indices, `items=` enables trace resolution), `ShortTextNode` (bounded free text). `render.py`'s `Episode` holds the append-only prompt — `[static prefix | task | history log | observation | session]` — where only the tail changes between steps so Ollama's KV cache covers the rest; `open_observation()` is the only sanctioned point where earlier content changes. `policy.py`'s `Policy.decide(episode, node)` runs the retry ladder (temperatures 0.0/0.0/0.4 — attempt 2 is a pure option reshuffle, which is what actually recovers failures), permutes menu options per attempt to defeat position bias, forces format via assistant prefill (`ANSWER:`, `RELEVANT:`) plus tight `num_predict` caps built by `backend/` raw-prompt templates (ChatML and R1 families), and traces every attempt through `trace.py` (the `Tracer.on_event` callback is what feeds the TUI ticker). Agents never touch the backend directly — always `policy.decide()`.

### Multi-step episodes

`engine.py`: the `Vertical` protocol (`episode` / `next_node()` / `apply(node, decision)` / `result()` / `steps_left`) driven by `run_episode(vertical, policy, max_steps)`. Verticals must handle `decision.valid == False` (treat as escape) and force an answer a few steps before the budget runs out.

### The agent mesh

`agents/base.py`: `AgentSpec(name, description, run(task, services, policy) -> dict)` — the dict needs at least `"answer"`; `"notes"`, `"agent"`, `"rounds"` are optional. `agents/router.py` routes with a single one-token menu over agent descriptions (few-shot prefix in `ROUTER_PREFIX`); an invalid pick falls back to the **first** registered spec, so `casual` stays first in `agents/__init__.py:default_agents()`. `services.py` carries shared capabilities (search provider, `fetch_page`, `files_root`, `max_research_rounds`, recalled memories). docs/AGENTS.md is the full how-to with a copyable minimal agent.

### Verticals

- `web/`: `search.py` (flat provider chain SearXNG → Bing HTML → DDG HTML behind one `FallbackProvider(*hops)`; falls back on *empty results*, not just request failure; 5s pacing; never caches empty; a blocked hop — SearXNG "Suspended:" boxes, Bing challenge shell, DDG anomaly page — raises `SearchUnavailable`, and the provider tracks `consecutive_failures`/`last_call_failed` so callers can tell "no hits" from "layer down"; Bing `ck/a` redirect hrefs are base64-decoded to real URLs), `textify.py` (HTML → numbered sentences with junk filtering), `notes.py` (`NoteStore`: verbatim quotes + provenance; pick order = ranking), `vertical.py` (the research state machine), `browser.py`/`stealth_browser.py` (Selenium fetchers behind `/browser plain|stealth`; stealth is the reliable path past bot walls).
- `research.py`: `deep_research` = judged rounds. Deterministic junk pre-checks reject free of charge; ambiguous answers go to a unanimous 3-lens judge panel of semantic two-option menus. A rejected round refreshes a `TargetBelief` (`web/target.py`: taxonomy menu + verbatim anchor note; settles after two matching picks, then costs nothing) and requeries via a menu of harness-composed strategy queries — tried and paraphrase strings are struck out in code (`too_similar`), with free text kept as an option. The next round adopts the vertical's **curated** NoteStore and opens its episode with the `TARGET SO FAR:` and `QUERIES ALREADY TRIED` lines, so evidence accumulates without junk piling up (DESIGN.md §15). Searches are strictly deduplicated — a normalized-query set (typographic quotes stripped too) and a seen-URL set are shared across all rounds; rejected duplicates/paraphrases cost no search budget but burn a 2-strike rejection budget; the written requery runs at `QUERY_TEMPERATURE` with a rotating angle hint so it cannot re-propose itself (DESIGN.md §17). Search-layer circuit breaker (DESIGN.md §18): when every provider hop fails, one `SITES:` node asks the model to name websites to fetch directly (LLM + fetch as last-resort search); after `SEARCH_ABORT_FAILURES` all-hops-failed searches with zero notes, the loop returns the honest `(search unavailable: …)` answer instead of burning rounds; in-round dead ends (nothing to open, search, or answer from) end without synthesis or judging, and results menus with one real option are taken by the harness for free.
- `code/`: the harness owns the file — `store.py`'s MethodStore renders the module deterministically (always parses; unfilled methods become `NotImplementedError` stubs). `gates.py` provides the free deterministic judge (ast, signature, undefined names, subprocess runs); its placeholder gate also rejects return-shaped cheats (`return NotImplementedError` / bare `return` / `return None` / `return ...`). Model-written self-tests are advisory, never gating (~25% encode wrong expectations); repair is blind resample at higher temperature, not error-feedback. Once all methods are terminal, `runner.py` executes the module for free (trusted examples + zero-arg smoke calls, one subprocess each; `input()` EOF counts as inconclusive, not broken), maps each failure's traceback frames onto method spans to blame the offending method, and the vertical unlocks the blamed method for another blind implement cycle — capped at 2 runner rounds and 2 repairs per method. The delivered answer is `render_script`: the module plus a harness-rendered `if __name__ == "__main__":` guard calling the deterministic entry (bodied zero-arg `main` > goal-named entry > sole zero-arg function; omitted when nothing qualifies) — gates and the runner always use the guard-free `render_module` (DESIGN-coding-agent.md §7b).
- `agents/files.py`: the file explorer mirrors the web vertical (menus, verbatim notes, curate→synthesise answer) under the `services.files_root` sandbox. Its shell option runs a model-written one-liner only past two gates: a deterministic deny-list (`dangerous_command`: metacharacters, write/exec/network words, sandbox-escaping paths — free) and `command_is_safe`, a fresh-context one-digit judge (two semantic options, fail-closed) that sees only the command — never file contents, so injected text can't lobby it. Approved output is numbered and gets a normal note pass; `MAX_SHELL_RUNS` proposals per episode.
- `memory.py`: cross-session per-agent (query, answer) store dumped to JSON at session end — deduped per query, capped per agent, gated by `worth_remembering` (no empty / `(no answer)` / judge-rejected answers). After routing, a free keyword shortlist (`MemoryStore.relevant`; no overlap → no model call) feeds one `PickManyNode` that picks up to 3 entries into `services.recalled`; `render_recall` turns the picks into an `EARLIER ANSWERS` block that `deep_research` logs into every round's episode (DESIGN.md §16).

### TUI

`tui.py` (entry: `python3 -m threetoks`): neon-gradient REPL streaming one ticker line per decision. Every task ends with a dim task-report footer below the result panel — wall time, the model's share of it, and per-call token/latency averages (plus a retry count when attempts reshuffled). Pure formatting/dispatch functions are deliberately separate from the input loop so the whole thing tests offline (`tests/test_tui.py`); `NO_COLOR` or non-tty output falls back to plain text.

## Load-bearing design rules (measured — do not "fix")

Each of these comes from Phase-0/2/3 measurements (`spikes/*/REPORT.md`, docs/DESIGN.md §11–14); changing them regresses accuracy:

- The escape option is an ordinary numbered **last** option; digit `0` always parses invalid (tiny models never emit an out-of-distribution 0).
- Menus have ≤9 numbered lines including the escape; `MenuNode` raises beyond that. Page, don't widen.
- Judgment decisions are two semantically described options, never bare yes/no — the 1.5b's yes/no accuracy was a coin flip.
- Prompts are append-only; never mutate anything above the current observation (KV-cache discipline, E2).
- Format is forced by prefill + token caps. Ollama JSON-schema output measured *worse* (-9pts, 3× latency) — don't switch to structured output.
- Notes are verbatim: the model points at numbered sentences; it never rewrites source content. Final answers are one short generation grounded ONLY in the curated notes (harvest→curate→synthesise — web and files verticals alike); persistent menu-digit bleed falls back to quoting the task-closest notes verbatim (`rank_by_overlap`).
- Run deterministic pre-checks before spending any model call (see `research._obviously_bad`).
- Majority voting is rejected as a default (it entrenches consistent errors); `PolicyConfig.vote_k` exists for irreversible actions only.
- Step budgets, forced-answer triggers, and caps on expensive sub-actions are mandatory in every multi-step agent.
- The policy model is settled: `qwen2.5:1.5b-instruct` for all nodes. `deepseek-r1:1.5b` is disqualified for decisions (23–26% menu accuracy); its raw-mode think suppression stays supported in the backend only.

## Conventions

- Commits: `type(scope): one-line plain description`, one type per commit, no footers, no emoji. Commit every green change without asking.
- Every behavior change ships an offline test. Tests use scripted fake backends (see `ScriptedBackend` in `tests/test_agents_core.py`) — never a live model or network.
- Every module ends with an `if __name__ == "__main__":` smoke block; run it after editing the module.
- Known practical limiter: live search quality. A research run that fails usually reflects a rate-limited or junk results page, not a bad model decision — check the search layer before blaming the policy.

## Docs map

- `docs/DESIGN.md` — full design, token math, prompt layout, Phase 0–3 results and the reasoning behind every settled decision
- `docs/AGENTS.md` — agent contract, node usage, registry wiring, the seven design rules with their evidence
- `docs/DESIGN-coding-agent.md` — coding-vertical design and E5/E6 evidence
- `docs/HANDOFF.md` — current state, known issues, environment caveats, prioritized next-step queue
- `spikes/*/REPORT.md` — the raw Phase-0 experiment reports the design rules cite
