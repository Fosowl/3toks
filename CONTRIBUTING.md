# Contributing to ThreeToks

ThreeToks is a token-frugal agent framework for tiny local LLMs. The core
design constraint — deterministic harness code, the model as a one-token
policy oracle — is documented in [docs/DESIGN.md](docs/DESIGN.md); read it
before changing anything in the decision/prompt pipeline (`nodes.py`,
`render.py`, `policy.py`, `engine.py`). The rules there come from measured
experiments (`spikes/*/REPORT.md`), not preference — see "Load-bearing
design rules" below before touching them.

## Getting set up

Requires Python 3.10+ and (for anything that talks to the model)
[Ollama](https://ollama.com) running locally:

```sh
ollama pull qwen2.5:1.5b-instruct
```

Dev install, with the optional extras (web search + browser fetching):

```sh
uv pip install -e ".[web,browser]"   # or: pip install -e ".[web,browser]"
```

The core package is stdlib-only; `web`/`browser` extras are imported lazily
so a bare install still runs the TUI (with the web agent disabled). Run the
TUI or a one-shot query to sanity-check your environment:

```sh
python3 -m threetoks
python3 -m threetoks.cli "When was the Eiffel Tower completed?"
```

## Running tests

All tests are offline — no LLM or network calls, scripted fake backends
stand in for the model:

```sh
python3 -m unittest discover -s tests
```

Single test:

```sh
python3 -m unittest tests.test_tui.TickerFormatTest.test_menu_event_shows_kind_value_and_stats
```

Every module that changes behavior ends with an `if __name__ == "__main__":`
smoke block — run it directly after editing:

```sh
python3 -m threetoks.<module>
```

The live eval (`python3 eval/run_eval.py`) needs Ollama and network access
and is not part of the offline test gate; scores vary with search quality,
so don't treat a single run as a regression signal.

## Before you open a PR

- Every behavior change ships an offline test (`ScriptedBackend` in
  `tests/test_agents_core.py` is the pattern for exercising policy-driven
  code without a real model).
- Run the full suite (`python3 -m unittest discover -s tests`) and the
  smoke block of any module you touched.
- Keep functions small and named constants over magic numbers; no dead
  code, no commented-out blocks, no TODOs left behind.
- Don't add abstractions, config flags, or error handling for cases that
  can't happen — see "Conventions" in [CLAUDE.md](CLAUDE.md) for the full
  house style.
- If you're adding a new agent, follow the contract in
  [docs/AGENTS.md](docs/AGENTS.md) and register it in
  `threetoks/agents/__init__.py`.

## Commit style

`type(scope): one-line plain description` — one type per commit, no
footers, no emoji. Keep commits small and green; a commit that breaks the
test suite should not exist on its own.

Examples from this repo's history:

```
fix(tui): honor config visibility in /browser and report failed browser startup
feat(web,research): detect a dead search layer, stop burning rounds, and fall back to model-named sites
docs(design,claude): record the search circuit breaker and site fallback
```

## Load-bearing design rules (do not "fix" without new evidence)

These come from Phase-0/2/3 measurements and are cited in
[docs/DESIGN.md §11–14](docs/DESIGN.md). Changing them without new
measurements will regress accuracy — if you think one is wrong, run a
spike under `spikes/` first and cite the numbers in your PR:

- The escape option is an ordinary numbered **last** menu option; digit
  `0` always parses invalid.
- Menus have ≤9 numbered lines including escape — page, don't widen.
- Judgment decisions are two semantically described options, never bare
  yes/no.
- Prompts are append-only; never mutate anything above the current
  observation.
- Format is forced by prefill + token caps, not structured output.
- Notes are verbatim pointers into source sentences, never rewritten.
- Deterministic pre-checks run before any model call.
- Majority voting is not a default; `PolicyConfig.vote_k` is for
  irreversible actions only.
- Every multi-step agent has a step budget and a forced-answer trigger.

## Where to look first

- [docs/DESIGN.md](docs/DESIGN.md) — full architecture, prompt layout,
  Phase 0–3 experiment results.
- [docs/AGENTS.md](docs/AGENTS.md) — the agent contract, node types, a
  minimal example agent.
- [docs/DESIGN-coding-agent.md](docs/DESIGN-coding-agent.md) — the code
  vertical (model-written modules, deterministic gating, blind repair).
- [docs/HANDOFF.md](docs/HANDOFF.md) — current state, known issues,
  prioritized next-step queue.

## Reporting bugs / requesting features

Open an issue with: what you ran, what you expected, what happened, and
(if relevant) a `--trace out.jsonl` capture from `threetoks.cli` — the
trace records every decision node, option set, and model output, which is
usually enough to diagnose a policy/prompt problem without reproducing it
live.

## Security issues

Do not open a public issue for a security vulnerability — see
[SECURITY.md](SECURITY.md).
