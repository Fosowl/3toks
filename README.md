# ThreeToks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An extremely token-frugal agentic framework for small local LLMs.
Target: a *useful* agent even on a 1.5b model on 4GB-class hardware, with
sub-second decision steps.

Core idea: the harness walks a **tree of decisions** and renders each one as
a numbered menu; the model answers with **~3 tokens**. Everything
deterministic (parsing, page rendering, notes, undo) is code; the model is
only the policy oracle. Notes are taken by *pointing at numbered sentences*,
never by rewriting content. A ReAct-style agent spends 200-500 tokens and
15-30s per step on this hardware; a ThreeToks menu step spends 1-3 tokens and
~0.3s. Phase 2 adds a small mesh of pluggable agents (routing is itself a
one-token menu decision) and a judged deep-research loop on top of the
same tree-of-menus engine.

- Architecture: [docs/DESIGN.md](docs/DESIGN.md)
- Writing a new agent: [docs/AGENTS.md](docs/AGENTS.md)
- Phase-0 experiments: `spikes/` (E1 menu accuracy, E2 KV-cache latency,
  E3 note-taking quality, E4 reformulation recovery)
- Default policy model: `qwen2.5:1.5b-instruct` via Ollama, for all nodes.
  `deepseek-r1:1.5b` is supported (raw-mode think suppression) but is not
  used by default — see Phase-0 findings below.

## Install

The core is stdlib-only and runs on a bare install; the `web` extra
(requests, beautifulsoup4, markdownify) powers the web-research agent and
`browser` adds Selenium fetching. Without the extras the TUI still starts —
it prints a notice and disables the web agent.

Global executable via [uv](https://docs.astral.sh/uv/):

```sh
uv tool install "threetoks[web] @ /path/to/ThreeToks"   # or [web,browser]
threetoks
```

Development install into the repo's own environment:

```sh
uv pip install -e ".[web,browser]"    # or: pip install -e ".[web,browser]"
```

Configuration resolves as `$THREETOKS_CONFIG` > `./config.ini` (dev runs
from the repo) > the per-user file — `~/.config/threetoks/config.ini` on
Linux/macOS (`$XDG_CONFIG_HOME` honoured), `%APPDATA%\threetoks\config.ini`
on Windows. The first `threetoks` start without any config walks you
through a short setup wizard and writes the per-user file; re-run it any
time with `/setup` inside the TUI, or just edit the file.

## Quickstart

Requires [Ollama](https://ollama.com) running locally with the default model
pulled:

```sh
ollama pull qwen2.5:1.5b-instruct
```

Interactive CLI:

```sh
threetoks
# or just 'threetoks'
python3 -m threetoks
```

One-shot from the command line:

```sh
python3 -m threetoks.cli "When was the Eiffel Tower completed?" [--rounds N]
```

Optional, both improve web research quality:

- **SearXNG** — run a local instance for better search results:
  `./start_search_engine.sh` (see
  [infra/searxng/README.md](infra/searxng/README.md)); auto-detected at
  startup, falls back to scraping DuckDuckGo's HTML endpoint with no setup.
- **Selenium browser** — `/browser on` in the TUI drives a real (visible by
  default) Chrome window instead of plain HTTP fetch, for JS-rendered pages.
  Requires Chrome and `selenium` installed; `/browser off` reverts to plain
  HTTP.

## Agents

`python3 -m threetoks` dispatches every request through a one-token router:
the model is shown a menu of agent names + one-line descriptions and picks
one digit. An invalid or unparseable choice falls back to the first
registered agent (`casual`).

| Agent | Handles | Notes |
|---|---|---|
| `casual` | small talk, quick replies | canned-reply bank ranked by keyword overlap; exact hint match answers with zero model calls; falls through to a bounded free-text reply only when nothing fits |
| `web` | research questions using internet search | thin wrapper over the judged deep-research loop (below); full vertical with search, page reading, link following, notes |
| `files` | questions about local files/folders | explorer sandboxed under `services.files_root`; same menu/notes/answer shape as the web vertical; can run a one-line shell command once a deterministic deny-list and a fresh one-token safety judge both clear it |

Slash commands in the TUI: `/help`, `/agents` (list registered agents),
`/deep N` (set research rounds), `/browser on|off`, `/model NAME` (swap the
policy model), `/quit`.

Adding an agent is one new module plus one line in the registry — see
[docs/AGENTS.md](docs/AGENTS.md) for the contract and a minimal example.

## Deep research

Both the `web` agent and `python3 -m threetoks.cli` run the same judged loop
(`threetoks/research.py`): run a research episode, then judge the answer.
Deterministic pre-checks reject obvious junk (empty, digits-only, or
note-less answers) with zero model calls; anything else goes to the model
as a **semantic** two-option menu ("states the facts asked for" vs. "vague,
off-topic, or not a real answer") rather than an abstract yes/no — E1/E4
found the 1.5b's yes/no confirm was close to a coin flip, while semantically
worded options are a normal menu decision it handles well. If the judge
rejects, the model proposes a new search angle that must differ from every
query already tried, and another round runs over the *same* note store, so
evidence accumulates across rounds instead of resetting. `--rounds N` /
`/deep N` cap how many rounds run.

Honest note: answer quality is bounded by the search provider. Both
SearXNG's upstream engines and the DuckDuckGo HTML fallback rate-limit or
degrade under heavy use; a run that fails to find an answer often reflects
a bad search result page, not a bad decision by the model. Running a local
SearXNG instance (see `infra/searxng/`) reduces but does not eliminate this.

## Architecture

See [docs/DESIGN.md](docs/DESIGN.md) for the full design: decision node
types, the append-only prompt layout, the model/backend strategy, the
escalation ladder for recovering from misclicks, and the repository layout.

## Phase-0 findings

Phase-0 picked the default policy model and the shape of the decision nodes
from measured behavior, not guesswork: `qwen2.5:1.5b-instruct` answers menus
at 76.7% accuracy versus 23-26% for `deepseek-r1:1.5b` (disqualifying it as
policy model), the escape option had to be rendered as an ordinary last
numbered choice rather than digit `0` (tiny models essentially never emit an
out-of-distribution `0`), and a temperature/reshuffle retry ladder recovers
roughly half of first-attempt failures. Full detail and the open questions
this leaves are in [docs/DESIGN.md §11](docs/DESIGN.md) and the individual
`spikes/*/REPORT.md` files.

## Tests

```sh
python3 -m unittest discover -s tests
```

## Eval

```sh
python3 eval/run_eval.py
```

Scores retrieval (did the notes capture the answer) and answer synthesis
separately per task; see `eval/tasks.jsonl` and `eval/results/`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, test conventions,
and the load-bearing design rules to know before touching the core
pipeline. Security issues go to [SECURITY.md](SECURITY.md), not a public
issue. Participation is covered by the [Code of
Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE)
