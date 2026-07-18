# ThreeToks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An extremely token-frugal agentic framework for small local LLMs.
Target: a *useful* agent even on a 1.5b model on <4GB-class hardware, with
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
- Default policy model: `qwen3.5:2b` via Ollama, for all nodes.
  `deepseek-r1:1.5b` is supported (raw-mode think suppression) but is not
  used by default — see Phase-0 findings below.

## Install

The core is stdlib-only and runs on a bare install. Every feature beyond
that is an optional extra: install the package, get the capability; skip
it, and the TUI still starts and simply leaves that feature out.

| Extra | Enables | Packages |
|---|---|---|
| `web` | web-research agent | requests, beautifulsoup4, markdownify |
| `browser` | Selenium page fetching (`/browser`) | selenium |
| `stt` | voice input: microphone speech-to-text | vosk, sounddevice |
| `tts` | voice output: spoken answers | piper-tts, sounddevice, numpy |
| `voice` | both `stt` and `tts` | — |
| `camera` | `look` agent (webcam + vision model) | opencv-python |
| `relay` | `light` agent (GPIO lamp, Raspberry Pi only) | RPi.GPIO (ARM only) |

Global executable via [uv](https://docs.astral.sh/uv/):

```sh
uv tool install "3toks[web] @ /path/to/ThreeToks"   # pick your extras
threetoks
```

Development install into the repo's own environment:

```sh
uv pip install -e ".[web,browser]"          # research setup
uv pip install -e ".[web,voice,camera]"     # voice assistant setup
uv pip install -e ".[web,voice,relay]"      # Raspberry Pi setup
```

Configuration resolves as `$THREETOKS_CONFIG` > `./config.ini` (dev runs
from the repo) > the per-user file — `~/.config/threetoks/config.ini` on
Linux/macOS (`$XDG_CONFIG_HOME` honoured), `%APPDATA%\threetoks\config.ini`
on Windows. The first `threetoks` start without any config walks you
through a short setup wizard and writes the per-user file; re-run it any
time with `/setup` inside the TUI, or just edit the file.

## Quickstart

The default setup requires [Ollama](https://ollama.com) running locally
with the default model pulled (a cloud provider instead of Ollama works
too — see [LLM providers](#llm-providers)):

```sh
ollama pull qwen3.5:2b
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
| `light` | "turn on the light", lamp control | only registered on a Raspberry Pi with the `relay` extra; clear phrasings are settled by a regex with zero model calls, anything else spends one menu decision |
| `look` | "what am I holding?", camera questions | only registered with the `camera` extra AND a vision-capable model (llava, qwen-vl, gpt-4o, claude, gemini, ...); one frame, one bounded vision decision |

Slash commands in the TUI: `/help`, `/agents` (list registered agents),
`/deep N` (set research rounds), `/browser http|plain|stealth`,
`/model NAME` (swap the policy model), `/voice on|off|debug` (talk instead
of typing; `debug` traces what the microphone hears and why each gate
dropped it), `/debug [on|off]` (show the raw model answer and finish
reason under each decision), `/setup` (re-run the config wizard), `/quit`.

Adding an agent is one new module plus one line in the registry — see
[docs/AGENTS.md](docs/AGENTS.md) for the contract and a minimal example.

## LLM providers

Ollama (local, raw mode) is the measured default and needs no keys. Any
chat API can drive the same decision tree instead — set `[llm] provider`
in `config.ini` and export the provider's API key:

| Provider | `provider =` | API key env var |
|---|---|---|
| Ollama (default) | `ollama` | — |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Together | `together` | `TOGETHER_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Google (Gemini) | `google` | `GOOGLE_API_KEY` |
| LM Studio | `lm-studio` | `LLM_API_KEY` (optional) |
| any OpenAI-shaped URL | `custom` + `api_base` | `LLM_API_KEY` (optional) |

```ini
[llm]
provider = openrouter
model = qwen/qwen-2.5-7b-instruct
```

Keys are read from the environment only, never from config files. The
transports are stdlib-only — no SDKs to install. Chat providers are
**supported, not recommended**: raw mode pre-seeds the assistant's reply
(`ANSWER:`), which is what makes 1-token menu decisions reliable on tiny
models. A chat API can't do that (Anthropic is the exception — its native
prefill is used), so those decisions run in a degraded mode that leans on
instruction-following and strips any echoed prefill; bigger cloud models
handle this fine, but the token math and the measured accuracy numbers in
this README all describe the local raw-mode path. Decisions made over a
chat transport are tagged `mode: chat` in traces. When a chat provider
answers with unparseable decisions (every ticker line renders `—`),
`/debug on` shows each raw completion and its finish reason — `length`
means the node's token cap truncated the reply before the model answered
(the caps are tuned for raw-mode prefill, so a chatty or reasoning model
spends them on preamble).

## Voice, camera, and relay (optional)

Each feature activates purely by installing its extra — no code changes.
All configuration lives in the `[voice]`, `[camera]`, `[relay]` sections
of `config.ini` (written with commented defaults by the setup wizard).

**Voice input — `3toks[stt]`** (vosk + sounddevice). `/voice on` in the
TUI listens on the microphone. Every transcript passes three gates before
reaching the agents: a hallucination list (junk Vosk invents from
silence), an echo filter (n-gram match against the assistant's last
spoken answers, so it never talks to itself), and a one-token pertinence
menu ("meant for the assistant" vs "background noise") decided by the
policy model — cost: 1 token per utterance that survives the free gates.
The vosk model auto-downloads for `[voice] lang` (default `en-us`); set
`stt_model_path` to use a local one. `/voice debug` traces each raw
transcript as it arrives, why a gate dropped it (empty, hallucination,
echo), and whether the pertinence menu judged it directed at you or noise
— and if nothing at all appears in debug mode while you speak, check your
OS microphone permission for the terminal you run 3toks in.

**Voice output — `3toks[tts]`** (piper-tts + sounddevice + numpy). Spoken
answers via a local Piper voice. Voice models are NOT auto-downloaded:
grab a `.onnx` + `.onnx.json` pair from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) and
point `[voice] tts_model_path` at the `.onnx`. The microphone is
hard-muted during playback plus `post_speak_delay` seconds of reverb
decay. Install both sides at once with `3toks[voice]`; either side alone
also works (voice-in/text-out or type-in/spoken-out). Set
`[voice] enabled = true` to start the TUI already listening. `/setup`
includes an optional voice step that walks the two-file piper download
interactively — it spells out that you need the `.onnx` *and* its
`.onnx.json` sidecar side by side, offers to open the vosk/piper model
pages, and re-asks any path that fails its check until you have a working
one (or keep yours). `/voice on` re-reads the saved config as it starts,
so a voice setup applies without restarting the TUI — if voice is already
running, `/voice off` first.

**Camera — `3toks[camera]`** (opencv-python). Registers the `look` agent
when the configured model is vision-capable. One JPEG frame (device
`[camera] index`) rides along with a single bounded decision — note that
an image costs far more tokens than a menu decision, so point this at a
vision model deliberately, e.g. `ollama pull llava` + `/model llava`.

**Relay — `3toks[relay]`** (RPi.GPIO, installs only on ARM). On a
Raspberry Pi, registers the `light` agent driving an active-low relay
board on BCM pin `[relay] pin` (default 17). On any other machine the
extra installs nothing and the agent stays unregistered. The light agent
is intentionally minimal — treat it as the **template for your own Pi
hardware tools**; the step-by-step recipe (hardware module, agent,
registration, packaging, tests) is in
[docs/AGENTS.md](docs/AGENTS.md#hardware-agents-raspberry-pi-the-light-agent-is-a-template).

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
from measured behavior, not guesswork: `qwen3.5:2b` answers menus
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
