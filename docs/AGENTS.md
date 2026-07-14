# Writing a ThreeToks agent

An agent is a named capability the router can dispatch a user request to.
This is the contract, the shapes available for building one, a minimal
example, and the rules an agent must follow to stay consistent with the
rest of the harness.

## The `AgentSpec` contract

`threetoks/agents/base.py`:

```python
@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    run: Callable[[str, Services, Policy], dict]
```

- `name` — short, unique, lowercase (`casual`, `web`, `files`). Shown in
  `/agents` and used as the router's fallback key (see below).
- `description` — one line, shown to the router model as the menu option
  text (`"{name} — {description}"`). Keep it short and disambiguating; the
  router is a single one-token `MenuNode` decision, so overlapping
  descriptions cost accuracy.
- `run(task, services, policy) -> dict` — does the work.

## `run()` signature and result dict

```python
def run(task: str, services: Services, policy: Policy) -> dict:
    ...
```

- `task: str` — the raw user request text.
- `services: Services` (`threetoks/services.py`) — shared capabilities:
  `provider` (search), `fetch_page` (url -> `PageText`), `files_root`,
  `max_research_rounds`. Take what you need, ignore the rest; do not assume
  every field is populated for every agent (e.g. `files` never touches
  `provider`).
- `policy: Policy` (`threetoks/policy.py`) — the only way to consult the
  model. Call `policy.decide(episode, node)`; never call a backend directly.

Required result key:

- `"answer": str` — the final answer text shown to the user.

Optional keys (fill in what applies; the TUI result panel and tests tolerate
missing ones):

- `"notes": str` — rendered note text (`NoteStore.render()`), shown dim
  under the answer as sources.
- `"agent": str` — your agent's name; set it in `run()` (or via a thin
  wrapper — see `threetoks/agents/web.py`) so the TUI panel labels itself
  correctly. Not automatic.
- `"rounds": int` — round count, shown in the stats footer when present.
- Anything else your agent wants to report (e.g. `files.py` adds
  `"files_opened"`) — extra keys are ignored by callers that don't know
  them.

## Node types available

Everything the model sees is one of the node types in `threetoks/nodes.py`.
An agent that does a single decision can build these directly; an agent
that runs a multi-step exploration (like `files` or `web`) drives them
through a `Vertical` (see `threetoks/engine.py`) so the harness's step loop,
budgets, and escape handling are shared instead of reimplemented.

| Node | Model output | Use it for |
|---|---|---|
| `MenuNode(question, options, escape=True)` | one digit | any single-choice decision: "what next", judge verdicts, routing |
| `PickManyNode(question, n_items, max_picks=4)` | comma-separated indices | "which numbered sentences/lines answer this" — the note-taking step |
| `ShortTextNode(question, prefill, max_tokens=24)` | bounded free text | search queries, final free-text answers |
| `confirm_node(question)` | `MenuNode` with `["yes", "no"]`, no escape | rarely — see the design-rules note on judge phrasing below |

Construct a node, then call `policy.decide(episode, node)` to get back a
`Decision(kind, value, raw_text, valid)`. Always handle `decision.valid ==
False` — treat it the same as an explicit escape, never crash or retry
silently forever.

## A minimal example agent

An agent that answers with the current line count of a fixed string,
via one menu decision (yes/no framed semantically, not as raw "yes"/"no"
per the design rule below) — enough structure to copy for a real one-shot
agent:

```python
"""Example agent: counts words in the task text after one confirmation."""
from threetoks.agents.base import AgentSpec
from threetoks.nodes import MenuNode
from threetoks.render import Episode

AGENT_NAME = "wordcount"
AGENT_DESCRIPTION = "counts words in your message"

PREFIX = """You confirm a simple action.
Example:
ACTIONS:
1 = go ahead and count the words
2 = do not count, just say hello instead
Reply with exactly ONE digit.
ANSWER: 1"""

OPT_COUNT = "go ahead and count the words"
OPT_SKIP = "do not count, just say hello instead"


def run(task: str, services, policy) -> dict:
    episode = Episode(PREFIX, task)
    episode.open_observation(f"MESSAGE:\n{task}")
    node = MenuNode("What should I do?", [OPT_COUNT, OPT_SKIP], escape=False)
    decision = policy.decide(episode, node)
    if decision.valid and decision.value == OPT_COUNT:
        answer = f"{len(task.split())} words"
    else:
        answer = "hello!"
    return {"answer": answer, "agent": AGENT_NAME}


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)
```

For anything with more than one decision, model the state machine as a
`Vertical` (`episode`, `next_node()`, `apply(node, decision)`, `result()`)
and drive it with `threetoks.engine.run_episode(vertical, policy,
max_steps=...)` — see `threetoks/agents/files.py` for the smallest real
example and `threetoks/web/vertical.py` for the fullest one.

## Registry wiring

`threetoks/agents/__init__.py`:

```python
from threetoks.agents import casual, code, files, web

def default_agents(services: Services) -> list[AgentSpec]:
    return [casual.SPEC, web.SPEC, files.SPEC, code.SPEC]
```

To add an agent: write the module exposing a module-level `SPEC =
AgentSpec(...)`, import it, and add `yourmodule.SPEC` to the list returned
by `default_agents`. Order matters only in one respect: **the first spec in
the list is the router's fallback** when the model's choice is invalid or
unparseable (`threetoks/agents/router.py`), so keep a cheap, safe default
(`casual`) first. Names must stay unique — the registry test enforces this.

An agent may itself route further: the `code` agent front-doors four modes
(navigate / edit / compute / author) through `threetoks/code/route.py` —
deterministic pre-checks first, one one-token menu otherwise (measured in
spike E7; see docs/DESIGN-coding-agent.md §9a). It also reads
`services.retriever` (the opt-in retrieval-as-repair hook, None unless
`[code] retrieval` is enabled and a search provider exists) and
`services.files_root` (the corpus the navigate/edit modes operate on).

## Design rules an agent must follow

These are load-bearing, not stylistic — Phase-0/Phase-2 measurements are
the reason for each one:

1. **Menus have at most 9 numbered lines, including the escape.**
   `MenuNode` enforces this (`MENU_MAX_OPTIONS = 9`) and raises if you try
   to exceed it. If a domain has more than 8 real options, page them —
   don't widen the menu.
2. **The escape is a numbered LAST option, never digit `0`.** Tiny models
   essentially never emit an out-of-distribution `0` (E4); an escape
   rendered as `0` is unreachable in practice. Use `MenuNode(...,
   escape=True)` (the default) unless the decision genuinely has no
   "none of these" case (e.g. a judge's good/bad verdict, which uses
   `escape=False` because both options are exhaustive).
3. **Notes are verbatim, never rewritten.** Take notes by having the model
   `PickManyNode` over numbered sentences/lines the harness already
   rendered, then copy the chosen text into a `NoteStore` byte-for-byte.
   Never ask the model to summarize or restate source content — that
   reintroduces hallucination risk the numbered-pointer design exists to
   remove.
4. **Answers are grounded generation over curated notes.** When notes
   exist, rank them down with a curation `PickManyNode` (pick order =
   ranking), then run ONE short `ShortTextNode` generation with the notes
   on screen that answers the task in plain words — the web and files
   verticals share this shape. Never dump picked notes verbatim as the
   default answer (a live files run answered "what are these files?" with
   naked code lines), and never generate without the notes visible:
   ungrounded synthesis garbles facts the notes already hold (eval data).
   If the generation keeps bleeding menu digits, fall back to quoting the
   task-closest notes (`rank_by_overlap`) — the fallback stays aimed at
   the goal, not at store order.
5. **Budgets and guards are mandatory, not optional.** Every multi-step
   agent needs: a step budget (`max_steps` passed to `run_episode`, or an
   equivalent `steps_left` counter), a forced-answer trigger a few steps
   before the budget runs out (see `FORCE_ANSWER_AT_STEPS_LEFT` in
   `files.py` / `vertical.py`), and a cap on expensive sub-actions (pages
   opened, files opened, searches run). A model that never terminates a
   loop is a bug in the vertical, not something to patch with a bigger
   step budget.
6. **Judge / confirm decisions should be phrased semantically, not as
   abstract yes/no.** `research.py`'s judge uses two option texts describing
   what "good" and "bad" mean for the task, rather than a bare
   `confirm_node`, because the 1.5b's plain yes/no accuracy was close to a
   coin flip in practice. Prefer descriptive option pairs over
   `confirm_node` whenever the model needs to make a judgment call (not a
   purely mechanical yes/no like "keep going?").
7. **Deterministic pre-checks before spending a model call.** If code alone
   can tell an answer is junk (empty, a bare number, no supporting notes),
   reject it before asking the model to judge — see
   `research._obviously_bad`. Every model call has a latency and error-rate
   cost; don't spend one where a regex or a length check will do.
