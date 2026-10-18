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

- `task: str` — the raw user request text. When the planner runs your
  agent as one step of a trajectory (see below), the task is only that
  step's short purpose (e.g. `find the population of Japan`), so search
  queries and keyword rules see a clean request. The overall goal and the
  earlier step answers reach every model call of the step through the
  policy (`policy.context`, added after the system block), not the task.
- `services: Services` (`threetoks/services.py`) — shared capabilities:
  `provider` (search), `fetch_page` (url -> `PageText`), `files_root`,
  `max_research_rounds`, `on_progress` (optional `line -> None` progress
  printer). Take what you need, ignore the rest; do not assume every field
  is populated for every agent (e.g. `files` never touches `provider`).
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
| `SequenceNode(question, n_items, max_len)` | numbers in order, repeats allowed | an ordered plan, e.g. the planner's agent order |
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

Agents gated on an optional capability register through
`optional_agents(services, model)` instead: `light` appears only when a
GPIO relay is live on `services.relay` (Raspberry Pi `relay` extra) and
`look` only when `services.capture_frame` is set (camera extra) AND the
model name passes `is_vision_model` (a name-mark heuristic in
`threetoks/backend/base.py` — extend `_VISION_MODEL_MARKS` when new
vision families appear). List any such agent's name in
`OPTIONAL_AGENT_NAMES` so `/model` can re-derive the set at runtime.
Import the agent module lazily inside `optional_agents` — the registry
must import on a bare install.

The TUI then calls `with_planner(specs)` on the final list. It appends the
`planner` agent last, bound to exactly those agents, so a plan never names
an agent that is not ready. With fewer than `MIN_PLANNED_AGENTS = 2`
agents there is nothing to plan, and no planner is added. Do not add
`planner.SPEC` to `default_agents`: it has no module-level `SPEC`. After
`/model` re-derives the optional agents, the planner is rebuilt the same
way, so `/agents` lists it last.

## The planner and trajectories

The router picks `planner` (`threetoks/agents/planner.py`) for a request
that needs several steps done by different agents. The plan is a
**trajectory** (`threetoks/trajectory.py`): an ordered list of steps, each
with one agent and one very short purpose sentence.

1. **Draft.** The model sees the other agents with a number each and gives
   their numbers in order (one `SequenceNode`, e.g. `PLAN: 2,4`; an agent
   can come back). Then one `ShortTextNode` per step gives the purpose
   (`STEP 1 (web): find the population of Japan`). An invalid purpose
   falls back to the full request.
2. **Follow.** Each agent runs with its purpose as the task and with
   `policy.with_context(trajectory.context())`: every model call of the
   step then carries `OVERALL GOAL: ...` and (after the first step)
   `EARLIER STEPS:` with one clipped answer per done step, also for nodes
   that bring their own system block.
3. **Correct.** After each step, one two-option menu asks if the plan needs
   an update. The options describe what yes and no mean (see the design
   rule on judge phrasing). On yes, the remaining steps are drafted again,
   with the done steps shown; an unparseable new draft keeps the old steps.

Caps: `MAX_PLAN_STEPS = 4` per draft (the reply rule says "at most 4";
extra numbers are dropped), `MAX_REPLANS = 2` update answers of "yes"
(counted even when the new draft fails to parse), and
`MAX_TRAJECTORY_STEPS = 6` per request. When a cap is reached the update
question is no longer asked; steps cut by the step cap are reported as
`step cap reached › N step(s) not run`. If the first draft fails to
parse, the planner asks the one-agent router instead and returns that
agent's result unchanged (no `trajectory` key). Otherwise the result has
only `agent` (`planner`), `answer` (the step answer itself for a one-step
plan, or every step answer under a numbered `N. agent — purpose` title),
`trajectory` (the done steps as dicts: `agent`, `purpose`, `answer`), and
`judged_good: False` when any step result had it. Progress lines
(`plan ›`, `plan updated ›`, `no plan › routed to X`, `step cap reached
› …`) go to `services.on_progress`; the TUI shows them as `⌁` rows in
the progress tree. Each step runs inside a `step N › agent: purpose`
span, so its decisions nest under it and its end row shows its status
(`answered`, `no answer`, or `unverified`).

### Showing an agent's parts in the progress tree

The TUI draws each task as a tree (`threetoks/progress.py`). Policy
decisions are rows. A plain menu decision (a `MenuNode` without a `tag`)
is an action: the decisions after it nest under it until the next plain
menu decision. So give a `tag` to every menu that is a check or a
classification (`judge`, `mode`, `replan`), and leave the menus that
choose the agent's next step untagged. To show a larger part of your
agent as its own branch, wrap it in a span on the policy tracer:

```python
from threetoks.trace import span

with span(getattr(policy, "tracer", None), "round 1 · search and read") as outcome:
    ...  # decisions here nest under the branch
    outcome["result"] = "accepted"  # optional; shown on the ╰ end row
```

An exception inside the block ends the branch with the result `stopped`,
unless the block already set a result. A `None` tracer records nothing,
so test doubles without a tracer work.

## Hardware agents (Raspberry Pi): the light agent is a template

The `light` agent is deliberately minimal — it exists as the worked
example for wiring ANY physical tool (fan, servo, sensor, second relay)
into the mesh. The recipe, with `threetoks/relay.py` +
`threetoks/agents/light.py` as the reference at every step:

1. **Hardware module** (`threetoks/yourtool.py`): one class owning the
   device. Import the hardware library lazily inside `__init__` and
   accept an injectable stand-in for tests (see `Relay(pin, gpio=None)`).
   Action methods return short spoken strings — never booleans dressed
   as strings. Add a pure availability gate like `relay_available()`
   (platform + `find_spec` checks, zero import side effects).
2. **Agent module** (`threetoks/agents/yourtool.py`): a module-level
   `SPEC = AgentSpec(name, one-line description, run)`. Inside
   `run(task, services, policy)`: a guard clause answers when the device
   is absent; a word-boundary regex settles clear phrasings with ZERO
   model calls (status-style questions must never drive the hardware);
   anything else spends exactly ONE MenuNode of semantically described
   options, escape last. Answers are one spoken-length sentence — voice
   mode reads them aloud.
3. **Services field**: add `yourtool: object = None` to
   `threetoks/services.py`; construct it in `session.build_state` behind
   the availability gate, degrading to `None` with a notice on failure
   (see `session.make_relay`). If a config section drives it, add an
   applier to `live_config._APPLIERS` so a saved change applies live.
4. **Registration**: extend `optional_agents()` (lazy import) and add
   the agent's name to `OPTIONAL_AGENT_NAMES`.
5. **Packaging**: a new extra in `pyproject.toml`; if the dependency is
   Pi-only, gate it on architecture the way `relay` does
   (`"RPi.GPIO; platform_machine == 'aarch64' or ..."`).
6. **Tests + smoke**: offline only — a fake device recording calls
   (`tests/test_relay.py`), agent tests asserting the regex paths never
   consult the backend (`tests/test_light_agent.py`), your module listed
   in `tests/test_lazy_extras.py`, and an `if __name__ == "__main__":`
   smoke block that passes with no hardware library installed.

An agent may itself route further: the `code` agent front-doors four modes
(navigate / edit / compute / author) through `threetoks/code/route.py` —
a free pre-check only for read-only navigate questions, one one-token menu
for everything else (docs/DESIGN-coding-agent.md §10). It also reads
`services.retriever` (the opt-in retrieval-as-repair hook, None unless
`[code] retrieval` is enabled and a search provider exists),
`services.files_root` (the corpus the navigate/edit/compute modes operate
on), and `services.confirm_edit` (`(rel_path, diff) -> bool`; the edit mode
writes a file only when it returns True — None means never write). Code
the agent runs goes through `threetoks/code/sandbox.py`, never a bare
`subprocess` call.

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
