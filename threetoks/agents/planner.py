"""Planner agent: draft a trajectory of agents, follow it, correct it.

The router picks this agent for a request that needs several steps.
The planner shows the model the other agents, each with a number, and
the model gives the agent numbers in the order they must work (one
SequenceNode decision). Then the model gives each step one very short
sentence that tells how the agent helps the goal (one ShortTextNode per
step). That plan is a trajectory (threetoks/trajectory.py).

Each agent then runs its step. Its task text is only its step sentence,
so search queries and keyword rules stay clean; the overall goal and the
clipped answers of the steps already done reach every model call of the
step through Policy.with_context. After each step, one yes/no menu asks
if the steps still to come must change; "yes" drafts the remaining steps
again. Caps on steps and on replans keep a confused model from looping.
"""
from dataclasses import asdict

from threetoks.agents.base import AgentSpec
from threetoks.agents.router import route
from threetoks.nodes import MenuNode, SequenceNode, ShortTextNode
from threetoks.render import Episode
from threetoks.trajectory import Step, Trajectory

AGENT_NAME = "planner"
AGENT_DESCRIPTION = ("plan a request that needs several steps done by "
                     "different agents, then run them in order")
MAX_PLAN_STEPS = 4          # steps one draft can hold
MAX_REPLANS = 2             # plan updates per request
MAX_TRAJECTORY_STEPS = 6    # steps run per request, all drafts together
PURPOSE_MAX_TOKENS = 24     # one very short sentence

PLAN_QUESTION = "Which agents, in order, reach the goal?"
PURPOSE_QUESTION = ("Step {number} of {total} is done by {name}. In under "
                    "10 words, what does {name} do toward the goal?")
UPDATE_QUESTION = "Does the plan need an update to reach the goal?"
OPT_UPDATE = "yes — the result shows the next steps are wrong or not enough"
OPT_KEEP = "no — the next steps still reach the goal"
DONE_HEADER = "DONE STEPS (plan only the steps still needed):"

# Example digits avoid the default order: models copy them.
PLAN_PREFIX = """You plan a request that needs several steps.
One agent does each step. Pick the agents in the order they must work.
The numbers come from the AGENTS list of the request, never from an example.
The same agent can come back for a later step. Use as few steps as you can.
Then give each step a very short sentence: what its agent does for the goal.

Example — the numbers follow the AGENTS list:
TASK: find the population of Japan, then write a python script that prints it
AGENTS:
1 = code — write or debug Python code
2 = casual — chat and small talk
3 = web — look up facts on the internet
PLAN: 3,1
STEP 1 (web): find the current population of Japan
STEP 2 (code): write a script that prints that number"""

UPDATE_PREFIX = """You check a plan while it runs. An agent just finished
one step. Decide if the steps still to come must change.
Say yes when the result shows that the step failed, found something the
plan did not expect, or that the goal needs more steps.
Say no when the next steps still reach the goal, or the goal is reached."""


class Planner:
    """The planner run function, bound to the agents it can plan with.

    AgentSpec.run must take (task, services, policy) only, so this
    callable object keeps the agent list that the registry gives it.
    """

    def __init__(self, specs: list[AgentSpec]):
        """Keep the agents that the plan can use.

        Args:
            specs (list[AgentSpec]): the registered agents without the
                planner, from threetoks.agents.with_planner.
        """
        self.specs = list(specs)
        self.by_name = {spec.name: spec for spec in self.specs}

    def __call__(self, task: str, services, policy) -> dict:
        """Draft a trajectory for the request, follow it, and correct it.

        The router calls this when it picks the planner for a request.
        It receives the user request, the shared services and the policy.
        It drafts the plan, runs each step with its agent, asks after
        each step if the plan needs an update, and joins the answers.
        A plan that fails to parse falls back to one routed agent.

        Args:
            task (str): the user request (the overall goal), from the TUI.
            services (Services): shared capabilities, from the TUI; its
                optional ``on_progress`` callable gets progress lines.
            policy (Policy): the decision policy, from the TUI.

        Returns:
            dict: the agent result, with "agent", "answer" and
            "trajectory" (list of step dicts); on fallback, the routed
            agent result as it is.
        """
        trajectory = Trajectory(task, [])
        trajectory.remaining = draft(trajectory, self.specs, policy)
        if not trajectory.remaining:
            return self._fallback(task, services, policy)
        _notify_plan(services, "plan", trajectory)
        while trajectory.remaining and \
                len(trajectory.done) < MAX_TRAJECTORY_STEPS:
            self._run_step(trajectory, services, policy)
            self._correct(trajectory, services, policy)
        if trajectory.remaining:
            _notify(services, f"step cap reached › "
                              f"{len(trajectory.remaining)} step(s) not run")
        return _result(trajectory)

    def _run_step(self, trajectory: Trajectory, services, policy) -> None:
        """Run the next step with its agent and record the answer.

        The follow loop in __call__ calls this once per step.
        It receives the trajectory and the run context.
        It pops the next step and runs its agent with the step purpose as
        the task and a policy that adds the plan context to every call.
        It then moves the step to ``done`` with the agent answer, and
        marks the trajectory unverified when a judge rejected the result.

        Args:
            trajectory (Trajectory): the plan being followed (changed).
            services (Services): shared capabilities, from __call__.
            policy (Policy): the decision policy, from __call__.
        """
        step = trajectory.remaining.pop(0)
        _notify(services, f"step {len(trajectory.done) + 1} › "
                          f"{step.agent}: {step.purpose}")
        step_policy = policy.with_context(trajectory.context())
        outcome = self.by_name[step.agent].run(step.purpose, services,
                                               step_policy)
        step.answer = str(outcome.get("answer") or "")
        trajectory.unverified |= outcome.get("judged_good") is False
        trajectory.done.append(step)

    def _correct(self, trajectory: Trajectory, services, policy) -> None:
        """Ask if the plan needs an update; draft the rest again on yes.

        The follow loop in __call__ calls this after each step.
        It receives the trajectory with the step just done.
        It skips the question when the replan or step cap is reached;
        on "yes" it replaces the remaining steps with a new draft, and
        keeps the old steps when the new draft fails to parse.

        Args:
            trajectory (Trajectory): the plan being followed (changed).
            services (Services): shared capabilities, from __call__.
            policy (Policy): the decision policy, from __call__.
        """
        if trajectory.replans >= MAX_REPLANS or \
                len(trajectory.done) >= MAX_TRAJECTORY_STEPS:
            return
        if not needs_update(trajectory, policy):
            return
        trajectory.replans += 1
        steps = draft(trajectory, self.specs, policy)
        if steps:
            trajectory.remaining = steps
            _notify_plan(services, "plan updated", trajectory)

    def _fallback(self, task: str, services, policy) -> dict:
        """Route the request to one agent when no plan could be drafted.

        __call__ uses this when the plan decision gives no valid agent.
        It receives the user request and the run context.
        It asks the one-agent router and runs the agent it picks.

        Args:
            task (str): the user request, from __call__.
            services (Services): shared capabilities, from __call__.
            policy (Policy): the decision policy, from __call__.

        Returns:
            dict: the result of the routed agent.
        """
        spec = route(task, self.specs, policy)
        _notify(services, f"no plan › routed to {spec.name}")
        return spec.run(task, services, policy)


def draft(trajectory: Trajectory, specs: list[AgentSpec],
          policy) -> list[Step]:
    """Ask the model for the steps still needed to reach the goal.

    The planner uses this for the first plan and for each plan update.
    The planner calls it with the trajectory (goal and done steps).
    It shows the numbered agents and any done steps, asks one ordered
    list of agent numbers, then one short purpose per chosen agent.

    Args:
        trajectory (Trajectory): the goal and the done steps, from the
            planner.
        specs (list[AgentSpec]): the agents the plan can use, numbered
            from 1 in this order.
        policy (Policy): the decision policy, from the planner.

    Returns:
        list[Step]: the new steps, or [] when the list fails to parse.
    """
    episode = Episode(PLAN_PREFIX, trajectory.goal)
    episode.open_observation(_plan_observation(trajectory, specs))
    node = SequenceNode(PLAN_QUESTION, len(specs), MAX_PLAN_STEPS,
                        items=[spec.name for spec in specs])
    decision = policy.decide(episode, node)
    if not decision.valid:
        return []
    episode.log_session("PLAN: " + ",".join(map(str, decision.value)))
    chosen = [specs[number - 1] for number in decision.value]
    purposes = _purposes(episode, chosen, policy)
    return [Step(spec.name, purpose or trajectory.goal)
            for spec, purpose in zip(chosen, purposes)]


def _plan_observation(trajectory: Trajectory, specs: list[AgentSpec]) -> str:
    """Give the numbered agent list, plus the done steps on a replan.

    draft shows this text as the observation of the plan episode.
    draft calls it with the trajectory and the agents.
    It lists "N = name — description" for each agent; after the first
    plan it adds the done steps so the model plans only what is left.

    Args:
        trajectory (Trajectory): the goal and done steps, from draft.
        specs (list[AgentSpec]): the agents, from draft.

    Returns:
        str: the observation text.
    """
    lines = ["AGENTS:"] + [f"{number} = {spec.name} — {spec.description}"
                           for number, spec in enumerate(specs, start=1)]
    if trajectory.done:
        lines += [DONE_HEADER, trajectory.render_done()]
    return "\n".join(lines)


def _purposes(episode: Episode, chosen: list[AgentSpec],
              policy) -> list[str | None]:
    """Ask one very short purpose sentence for each chosen agent.

    draft uses this after the model has put the agents in order.
    draft calls it with the plan episode and the chosen agents.
    It asks one bounded text per step and logs each valid answer into
    the episode, so later steps see the earlier ones.

    Args:
        episode (Episode): the plan episode, from draft (changed).
        chosen (list[AgentSpec]): the agents in plan order, from draft.
        policy (Policy): the decision policy, from draft.

    Returns:
        list[str | None]: one purpose per chosen agent, in order; None
        where the answer was invalid (draft then uses the goal).
    """
    purposes: list[str | None] = []
    for number, spec in enumerate(chosen, start=1):
        label = f"STEP {number} ({spec.name}):"
        question = PURPOSE_QUESTION.format(number=number, total=len(chosen),
                                           name=spec.name)
        node = ShortTextNode(question, label, max_tokens=PURPOSE_MAX_TOKENS)
        node.tag = "step"
        decision = policy.decide(episode, node)
        purposes.append(decision.value if decision.valid else None)
        if decision.valid:
            episode.log_session(f"{label} {decision.value}")
    return purposes


def needs_update(trajectory: Trajectory, policy) -> bool:
    """Ask the yes/no question: must the steps still to come change?

    The planner uses this after each step to correct the trajectory.
    The planner calls it with the trajectory right after a step is done.
    It shows the plan and the latest result, and asks one menu whose
    two options say what "yes" and "no" mean. An invalid answer keeps
    the plan.

    Args:
        trajectory (Trajectory): the plan with at least one done step,
            from the planner.
        policy (Policy): the decision policy, from the planner.

    Returns:
        bool: True when the model says the plan needs an update.
    """
    episode = Episode(UPDATE_PREFIX, trajectory.goal)
    episode.open_observation(f"PLAN:\n{trajectory.render_plan()}\n\n"
                             f"{trajectory.render_latest()}")
    node = MenuNode(UPDATE_QUESTION, [OPT_UPDATE, OPT_KEEP], escape=False)
    node.tag = "replan"
    decision = policy.decide(episode, node)
    return decision.valid and decision.value == OPT_UPDATE


def _result(trajectory: Trajectory) -> dict:
    """Build the planner result from the followed trajectory.

    The planner returns this to the TUI when the trajectory ends.
    __call__ calls it with the trajectory after the last step.
    It sets the agent name, the joined answer and the step list; it adds
    judged_good False when any step result was rejected by a judge, so
    the panel shows UNVERIFIED and memory does not keep it.

    Args:
        trajectory (Trajectory): the followed plan, from __call__.

    Returns:
        dict: the result for the TUI panel and for memory.
    """
    result = {"agent": AGENT_NAME, "answer": trajectory.summary(),
              "trajectory": [asdict(step) for step in trajectory.done]}
    if trajectory.unverified:
        result["judged_good"] = False
    return result


def _notify(services, line: str) -> None:
    """Send one progress line to the optional ``on_progress`` callable.

    The planner uses this to show plans and steps while they run.
    The planner calls it with the shared services and a plain line.
    It does nothing when the services have no callable.

    Args:
        services (Services): shared capabilities, from the planner.
        line (str): the progress text, without color.
    """
    report = getattr(services, "on_progress", None)
    if callable(report):
        report(line)


def _notify_plan(services, title: str, trajectory: Trajectory) -> None:
    """Send the title and then one progress line per plan step.

    The planner uses this after the first draft and each update.
    The planner calls it with the services, a title and the trajectory.
    It sends "title ›" and then the numbered plan lines.

    Args:
        services (Services): shared capabilities, from the planner.
        title (str): "plan" or "plan updated".
        trajectory (Trajectory): the current plan, from the planner.
    """
    _notify(services, f"{title} ›")
    for line in trajectory.render_plan().splitlines():
        _notify(services, f"  {line}")


def make_spec(specs: list[AgentSpec]) -> AgentSpec:
    """Build the planner AgentSpec over the given agents.

    The registry (threetoks.agents.with_planner) uses this to add the
    planner after the other agents. It receives the agents the plan can
    use. It binds them to a Planner run function.

    Args:
        specs (list[AgentSpec]): the agents without the planner.

    Returns:
        AgentSpec: the planner spec.
    """
    return AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, Planner(specs))


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    class _ScriptedBackend:
        """Plan "web then code", short purposes, never ask for an update."""

        def complete(self, model, raw_prompt, opts):
            tail = raw_prompt.rstrip()
            if tail.endswith("PLAN:"):
                return GenResult(" 1,2", 5, 3, 0.0, "stop")
            if tail.endswith("ANSWER:"):
                digit = "1" if f"1 = {OPT_KEEP}" in raw_prompt else "2"
                return GenResult(f" {digit}", 5, 1, 0.0, "stop")
            return GenResult(" do the part", 5, 3, 0.0, "stop")

    seen: list[tuple[str, str]] = []
    agents = [AgentSpec("web", "look up facts",
                        lambda task, s, p: seen.append((task, p.context))
                        or {"answer": "68 million"}),
              AgentSpec("code", "write code",
                        lambda task, s, p: seen.append((task, p.context))
                        or {"answer": "print(68)"})]
    policy = Policy(_ScriptedBackend(),
                    PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    result = make_spec(agents).run("print France's population", None, policy)
    assert [step["agent"] for step in result["trajectory"]] == ["web", "code"]
    code_task, code_context = seen[1]
    assert code_task == "do the part", seen
    assert "OVERALL GOAL: print France's population" in code_context, seen
    assert "-> 68 million" in code_context, seen
    assert policy.context == "", "the caller policy must not change"
    assert result["agent"] == AGENT_NAME and "print(68)" in result["answer"]
    print("smoke OK")
