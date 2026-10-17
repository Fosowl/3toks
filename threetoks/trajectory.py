"""A trajectory: the plan that the planner agent drafts and then follows.

A trajectory is an ordered list of steps. Each step names one agent and
one short sentence that tells how the agent helps the overall goal. The
planner agent (threetoks/agents/planner.py) drafts the steps, runs them
one by one, and can replace the steps that are still to come. This
module keeps only the data and the texts that the agents and the model
read; it makes no model call.
"""
from dataclasses import dataclass, field

STEP_ANSWER_CLIP = 300   # earlier answer chars a later step reads
NO_ANSWER = "(no answer)"
CONTEXT_HEADER = "YOUR TASK IS ONE STEP OF A LARGER PLAN."


@dataclass
class Step:
    """One planned step: which agent runs, and what it does for the goal.

    ``answer`` stays None until the agent has run the step.
    """
    agent: str
    purpose: str
    answer: str | None = None


@dataclass
class Trajectory:
    """The goal, the steps already done, and the steps still to run.

    ``unverified`` turns True when a step result says a judge rejected it.
    """
    goal: str
    remaining: list[Step]
    done: list[Step] = field(default_factory=list)
    replans: int = 0
    unverified: bool = False

    def context(self) -> str:
        """Give the plan context that every model call of a step sees.

        The planner agent adds this text to the system block of the step
        agent (Policy.with_context); the agent task stays the short step
        purpose, so search queries and keyword rules see only the purpose.
        The planner calls it before it runs the next step.
        It gives the overall goal, then the clipped answers of the steps
        already done, so a later step can use them.

        Returns:
            str: the multi-line context text.
        """
        lines = [CONTEXT_HEADER, f"OVERALL GOAL: {self.goal}"]
        if self.done:
            lines += ["EARLIER STEPS:", self.render_done()]
        return "\n".join(lines)

    def render_done(self) -> str:
        """Give one line per done step, with its clipped answer.

        Later agents read these lines in their plan context, and the
        planner reads them when it drafts the steps still needed.
        context and the planner call it when at least one step is done.
        It gives "- web: find the price -> 12 euros" for each done step.

        Returns:
            str: the done-step lines, joined by newlines.
        """
        return "\n".join(f"- {step.agent}: {step.purpose} -> "
                         f"{_clip(step.answer or NO_ANSWER)}"
                         for step in self.done)

    def render_latest(self) -> str:
        """Give the purpose and clipped answer of the last done step.

        The planner shows this text when it asks if the plan needs an
        update. The planner calls it right after a step is done.
        It names the agent and purpose, then gives the clipped answer.

        Returns:
            str: a two-line block, or "" when no step is done.
        """
        if not self.done:
            return ""
        step = self.done[-1]
        return (f"LATEST RESULT ({step.agent}: {step.purpose}):\n"
                f"{_clip(step.answer or NO_ANSWER)}")

    def render_plan(self) -> str:
        """Give the numbered plan, with the done steps marked.

        The planner agent shows this text to the model when it asks if
        the plan needs an update, and prints it as progress.
        The planner calls it after a step is done or a plan is drafted.
        It gives one line per step: "2. web — find the price (done)".

        Returns:
            str: the plan lines, joined by newlines.
        """
        steps = [(step, " (done)") for step in self.done]
        steps += [(step, "") for step in self.remaining]
        return "\n".join(f"{number}. {step.agent} — {step.purpose}{mark}"
                         for number, (step, mark) in
                         enumerate(steps, start=1))

    def summary(self) -> str:
        """Give the final answer text for the result panel.

        The planner agent puts this text in the ``answer`` of its result.
        The planner calls it when no step is left.
        One done step gives its answer as it is; more steps give one
        titled block per step, so no answer is lost.

        Returns:
            str: the final answer text.
        """
        if len(self.done) == 1:
            return self.done[0].answer or NO_ANSWER
        return "\n\n".join(f"{number}. {step.agent} — {step.purpose}\n"
                           f"{step.answer or NO_ANSWER}"
                           for number, step in
                           enumerate(self.done, start=1))


def _clip(text: str) -> str:
    """Make a text one line and clip it to STEP_ANSWER_CLIP characters.

    render_done and render_latest use this to keep earlier answers short.
    It receives the answer of a done step.
    It joins all whitespace into single spaces and cuts the end.

    Args:
        text (str): an agent answer, from a done Step.

    Returns:
        str: the clipped one-line text, with "…" when it was cut.
    """
    flat = " ".join(text.split())
    if len(flat) <= STEP_ANSWER_CLIP:
        return flat
    return flat[:STEP_ANSWER_CLIP - 1] + "…"


if __name__ == "__main__":
    trajectory = Trajectory("print the population of France",
                            [Step("web", "find the population of France"),
                             Step("code", "write a script printing it")])
    first = trajectory.remaining.pop(0)
    assert "EARLIER STEPS" not in trajectory.context()
    first.answer = "68 million"
    trajectory.done.append(first)
    context = trajectory.context()
    assert "OVERALL GOAL: print the population of France" in context
    assert "- web: find the population of France -> 68 million" in context
    assert "1. web — find the population of France (done)" in \
        trajectory.render_plan()
    assert trajectory.summary() == "68 million"
    print("smoke OK")
