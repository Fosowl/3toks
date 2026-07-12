"""Code agent: write a small single-file Python module from a goal.

A thin adapter over :class:`threetoks.code.vertical.CodeVertical` — the whole
plan-implement-test state machine lives there. The agent runs one episode
and tags the rendered module as the answer.
"""
from threetoks.agents.base import AgentSpec
from threetoks.code.vertical import CodeVertical
from threetoks.engine import run_episode
from threetoks.services import Services

AGENT_NAME = "code"
AGENT_DESCRIPTION = "write or debug Python code: a function, script, or program"
MAX_STEPS = 40


def run(task: str, services: Services, policy) -> dict:
    """Write a module for the task; return it as the answer.

    Self-tests are off: E6 showed model-written asserts don't improve
    whole-module correctness on a 1.5b (the model resamples the same wrong
    logic) and only add latency, while the deterministic gates carry the
    quality. The mechanism stays available for trusted-example runs.
    """
    vertical = CodeVertical(task, gen_tests=False)
    result = run_episode(vertical, policy, max_steps=MAX_STEPS)
    result["agent"] = AGENT_NAME
    return result


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    class _Backend:
        def __init__(self, texts):
            self.texts = list(texts)

        def complete(self, model, raw_prompt, opts):
            return GenResult(self.texts.pop(0) if self.texts else "return None",
                             8, 4, 0.0, "stop")

    scripted = _Backend(["square(n): multiply n by itself",
                         "return n * n",
                         "assert square(3) == 9"])
    policy = Policy(scripted, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    outcome = SPEC.run("square a number", Services(), policy)
    assert outcome["agent"] == "code", outcome
    assert "return n * n" in outcome["answer"], outcome["answer"]
    print("smoke OK")
