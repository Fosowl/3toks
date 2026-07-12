"""Web agent: research questions using internet search.

A thin adapter over :func:`threetoks.research.deep_research` — the whole
research state machine (judged rounds, note accumulation) lives there.
The agent just runs it and tags the result with its own name.
"""
from threetoks.agents.base import AgentSpec
from threetoks.research import deep_research
from threetoks.services import Services

AGENT_NAME = "web"
AGENT_DESCRIPTION = "look up facts or current information on the internet"


def run(task: str, services: Services, policy) -> dict:
    """Run judged deep research and tag the result with the agent name."""
    result = deep_research(task, services, policy)
    result["agent"] = AGENT_NAME
    return result


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    class _NoResults:
        def search(self, query, max_results=8):
            return []

    class _YesBackend:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" 1", 5, 2, 0.0, "stop")

    policy = Policy(_YesBackend(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    services = Services(provider=_NoResults(), max_research_rounds=1)
    outcome = SPEC.run("what is 2+2?", services, policy)
    assert outcome["agent"] == "web"
    print("smoke OK")
