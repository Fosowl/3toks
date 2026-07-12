"""Agent registry: the list the router dispatches over.

Adding an agent is one module exposing a ``SPEC``; list it here. Order
matters: the FIRST spec is the router's fallback for an invalid choice
(see threetoks/agents/router.py), so the casual agent leads.
"""
from threetoks.agents import casual, code, files, web
from threetoks.agents.base import AgentSpec
from threetoks.services import Services

__all__ = ["AgentSpec", "default_agents"]


def default_agents(services: Services) -> list[AgentSpec]:
    """The built-in agents, casual first (router fallback)."""
    return [casual.SPEC, web.SPEC, files.SPEC, code.SPEC]


if __name__ == "__main__":
    specs = default_agents(Services())
    names = [spec.name for spec in specs]
    assert names == ["casual", "web", "files", "code"], names
    assert len(set(names)) == len(names), "names must be unique"
    print("smoke OK")
