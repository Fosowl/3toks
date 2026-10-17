"""Agent registry: the list the router dispatches over.

Adding an agent is one module exposing a ``SPEC``; list it here. Order
matters: the FIRST spec is the router's fallback for an invalid choice
(see threetoks/agents/router.py), so the casual agent leads. The
planner comes last: ``with_planner`` builds it over the final list.
"""
from threetoks.agents import casual, code, files, planner, web
from threetoks.agents.base import AgentSpec
from threetoks.services import Services

__all__ = ["OPTIONAL_AGENT_NAMES", "PLANNER_NAME", "AgentSpec",
           "default_agents", "optional_agents", "with_planner"]

# Every name optional_agents can produce; /model uses this to re-derive
# the capability agents when the model (and its vision ability) changes.
OPTIONAL_AGENT_NAMES = ("light", "look")
PLANNER_NAME = planner.AGENT_NAME
MIN_PLANNED_AGENTS = 2  # one agent alone needs no plan


def default_agents(services: Services) -> list[AgentSpec]:
    """The built-in agents, casual first (router fallback)."""
    return [casual.SPEC, web.SPEC, files.SPEC, code.SPEC]


def with_planner(specs: list[AgentSpec]) -> list[AgentSpec]:
    """Add the planner agent after the agents it can plan with.

    The TUI uses this when it builds the agent list and after /model
    changes the optional agents. It receives the final agent list
    without a planner. It appends a planner bound to that list, so a
    plan can only use agents that are really ready.

    Args:
        specs (list[AgentSpec]): the ready agents, casual first.

    Returns:
        list[AgentSpec]: a new list; the planner is last when there are
        at least MIN_PLANNED_AGENTS agents.
    """
    if len(specs) < MIN_PLANNED_AGENTS:
        return list(specs)
    return list(specs) + [planner.make_spec(specs)]


def optional_agents(services: Services, model: str) -> list[AgentSpec]:
    """Agents that exist only when their optional capability is wired.

    ``light`` needs a live relay on ``services.relay`` (Pi-only extra);
    ``look`` needs a camera capture callable AND a vision-capable model.
    Imports stay lazy so a bare install never loads them.
    """
    specs: list[AgentSpec] = []
    if getattr(services, "relay", None) is not None:
        from threetoks.agents import light
        specs.append(light.SPEC)
    if getattr(services, "capture_frame", None) is not None and _sees(model):
        from threetoks.agents import look
        specs.append(look.SPEC)
    return specs


def _sees(model: str) -> bool:
    """Whether the model accepts image input (by name)."""
    from threetoks.backend.base import is_vision_model
    return is_vision_model(model)


if __name__ == "__main__":
    specs = default_agents(Services())
    names = [spec.name for spec in specs]
    assert names == ["casual", "web", "files", "code"], names
    assert len(set(names)) == len(names), "names must be unique"
    assert optional_agents(Services(), "qwen3.5:2b") == []
    wired = Services(relay=object(), capture_frame=lambda: "b64")
    extras = [spec.name for spec in optional_agents(wired, "llava:7b")]
    assert extras == ["light", "look"], extras
    text_only = [spec.name for spec in
                 optional_agents(wired, "qwen3.5:2b")]
    assert text_only == ["light"], text_only
    planned = [spec.name for spec in with_planner(specs)]
    assert planned == names + [PLANNER_NAME], planned
    assert with_planner(specs[:1]) == specs[:1]
    print("smoke OK")
