"""Agent registry: the list the router dispatches over.

Adding an agent is one module exposing a ``SPEC``; list it here. Order
matters: the FIRST spec is the router's fallback for an invalid choice
(see threetoks/agents/router.py), so the casual agent leads.
"""
from threetoks.agents import casual, code, files, web
from threetoks.agents.base import AgentSpec
from threetoks.services import Services

__all__ = ["AgentSpec", "OPTIONAL_AGENT_NAMES", "default_agents",
           "optional_agents"]

# Every name optional_agents can produce; /model uses this to re-derive
# the capability agents when the model (and its vision ability) changes.
OPTIONAL_AGENT_NAMES = ("light", "look")


def default_agents(services: Services) -> list[AgentSpec]:
    """The built-in agents, casual first (router fallback)."""
    return [casual.SPEC, web.SPEC, files.SPEC, code.SPEC]


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
    print("smoke OK")
