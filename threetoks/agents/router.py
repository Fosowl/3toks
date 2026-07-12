"""Routing is itself a one-token decision: an agent picks the agent.

The router shows the model a menu of agent descriptions and the user
request; the chosen AgentSpec handles the task. Invalid choice falls
back to the first spec (by convention the casual agent). Routing is
deliberately a MODEL decision — no keyword shortcuts; when it misfires
the fix is a stronger instruction or few-shot below, never a heuristic.
"""
from threetoks.agents.base import AgentSpec
from threetoks.nodes import MenuNode
from threetoks.policy import Policy
from threetoks.render import Episode

ROUTER_PREFIX = """You dispatch each request to the agent that fits it best.
Rules:
- Any request to search, research, look up, find out, or investigate
  something goes to web — even when it is phrased as a question.
- Questions about facts, people, projects, news, or forecasts go to web.
- casual is ONLY for greetings, thanks, and chit-chat that asks for no
  information.
- files is ONLY when the request itself names local files, folders, or
  a path ("my notes folder", "this directory"). A search or research
  request NEVER goes to files.
- Writing or debugging a program goes to code.

Example — small talk:
Request: hey there, how are you
ACTIONS:
1 = casual — chat and small talk
2 = web — look up facts on the internet
3 = code — write or debug Python code
ANSWER: 1

Example — search for a person:
Request: do a deep search about who is Ada Lovelace
ACTIONS:
1 = casual — chat and small talk
2 = web — look up facts on the internet
3 = files — browse local files and folders
ANSWER: 2

Example — find out about something:
Request: search what the FooBar project is
ACTIONS:
1 = code — write or debug Python code
2 = files — browse local files and folders
3 = web — look up facts on the internet
ANSWER: 3

Example — forecast question:
Request: how will the housing market do next year
ACTIONS:
1 = casual — chat and small talk
2 = code — write or debug Python code
3 = web — look up facts on the internet
ANSWER: 3

Example — a question is never chit-chat:
Request: what are the economy projections for France
ACTIONS:
1 = casual — chat and small talk
2 = web — look up facts on the internet
ANSWER: 2

Example — a search is never about local files:
Request: deep search about the Marianas trench
ACTIONS:
1 = files — browse local files and folders
2 = casual — chat and small talk
3 = web — look up facts on the internet
ANSWER: 3

Example — write code:
Request: write a function that reverses a string
ACTIONS:
1 = web — look up facts on the internet
2 = code — write or debug Python code
3 = casual — chat and small talk
ANSWER: 2

Example — local files:
Request: what's inside my notes folder
ACTIONS:
1 = files — browse local files and folders
2 = web — look up facts on the internet
ANSWER: 1"""


def route(task: str, specs: list[AgentSpec], policy: Policy) -> AgentSpec:
    """Pick the agent for a user request with a single menu decision."""
    if len(specs) == 1:
        return specs[0]
    by_option = {f"{s.name} — {s.description}": s for s in specs}
    episode = Episode(ROUTER_PREFIX, task)
    node = MenuNode("Which agent should handle this request?",
                    list(by_option), escape=False)
    decision = policy.decide(episode, node)
    if decision.valid and decision.value in by_option:
        return by_option[decision.value]
    return specs[0]


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import PolicyConfig

    class _PickSecond:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" 2", 5, 2, 0.0, "stop")

    specs = [AgentSpec("casual", "small talk", lambda *a: {"answer": "hi"}),
             AgentSpec("web", "internet research", lambda *a: {"answer": ""})]
    policy = Policy(_PickSecond(),
                    PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    chosen = route("find the population of France", specs, policy)
    assert chosen.name in {"casual", "web"}
    print("smoke OK")
