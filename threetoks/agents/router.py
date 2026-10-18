"""Routing is itself a one-token decision: an agent picks the agent.

The router shows the model a menu of agent descriptions and the user
request; the chosen AgentSpec handles the task. Invalid choice falls
back to the first spec (by convention the casual agent). Routing is
deliberately a MODEL decision — no keyword shortcuts; when it misfires
the fix is a stronger instruction or few-shot below, never a heuristic.

Small models have a position bias on borderline requests: one option
order gives web, another gives code. So the router asks with two option
orders; when they agree (the common case) it is done, else it asks up
to ROUTER_MAX_VOTES times and takes the majority.
"""
from collections import Counter

from threetoks.agents.base import AgentSpec
from threetoks.nodes import MenuNode
from threetoks.policy import Policy
from threetoks.render import Episode

ROUTER_AGREE_VOTES = 2    # this many first answers agreeing ends the vote
ROUTER_MAX_VOTES = 5      # most model calls one routing decision may take

ROUTER_PREFIX = """You dispatch each request to the agent that fits it best.
Rules:
- Any request to search, research, look up, find out, or investigate
  something goes to web — even when it is phrased as a question.
- Questions about facts of the world (people, public projects, news,
  forecasts) go to web.
- Questions about THIS code go to code: this project, this repo, the
  source here, or a function, class, or file in it — what it is, what
  it does, how it works, where something is. Words like "this code",
  "this project", or "this repo" mean the local code, never the web.
- General questions about a language, library, tool, or concept (how
  decorators work, what a framework is) go to web: they ask about the
  world, not about this code.
- casual is ONLY for greetings, thanks, and chit-chat that asks for no
  information.
- files is ONLY when the request itself names local files, folders, or
  a path ("my notes folder", "this directory"). A search or research
  request NEVER goes to files.
- Writing, fixing, explaining, or exploring a program goes to code.
- planner is ONLY for a request that needs SEVERAL steps done by
  different agents, e.g. look something up AND then write code with it.
  A request one agent can finish NEVER goes to planner.

Example — small talk:
Request: hey there, how are you
ACTIONS:
1 = casual — chat and small talk
2 = web — look up facts on the internet
3 = code — write or fix Python code, or explain the code in this project
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
1 = code — write or fix Python code, or explain the code in this project
2 = files — browse local files and folders
3 = web — look up facts on the internet
ANSWER: 3

Example — forecast question:
Request: how will the housing market do next year
ACTIONS:
1 = casual — chat and small talk
2 = code — write or fix Python code, or explain the code in this project
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
2 = code — write or fix Python code, or explain the code in this project
3 = casual — chat and small talk
ANSWER: 2

Example — local files:
Request: what's inside my notes folder
ACTIONS:
1 = files — browse local files and folders
2 = web — look up facts on the internet
ANSWER: 1

Example — several steps by different agents:
Request: find the population of Japan, then write a python script that prints it
ACTIONS:
1 = web — look up facts on the internet
2 = planner — plan several steps and run the agents in order
3 = code — write or fix Python code, or explain the code in this project
ANSWER: 2

Example — one agent is enough:
Request: write a script that prints the first 10 primes
ACTIONS:
1 = planner — plan several steps and run the agents in order
2 = code — write or fix Python code, or explain the code in this project
3 = web — look up facts on the internet
ANSWER: 2

Example — a question about this code:
Request: how does the retry logic in this repo work
ACTIONS:
1 = web — look up facts on the internet
2 = casual — chat and small talk
3 = code — write or fix Python code, or explain the code in this project
ANSWER: 3

Example — what this program is:
Request: tell me what this program is for
ACTIONS:
1 = code — write or fix Python code, or explain the code in this project
2 = web — look up facts on the internet
3 = files — browse local files and folders
ANSWER: 1

Example — a general programming question is not about this code:
Request: how do python decorators work
ACTIONS:
1 = code — write or fix Python code, or explain the code in this project
2 = casual — chat and small talk
3 = web — look up facts on the internet
ANSWER: 3"""


def route(task: str, specs: list[AgentSpec], policy: Policy) -> AgentSpec:
    """Pick the agent for a user request by a short consensus vote.

    The TUI calls this for each request, and the planner calls it when
    it makes no plan.

    Args:
        task (str): the user request.
        specs (list of AgentSpec): the agents to choose from; the first
            one is the fallback when no answer is valid.
        policy (Policy): the live decision policy.

    Returns:
        AgentSpec: the chosen agent.
    """
    if len(specs) == 1:
        return specs[0]
    by_option = {f"{s.name} — {s.description}": s for s in specs}
    episode = Episode(ROUTER_PREFIX, task)
    node = MenuNode("Which agent should handle this request?",
                    list(by_option), escape=False)
    return by_option.get(_consensus(episode, node, policy), specs[0])


def _consensus(episode: Episode, node: MenuNode, policy: Policy):
    """Ask the menu with new option orders until the answer is settled.

    Each ``policy.decide`` call raises the episode step, so each vote
    sees another option order and position bias cannot repeat itself.

    Args:
        episode (Episode): the router episode (system text and request).
        node (MenuNode): the agent menu.
        policy (Policy): the live decision policy.

    Returns:
        str or None: the winning option text; ties go to the earliest
        answer; None when no answer was valid.
    """
    counts = Counter()
    for asked in range(1, ROUTER_MAX_VOTES + 1):
        decision = policy.decide(episode, node)
        if decision.valid:
            counts[decision.value] += 1
        if _settled(counts, asked):
            break
    return counts.most_common(1)[0][0] if counts else None


def _settled(counts: Counter, asked: int) -> bool:
    """Say whether more votes can no longer change the answer.

    Args:
        counts (Counter): valid answers so far, option text -> votes.
        asked (int): how many votes were asked so far.

    Returns:
        bool: True when the first votes all agree, or one answer
        already holds a majority of ROUTER_MAX_VOTES.
    """
    top = max(counts.values(), default=0)
    agreed = asked == ROUTER_AGREE_VOTES and top == ROUTER_AGREE_VOTES
    return agreed or top > ROUTER_MAX_VOTES // 2


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
