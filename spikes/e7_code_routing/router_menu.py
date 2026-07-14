"""One-token MenuNode router for code requests that survive the pre-router.

Built on ThreeToks' real Policy + MenuNode + Episode machinery (see
threetoks/agents/router.py for the pattern this copies: a single one-token
menu decision, few-shot prefix, invalid choice falls back to a default).
The four modes are treated as exhaustive for a coding request, so the menu
carries no escape option (same choice router.py makes for its top-level
agent menu).
"""
from pathlib import Path

from threetoks.nodes import MenuNode
from threetoks.policy import Policy
from threetoks.render import Episode

from pre_router import AUTHOR, COMPUTE, EDIT, NAVIGATE, MODES

# Fallback when the model's pick doesn't parse to one of the four modes.
FALLBACK_MODE = NAVIGATE

OPTION_TEXT = {
    NAVIGATE: "navigate — find or explain code that already exists",
    EDIT: "edit — fix, rename, or modify code that already exists",
    COMPUTE: "compute — run code to produce a fact or number the user wants",
    AUTHOR: "author — write a brand-new script, function, or program",
}

CODE_ROUTER_PREFIX = """You classify a coding request into one of four modes.
Rules:
- navigate is a QUESTION about code that already exists: where something
  is defined, what a function does, how a piece of logic works. Nothing
  changes and no new file is produced.
- edit is for CHANGING code that already exists: fixing a bug, renaming,
  refactoring, adding a function to a file the user names.
- compute is for when the user wants the ANSWER a computation produces —
  code is only the means (parsing a log for the top IP, summing a column).
  The deliverable is the fact or number, not the code itself.
- author is for when the user wants a NEW code artifact delivered: "write
  a script", "create a function", "build a tool". The deliverable is the
  code itself, even when the script's purpose is a computation.

Example — locate code:
Request: where is the login check implemented?
ACTIONS:
1 = edit — fix, rename, or modify code that already exists
2 = navigate — find or explain code that already exists
3 = author — write a brand-new script, function, or program
4 = compute — run code to produce a fact or number the user wants
ANSWER: 2

Example — fix a bug:
Request: there's an off-by-one error in the loop in parser.py, please fix it
ACTIONS:
1 = navigate — find or explain code that already exists
2 = compute — run code to produce a fact or number the user wants
3 = edit — fix, rename, or modify code that already exists
4 = author — write a brand-new script, function, or program
ANSWER: 3

Example — wants an answer, not code:
Request: how many times does the word "error" show up in app.log?
ACTIONS:
1 = author — write a brand-new script, function, or program
2 = edit — fix, rename, or modify code that already exists
3 = navigate — find or explain code that already exists
4 = compute — run code to produce a fact or number the user wants
ANSWER: 4

Example — wants a deliverable:
Request: can you build me a tool that dedupes a list of email addresses
ACTIONS:
1 = compute — run code to produce a fact or number the user wants
2 = navigate — find or explain code that already exists
3 = author — write a brand-new script, function, or program
4 = edit — fix, rename, or modify code that already exists
ANSWER: 3

Example — asking what code does:
Request: what does the retry ladder in policy.py actually do?
ACTIONS:
1 = compute — run code to produce a fact or number the user wants
2 = author — write a brand-new script, function, or program
3 = edit — fix, rename, or modify code that already exists
4 = navigate — find or explain code that already exists
ANSWER: 4

Example — renaming, not asking:
Request: rename the helper() function to normalize() everywhere it's used
ACTIONS:
1 = navigate — find or explain code that already exists
2 = author — write a brand-new script, function, or program
3 = compute — run code to produce a fact or number the user wants
4 = edit — fix, rename, or modify code that already exists
ANSWER: 4

Example — a script whose purpose is a computation is still author:
Request: write a script that reads a CSV and prints the average of column 3
ACTIONS:
1 = edit — fix, rename, or modify code that already exists
2 = compute — run code to produce a fact or number the user wants
3 = navigate — find or explain code that already exists
4 = author — write a brand-new script, function, or program
ANSWER: 4

Example — the fact is the deliverable, no script requested:
Request: what's the total revenue in this spreadsheet?
ACTIONS:
1 = author — write a brand-new script, function, or program
2 = navigate — find or explain code that already exists
3 = compute — run code to produce a fact or number the user wants
4 = edit — fix, rename, or modify code that already exists
ANSWER: 3"""


def classify_with_menu(task: str, policy: Policy):
    """Ask the model to pick one of the four modes; returns (mode, decision)."""
    episode = Episode(CODE_ROUTER_PREFIX, task)
    node = MenuNode("Which mode does this coding request need?",
                    [OPTION_TEXT[mode] for mode in MODES], escape=False)
    decision = policy.decide(episode, node)
    if decision.valid:
        for mode in MODES:
            if decision.value == OPTION_TEXT[mode]:
                return mode, decision
    return FALLBACK_MODE, decision


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import PolicyConfig

    class _PickThird:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" 3", 5, 2, 0.0, "stop")

    policy = Policy(_PickThird(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    mode, decision = classify_with_menu("write me a script that sorts a list",
                                        policy)
    assert mode in MODES and decision.valid
    print("smoke OK")
