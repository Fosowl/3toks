"""Code-mode routing: which of four modes does a coding request need?

    navigate  answer a question about existing code (nothing changes)
    edit      modify existing code (fix, rename, delete, add-to)
    compute   the user wants the ANSWER a computation produces
    author    the user wants a NEW code artifact delivered

Spike E7 measured a two-stage design live: deterministic pre-checks, then
a one-token menu (~3 output tokens per model-routed request). The free
pre-check is now limited to the one mode whose mistakes cost nothing:
navigate is read-only. Edit, compute and author change files or run code,
so they always cost one menu call. E7's confusion table showed every menu
error funnels into compute/author, so the few-shot prefix carries explicit
navigate/edit-vs-compute contrast examples.
"""
from threetoks.code.edit import rank
from threetoks.nodes import MenuNode
from threetoks.policy import Policy
from threetoks.render import Episode

NAVIGATE, EDIT, COMPUTE, AUTHOR = "navigate", "edit", "compute", "author"
MODES = (NAVIGATE, EDIT, COMPUTE, AUTHOR)
FALLBACK_MODE = NAVIGATE      # a wrong navigate is read-only, cheapest miss

EDIT_VERBS = ("fix", "rename", "refactor", "debug", "patch", "remove",
              "delete", "modify", "update", "change", "clean up", "add",
              "correct", "optimize", "simplify", "improve", "reformat")
NAVIGATE_VERBS = ("where is", "where does", "where's", "what does",
                  "what is", "explain", "show me", "find the definition",
                  "locate", "how does", "which line", "which function",
                  "which method", "which class", "walk me through",
                  "describe")
COMPUTE_VERBS = ("parse", "sum", "count", "compute", "calculate", "extract",
                 "average", "how many", "total", "most common", "median",
                 "filter", "group by")
AUTHOR_VERBS = ("write", "create", "build", "generate", "make me", "author")
BLOCKING_CUES = EDIT_VERBS + COMPUTE_VERBS + AUTHOR_VERBS

NAVIGATE_ENTITY_WORDS = ("function", "method", "class", "variable", "module",
                         "file", "defined", "definition", "declared",
                         "implemented", "logic", "symbol")

WORD_CHARS = "_'"             # kept inside words ("where's", "parse_row")
CURLY_APOSTROPHE = "\u2019"
TOKEN_EDGE_CHARS = "\"'`()[]{}<>,;:!?."   # stripped from path tokens


def pre_route(request: str) -> tuple[str | None, str]:
    """Give the free route for a coding request, or ask for the menu.

    The code agent uses this check to save one model call on safe requests.
    route_mode calls it with the raw user request text.
    It returns NAVIGATE only for a read-only question that names code: a
    navigate cue, a code-entity word or a path-like token, and no edit,
    compute or author cue. Every other request goes to the menu.

    Args:
        request (str): the user's coding request, from route_mode.

    Returns:
        tuple[str | None, str]: (NAVIGATE or None, reason for the log).
    """
    padded = _padded_words(request)
    if _contains_any(padded, BLOCKING_CUES):
        return None, "edit, compute or author cue: ask the menu"
    if not _contains_any(padded, NAVIGATE_VERBS):
        return None, "no navigate cue: ask the menu"
    if _contains_any(padded, NAVIGATE_ENTITY_WORDS):
        return NAVIGATE, "navigate cue + code-entity word"
    if any(_is_path_like(token) for token in request.split()):
        return NAVIGATE, "navigate cue + path-like token"
    return None, "navigate cue but no code entity: ask the menu"


def _padded_words(request: str) -> str:
    """Turn a request into lowercase words for phrase matching.

    pre_route uses this text so that a cue matches whole words only.
    pre_route calls it with the raw request text.
    It lowercases the text, splits it with rank.words (keeping "_" and
    "'"), joins the words with single spaces and adds one space at each
    end.

    Args:
        request (str): the raw user request, from pre_route.

    Returns:
        str: the normalised words, with a space at each end.
    """
    low = request.lower().replace(CURLY_APOSTROPHE, "'")
    return " " + " ".join(rank.words(low, keep=WORD_CHARS)) + " "


def _contains_any(padded: str, phrases: tuple[str, ...]) -> bool:
    """Tell if one of the phrases occurs as whole words in the text.

    pre_route uses this to find cue words ("sum" must not match "consume").
    pre_route calls it with the output of _padded_words and a cue tuple.
    It looks for each phrase with one space on each side.

    Args:
        padded (str): normalised words with edge spaces, from _padded_words.
        phrases (tuple[str, ...]): lowercase cue phrases, module constants.

    Returns:
        bool: True when at least one phrase occurs.
    """
    return any(f" {phrase} " in padded for phrase in phrases)


def _is_path_like(token: str) -> bool:
    """Tell if one request token looks like a file path.

    pre_route uses this so "what does parser.py do?" routes free.
    pre_route calls it with one whitespace-separated request token.
    It removes edge punctuation, then accepts a token with "/" between two
    parts, or with a "." followed by a letter-bearing extension.

    Args:
        token (str): one token of the raw request, from pre_route.

    Returns:
        bool: True when the token looks like a path or a file name.
    """
    core = token.strip(TOKEN_EDGE_CHARS)
    if len([part for part in core.split("/") if part]) >= 2:
        return True
    stem, dot, extension = core.rpartition(".")
    return bool(stem and dot and extension.isalnum()
                and not extension.isdigit())


OPTION_TEXT = {
    NAVIGATE: "navigate — find or explain code that already exists",
    EDIT: "edit — fix, rename, or modify code that already exists",
    COMPUTE: "compute — run code to produce a fact or number the user wants",
    AUTHOR: "author — write a brand-new script, function, or program",
}

CODE_ROUTER_PREFIX = """You classify a coding request into one of four modes.
Rules:
- navigate is a QUESTION about code that already exists: where something
  is defined, what a function does, how a piece of logic works, why it
  behaves the way it does. Nothing changes and no new file is produced.
- edit is for CHANGING code that already exists: fixing a bug, renaming,
  refactoring, adding a function to a file the user names. "Debug why X
  returns the wrong value" wants the bug FIXED: that is edit.
- compute is for when the user wants the ANSWER a computation produces —
  code is only the means (parsing a log for the top IP, summing a column).
  The deliverable is the fact or number, not the code itself. A request
  about wrong or broken EXISTING code is never compute.
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

Example — asking HOW code works is navigate even when it says "computed":
Request: how is the discount computed in billing.py?
ACTIONS:
1 = compute — run code to produce a fact or number the user wants
2 = navigate — find or explain code that already exists
3 = edit — fix, rename, or modify code that already exists
4 = author — write a brand-new script, function, or program
ANSWER: 2

Example — "debug why it returns the wrong value" is edit, not compute:
Request: debug why parse_row returns the wrong column sometimes
ACTIONS:
1 = compute — run code to produce a fact or number the user wants
2 = author — write a brand-new script, function, or program
3 = navigate — find or explain code that already exists
4 = edit — fix, rename, or modify code that already exists
ANSWER: 4

Example — wants an answer, not code:
Request: how many times does the word "error" show up in app.log?
ACTIONS:
1 = author — write a brand-new script, function, or program
2 = edit — fix, rename, or modify code that already exists
3 = navigate — find or explain code that already exists
4 = compute — run code to produce a fact or number the user wants
ANSWER: 4

Example — "tell me which ..." wants the fact, not a script:
Request: group the orders by country and tell me which country ordered most
ACTIONS:
1 = author — write a brand-new script, function, or program
2 = compute — run code to produce a fact or number the user wants
3 = edit — fix, rename, or modify code that already exists
4 = navigate — find or explain code that already exists
ANSWER: 2

Example — wants a deliverable:
Request: can you build me a tool that dedupes a list of email addresses
ACTIONS:
1 = compute — run code to produce a fact or number the user wants
2 = navigate — find or explain code that already exists
3 = author — write a brand-new script, function, or program
4 = edit — fix, rename, or modify code that already exists
ANSWER: 3

Example — a script whose purpose is a computation is still author:
Request: write a script that reads a CSV and prints the average of column 3
ACTIONS:
1 = edit — fix, rename, or modify code that already exists
2 = compute — run code to produce a fact or number the user wants
3 = navigate — find or explain code that already exists
4 = author — write a brand-new script, function, or program
ANSWER: 4

Example — renaming, not asking:
Request: rename the helper() function to normalize() everywhere it's used
ACTIONS:
1 = navigate — find or explain code that already exists
2 = author — write a brand-new script, function, or program
3 = compute — run code to produce a fact or number the user wants
4 = edit — fix, rename, or modify code that already exists
ANSWER: 4"""


def route_mode(request: str, policy: Policy) -> tuple[str, str]:
    """Choose the mode for one coding request.

    The code agent uses this as its front door before it runs a mode.
    agents/code.run calls it with the user request and the live policy.
    It takes the free pre-check route when there is one, else it asks one
    one-digit menu. An invalid menu answer gives FALLBACK_MODE.

    Args:
        request (str): the user's coding request, from the agent run.
        policy (Policy): the live decision policy, from the agent run.

    Returns:
        tuple[str, str]: (mode, how) where how is "pre:<reason>", "menu"
        or "menu-fallback".
    """
    mode, reason = pre_route(request)
    if mode is not None:
        return mode, f"pre:{reason}"
    episode = Episode(CODE_ROUTER_PREFIX, request)
    node = MenuNode("Which mode does this coding request need?",
                    [OPTION_TEXT[mode] for mode in MODES], escape=False)
    node.tag = "mode"  # a classification, not an action
    decision = policy.decide(episode, node)
    chosen = next((mode for mode in MODES
                   if decision.valid and decision.value == OPTION_TEXT[mode]),
                  None)
    if chosen is None:
        return FALLBACK_MODE, "menu-fallback"
    return chosen, "menu"


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import PolicyConfig

    assert pre_route("fix the bug in parser.py")[0] is None
    assert pre_route("where is the Cache class implemented?")[0] == NAVIGATE
    assert pre_route("what does parser.py actually do?")[0] == NAVIGATE
    assert pre_route("where is the add function in parser.py?")[0] is None
    assert pre_route("sum the revenue column in the csv")[0] is None
    assert pre_route("write a script that reverses a string")[0] is None
    assert pre_route("hello there")[0] is None

    class _PickFirst:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" 1", 5, 2, 0.0, "stop")

    policy = Policy(_PickFirst(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    mode, how = route_mode("hmm, something about the code", policy)
    assert mode in MODES and how == "menu"
    print("smoke OK")
