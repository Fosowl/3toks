"""Code-mode routing: which of four modes does a coding request need?

    navigate  answer a question about existing code (nothing changes)
    edit      modify existing code (fix, rename, delete, add-to)
    compute   the user wants the ANSWER a computation produces
    author    the user wants a NEW code artifact delivered

Spike E7 measured this two-stage design live: deterministic pre-checks
resolve 69.6% of requests for free at 97.9% precision, and a one-token
menu handles the rest at 76.2% (91.3% end-to-end, ~3 output tokens per
model-routed request). The top-level agent router deliberately has no
keyword shortcuts; this sub-router earns its pre-checks with that
measurement — every check is high-PRECISION and falls through to the
menu on any doubt. E7's confusion table showed every menu error funnels
into compute/author, so the few-shot prefix carries explicit
navigate/edit-vs-compute contrast examples.
"""
import re
from pathlib import Path

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

NAVIGATE_ENTITY_WORDS = ("function", "method", "class", "variable", "module",
                         "file", "defined", "definition", "declared",
                         "implemented", "logic", "symbol")
COMPUTE_DATA_WORDS = ("csv", "log", "column", "data", "rows", "lines", "file")
AUTHOR_ARTIFACT_WORDS = ("script", "program", "module", "function", "class",
                         "tool", "cli", "library", "package", "code")

# bare "top" is too common ("top-level function"); only "top <number>" is
# a confident top-K compute signal (E7 found the bare-word false positive).
_TOP_K_RE = re.compile(r"\btop\s*\d+\b")
_PATH_TOKEN_RE = re.compile(r"[\w][\w./-]*\.\w+|[\w][\w/-]*/[\w./-]+")


def pre_route(request: str, root: Path | None) -> tuple[str | None, str]:
    """Deterministic, zero-token routing; (None, reason) means ask the menu.

    High precision over recall throughout: a wrong free classification is
    worse than one spent model call, so every ambiguous shape falls
    through.
    """
    low = request.lower()
    if _is_author_phrasing(low):
        return AUTHOR, "imperative write/create + new-artifact noun"
    if root is not None and _mentions_existing_path(request, root):
        return _classify_with_path(low)
    return _classify_without_path(low)


def _is_author_phrasing(low: str) -> bool:
    """"Write/create a script/function/..." — checked before any path
    mention, so a file named only as an example never steals the route."""
    return (_contains_any(low, AUTHOR_VERBS)
            and _contains_any(low, AUTHOR_ARTIFACT_WORDS))


def _mentions_existing_path(request: str, root: Path) -> bool:
    """True when a token names a file that really exists under root."""
    tokens = set(_PATH_TOKEN_RE.findall(request))
    if not tokens or not root.is_dir():
        return False
    for path in root.rglob("*"):
        if path.is_file() and (path.name in tokens
                               or str(path.relative_to(root)) in tokens):
            return True
    return False


def _classify_with_path(low: str) -> tuple[str | None, str]:
    """A real path was named: a verb cue resolves the mode for free."""
    if _contains_any(low, EDIT_VERBS):
        return EDIT, "names existing path + edit verb"
    if _contains_any(low, NAVIGATE_VERBS):
        return NAVIGATE, "names existing path + navigate verb"
    if _has_compute_verb(low):
        return COMPUTE, "names existing path + compute verb"
    return None, "path named but no confident verb cue"


def _classify_without_path(low: str) -> tuple[str | None, str]:
    """No real path: require a verb cue AND a supporting noun, with the
    competing mode's cue words absent, before committing for free."""
    if _contains_any(low, NAVIGATE_VERBS) \
            and _contains_any(low, NAVIGATE_ENTITY_WORDS) \
            and not _contains_any(low, COMPUTE_DATA_WORDS):
        return NAVIGATE, "interrogative code-entity shape, no path"
    if _has_compute_verb(low) and _contains_any(low, COMPUTE_DATA_WORDS):
        return COMPUTE, "compute verb + data noun, no path"
    return None, "no confident deterministic signal"


def _has_compute_verb(low: str) -> bool:
    return _contains_any(low, COMPUTE_VERBS) or bool(_TOP_K_RE.search(low))


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    """Word-boundaried match ('sum' must not fire inside 'consume')."""
    return any(re.search(rf"\b{re.escape(phrase)}\b", text)
               for phrase in phrases)


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


def route_mode(request: str, policy: Policy,
               root: Path | None = None) -> tuple[str, str]:
    """(mode, how) for a coding request: free pre-check, else one menu."""
    mode, reason = pre_route(request, root)
    if mode is not None:
        return mode, f"pre:{reason}"
    episode = Episode(CODE_ROUTER_PREFIX, request)
    node = MenuNode("Which mode does this coding request need?",
                    [OPTION_TEXT[mode] for mode in MODES], escape=False)
    decision = policy.decide(episode, node)
    if decision.valid:
        for mode in MODES:
            if decision.value == OPTION_TEXT[mode]:
                return mode, "menu"
    return FALLBACK_MODE, "menu-fallback"


if __name__ == "__main__":
    import tempfile

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import PolicyConfig

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "parser.py").write_text("def parse():\n    return 1\n")

        assert pre_route("fix the bug in parser.py", root)[0] == EDIT
        assert pre_route("where is the Cache class implemented?", root)[0] \
            == NAVIGATE
        assert pre_route("sum the revenue column in the csv", root)[0] \
            == COMPUTE
        assert pre_route("write a script that reverses a string", root)[0] \
            == AUTHOR
        # a path named only as an illustrative example must stay author
        assert pre_route("write a script that parses log files like "
                         "parser.py", root)[0] == AUTHOR
        assert pre_route("parser.py looks weird today", root)[0] is None
        assert pre_route("hello there", root)[0] is None

    class _PickFirst:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" 1", 5, 2, 0.0, "stop")

    policy = Policy(_PickFirst(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    mode, how = route_mode("hmm, something about the code", policy, None)
    assert mode in MODES and how == "menu"
    print("smoke OK")
