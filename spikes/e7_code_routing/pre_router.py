"""Deterministic, zero-model-call pre-router for code-mode classification.

Runs BEFORE any model call (see ``router_menu.py`` for the one-token
fallback). Free string + filesystem checks that only fire on a confident
signal; anything ambiguous returns ``None`` and falls through to the model
menu. Design rule from the spike brief: pre-checks must be high-PRECISION,
never high-recall — a wrong free classification is worse than one spent
model call.

Two independent signal families, checked in order:

1. path-mention: a token in the request names a file that actually exists
   under one of ``roots``, combined with a verb cue -> edit/navigate/compute.
   A path mention with no matching verb cue is left ambiguous (falls
   through) rather than guessed.
2. phrasing-only: strong imperative "write a script that..." (author) or
   interrogative "where is X defined" (navigate) / "parse this and tell me
   the total" (compute) shapes that need no real file, gated by requiring
   a supporting word AND the absence of a competing mode's cue words.
"""
import re
from pathlib import Path

NAVIGATE = "navigate"
EDIT = "edit"
COMPUTE = "compute"
AUTHOR = "author"
MODES = (NAVIGATE, EDIT, COMPUTE, AUTHOR)

EDIT_VERBS = ("fix", "rename", "refactor", "debug", "patch", "remove",
             "delete", "modify", "update", "change", "clean up", "add",
             "correct", "optimize", "simplify", "improve", "reformat")
NAVIGATE_VERBS = ("where is", "where does", "where's", "what does",
                  "what is", "explain", "show me", "find the definition",
                  "locate", "how does", "which line", "which function",
                  "which method", "which class", "walk me through",
                  "describe")
COMPUTE_VERBS = ("parse", "sum", "count", "compute", "calculate", "extract",
                 "average", "how many", "total", "which ip",
                 "most common", "median", "filter", "group by")
# bare "top" is too common a word on its own ("at the top", "top-level
# function"); only "top <number>" is a confident top-K compute signal.
_TOP_K_RE = re.compile(r"\btop\s*\d+\b")
AUTHOR_VERBS = ("write", "create", "build", "generate", "make me", "author")

NAVIGATE_ENTITY_WORDS = ("function", "method", "class", "variable",
                         "module", "file", "defined", "definition",
                         "declared", "implemented", "logic", "symbol")
COMPUTE_DATA_WORDS = ("csv", "log", "column", "data", "rows", "lines",
                     "file")
AUTHOR_ARTIFACT_WORDS = ("script", "program", "module", "function", "class",
                         "tool", "cli", "library", "package")

_PATH_TOKEN_RE = re.compile(r"[\w][\w./-]*\.\w+|[\w][\w/-]*/[\w./-]+")


def _mentioned_paths(request: str, roots: list[Path]) -> list[Path]:
    """Existing files under roots whose name or relative path is named."""
    tokens = set(_PATH_TOKEN_RE.findall(request))
    hits = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            rel = str(path.relative_to(root))
            if path.name in tokens or rel in tokens:
                hits.append(path)
    return hits


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    """Whole-word/phrase match (word-boundaried, so 'sum' != 'consume').

    Plain substring containment would let "parse" match inside "parser.py"
    or "add" match inside "address" — regex word boundaries avoid that.
    """
    return any(re.search(rf"\b{re.escape(phrase)}\b", text)
               for phrase in phrases)


def _has_compute_verb(low: str) -> bool:
    """A compute-verb cue, including the "top <number>" top-K shape."""
    return _contains_any(low, COMPUTE_VERBS) or bool(_TOP_K_RE.search(low))


def _classify_with_path(low: str, paths: list[Path]) -> tuple[str | None, str]:
    """A real path was named: a verb cue resolves the mode, else fall through."""
    name = paths[0].name
    if _contains_any(low, EDIT_VERBS):
        return EDIT, f"names existing path '{name}' + edit verb"
    if _contains_any(low, NAVIGATE_VERBS):
        return NAVIGATE, f"names existing path '{name}' + navigate verb"
    if _has_compute_verb(low):
        return COMPUTE, f"names existing path '{name}' + compute verb"
    return None, f"path '{name}' named but no confident verb cue"


def _is_author_phrasing(low: str) -> bool:
    """"Write/create a script/function/..." — a new-artifact deliverable.

    Checked ahead of any path mention: a file named only as an illustrative
    example ("like access.log") must not steer this to compute/edit.
    """
    return (_contains_any(low, AUTHOR_VERBS)
            and _contains_any(low, AUTHOR_ARTIFACT_WORDS))


def _classify_without_path(low: str) -> tuple[str | None, str]:
    """No real path named: require a verb cue AND a supporting word, with
    the competing modes' cue words absent, before committing for free."""
    is_navigate = (_contains_any(low, NAVIGATE_VERBS)
                  and _contains_any(low, NAVIGATE_ENTITY_WORDS)
                  and not _contains_any(low, COMPUTE_DATA_WORDS))
    if is_navigate:
        return NAVIGATE, "interrogative code-entity shape, no path"

    is_compute = _has_compute_verb(low) and _contains_any(low, COMPUTE_DATA_WORDS)
    if is_compute:
        return COMPUTE, "compute verb + data noun, no path"

    return None, "no confident deterministic signal"


def pre_classify(request: str, roots: list[Path]) -> tuple[str | None, str]:
    """Return (mode, reason) or (None, reason) meaning: ask the model.

    ``roots`` are directories searched for a mentioned, really-existing
    path (e.g. the spike's ``fixtures/`` folder). Every branch is a plain
    string/filesystem check — no model call anywhere in this function.
    """
    low = request.lower()
    if _is_author_phrasing(low):
        return AUTHOR, "imperative write/create + new-artifact noun"
    paths = _mentioned_paths(request, roots)
    if paths:
        return _classify_with_path(low, paths)
    return _classify_without_path(low)


if __name__ == "__main__":
    fixtures = Path(__file__).parent / "fixtures"
    roots = [fixtures]

    mode, reason = pre_classify("fix the bug in parser.py", roots)
    assert mode == EDIT, (mode, reason)

    mode, reason = pre_classify("where is the Cache class implemented in "
                                "pkg/utils.py?", roots)
    assert mode == NAVIGATE, (mode, reason)

    mode, reason = pre_classify("what's the sum of the revenue column in "
                                "sales.csv?", roots)
    assert mode == COMPUTE, (mode, reason)

    mode, reason = pre_classify("write a script that reverses a string",
                                roots)
    assert mode == AUTHOR, (mode, reason)

    mode, reason = pre_classify("write a script that parses log files like "
                                "access.log and reports the top 5 IPs",
                                roots)
    assert mode == AUTHOR, (mode, reason)  # illustrative path must not steer

    mode, reason = pre_classify("parser.py looks weird today", roots)
    assert mode is None, (mode, reason)  # path named, no verb cue

    mode, reason = pre_classify("hello there", roots)
    assert mode is None, (mode, reason)

    print("smoke OK")
