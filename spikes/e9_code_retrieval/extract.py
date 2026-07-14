"""Candidate extraction: fetched source text -> gated, renamed snippets.

Pipeline per fetched "file" (a raw .py file, or one <pre>/<code> block
pulled out of an HTML page — see fetchers.py):

  1. ``ast.parse`` the whole file standalone (gate 1). A file that fails
     to parse contributes zero candidates and is logged as such.
  2. Walk top-level FunctionDefs, name-match each against the spec
     (exact -> substring -> fuzzy, in that tier order per task spec §b).
  3. For every matched def, ``assemble`` a standalone, renamed snippet:
     pull in any top-level import/helper-function/constant the body
     actually needs (one hop), reject on an unresolvable name, an
     unwhitelisted import, or a forbidden-builtin touch (safety_gates.py).
  4. Re-``ast.parse`` the assembled snippet (gate: "parses standalone"
     again, now for the *assembled* unit, not just the original file).

Nothing here runs the candidate — that is judge.py's job, in a subprocess.
"""
import ast
import difflib
import re
from dataclasses import dataclass, field

from safety_gates import (
    FORBIDDEN_BUILTINS, SAFE_IMPORT_WHITELIST, has_star_import, node_source,
    rename_def, top_level_assigns, top_level_functions, top_level_imports,
    uses_forbidden_builtin,
)

MIN_FUZZY_RATIO = 0.72
MAX_HELPER_HOPS = 2   # how many sibling-helper levels we will chase


def _normalize(name: str) -> str:
    return re.sub(r"[_\-]", "", name).lower()


def name_match_tier(candidate_name: str, spec) -> str | None:
    """'exact' / 'substring' / 'fuzzy' / None, matching candidate to spec.

    Checked in that tier order (task §b: "exact match first, then
    substring/fuzzy") against the spec's canonical name plus every alias.
    """
    targets = (spec.name,) + tuple(spec.aliases)
    cand_norm = _normalize(candidate_name)
    norm_targets = [_normalize(t) for t in targets]
    if cand_norm in norm_targets:
        return "exact"
    if any(t and (t in cand_norm or cand_norm in t) for t in norm_targets):
        return "substring"
    best = max((difflib.SequenceMatcher(None, cand_norm, t).ratio()
                for t in norm_targets), default=0.0)
    if best >= MIN_FUZZY_RATIO:
        return f"fuzzy:{best:.2f}"
    return None


@dataclass
class Candidate:
    """One name-matched def, before or after gating."""

    source_url: str
    original_name: str
    tier: str
    node: ast.AST = field(repr=False)
    file_lines: list = field(repr=False)
    snippet: str | None = None       # set once assembled+gated OK
    rejected: str | None = None      # gate-failure reason, else None


def find_matches(file_text: str, spec, source_url: str) -> tuple:
    """Parse ``file_text``; return (candidates, parsed_ok).

    ``candidates`` is every top-level def whose name matches the spec,
    best tier first. ``parsed_ok`` is False when the whole file failed
    gate 1 (ast.parse) — the caller logs that as a zero-candidate file,
    not silently.
    """
    try:
        tree = ast.parse(file_text)
    except (SyntaxError, ValueError):
        return [], False
    lines = file_text.splitlines()
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    tier_rank = {"exact": 0, "substring": 1}
    found = []
    for node in tree.body:
        if not isinstance(node, defs):
            continue
        tier = name_match_tier(node.name, spec)
        if tier is None:
            continue
        found.append(Candidate(source_url, node.name, tier, node, lines))
    found.sort(key=lambda c: tier_rank.get(c.tier, 2))
    return found, True


def _resolve_helper(name: str, tree_funcs: dict, tree_assigns: dict,
                     imports: dict, lines: list, depth: int) -> tuple:
    """Best-effort resolve one missing name to source text to prepend.

    Returns (source_or_None, rejection_or_None). A sibling top-level
    function is pulled in (recursively, up to MAX_HELPER_HOPS) rather than
    just failing the gate — real snippets often factor a classic function
    into a small helper (e.g. caesar's ``shift_char``).
    """
    if name in imports:
        module = imports[name]
        if module not in SAFE_IMPORT_WHITELIST:
            return None, f"unsafe_import:{module}"
        return f"import {module}", None
    if name in tree_assigns and depth > 0:
        return node_source(lines, tree_assigns[name]), None
    if name in tree_funcs and depth > 0:
        helper_node = tree_funcs[name]
        bad = uses_forbidden_builtin(helper_node)
        if bad:
            return None, f"forbidden_builtin:{bad}"
        return node_source(lines, helper_node), None
    return None, None  # genuinely unresolved


def assemble(candidate: Candidate, canonical_name: str,
             file_tree: ast.Module) -> None:
    """Fill ``candidate.snippet`` xor ``candidate.rejected`` in place.

    Builds a standalone module: whitelisted imports the body needs, any
    sibling helper functions/constants it calls, then the target def
    renamed to ``canonical_name`` so the anchor assert can call it.
    """
    node = candidate.node
    bad = uses_forbidden_builtin(node)
    if bad:
        candidate.rejected = f"forbidden_builtin:{bad}"
        return
    if has_star_import(file_tree):
        candidate.rejected = "star_import_in_file"
        return

    imports = top_level_imports(file_tree)
    tree_funcs = top_level_functions(file_tree)
    tree_assigns = top_level_assigns(file_tree)

    target_source = node_source(candidate.file_lines, node)
    prelude_parts = []
    # The target's own (pre-rename) name is already emitted as the target
    # itself; a recursive call (fibonacci calling fibonacci) would
    # otherwise be treated as an unresolved sibling and re-pull an
    # identical duplicate copy of the same def (harmless but untidy —
    # caught live when the fibonacci fixture's accepted snippet came back
    # with the function defined twice).
    seen_names = {candidate.original_name}
    frontier = [(name, MAX_HELPER_HOPS) for name in
                _free_names(node, node.args)]
    while frontier:
        name, depth = frontier.pop()
        if name in seen_names:
            continue
        seen_names.add(name)
        src, reason = _resolve_helper(name, tree_funcs, tree_assigns,
                                       imports, candidate.file_lines, depth)
        if reason:
            candidate.rejected = reason
            return
        if src is None:
            continue  # not resolvable here; may be a builtin, checked below
        prelude_parts.append(src)
        if name in tree_funcs and depth > 0:
            helper_free = _free_names(tree_funcs[name], tree_funcs[name].args)
            frontier.extend((n, depth - 1) for n in helper_free)

    assembled = "\n".join(prelude_parts + [
        rename_def(target_source, candidate.original_name, canonical_name)])
    try:
        ast.parse(assembled)
    except (SyntaxError, ValueError) as exc:
        candidate.rejected = f"assembled_does_not_parse:{exc}"
        return
    candidate.snippet = assembled


_BUILTINS_NAMES = frozenset(dir(__builtins__)) if isinstance(
    __builtins__, dict) else frozenset(dir(__builtins__))


def _free_names(func_node, args_node) -> set:
    """Load-context names in the function body not bound locally/by args.

    Deliberately simple (no full scope resolution) — a name bound
    ANYWHERE inside (assignment, comprehension target, nested def) counts
    as local, matching the conservative style of
    threetoks/code/gates.undefined_names. False negatives here just mean
    we skip pulling in a helper that turns out to be needed, which fails
    safe (the assembled snippet then fails to execute -> honest miss),
    never fails unsafe.
    """
    bound = {a.arg for a in list(args_node.args) + list(args_node.posonlyargs)
             + list(args_node.kwonlyargs)}
    if args_node.vararg:
        bound.add(args_node.vararg.arg)
    if args_node.kwarg:
        bound.add(args_node.kwarg.arg)
    for n in ast.walk(func_node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and n is not func_node:
            bound.add(n.name)
    loads = {n.id for n in ast.walk(func_node)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loads - bound - _BUILTINS_NAMES


if __name__ == "__main__":
    from specs import Spec

    spec = Spec(name="roman_to_int", aliases=("romanToInt",),
                anchor='assert roman_to_int("MCMXCIV") == 1994',
                queries=(), classic=True)
    fixture = (
        "ROMANS = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, "
        "'M': 1000}\n\n"
        "def romanToInt(s):\n"
        "    total = 0\n"
        "    for i in range(len(s)):\n"
        "        v = ROMANS[s[i]]\n"
        "        if i + 1 < len(s) and ROMANS[s[i + 1]] > v:\n"
        "            total -= v\n"
        "        else:\n"
        "            total += v\n"
        "    return total\n"
    )
    candidates, parsed_ok = find_matches(fixture, spec, "fixture://roman.py")
    assert parsed_ok
    assert len(candidates) == 1 and candidates[0].tier == "exact", \
        candidates
    tree = ast.parse(fixture)
    assemble(candidates[0], spec.name, tree)
    assert candidates[0].rejected is None, candidates[0].rejected
    assert "def roman_to_int(" in candidates[0].snippet
    assert "ROMANS" in candidates[0].snippet
    ns = {}
    exec(candidates[0].snippet, ns)
    assert ns["roman_to_int"]("MCMXCIV") == 1994

    unsafe_fixture = ("import os\n\n"
                      "def caesar_encode(text, shift):\n"
                      "    os.system('echo pwned')\n"
                      "    return text\n")
    spec2 = Spec(name="caesar_encode", aliases=(), anchor="assert True",
                queries=(), classic=True)
    cands2, ok2 = find_matches(unsafe_fixture, spec2, "fixture://unsafe.py")
    assert ok2 and len(cands2) == 1
    assemble(cands2[0], spec2.name, ast.parse(unsafe_fixture))
    assert cands2[0].rejected and cands2[0].rejected.startswith(
        "unsafe_import"), cands2[0].rejected

    bad_syntax = "def f(:\n    pass\n"
    cands3, ok3 = find_matches(bad_syntax, spec, "fixture://bad.py")
    assert cands3 == [] and ok3 is False
    print("smoke OK")
