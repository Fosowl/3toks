"""Free deterministic gates for FOREIGN code (retrieved off the internet).

This is deliberately stricter than ``threetoks/code/gates.py``: that gate
trusts code the harness itself planned and generated one method at a time.
Here the source is whatever a stranger's repo contains, so the import
whitelist is much tighter and there is an extra forbidden-builtins scan
that the greenfield vertical does not need (its model never calls
``eval``/``exec``/``open`` because it never had a reason to plan them).

Everything here is ``ast`` analysis only — zero network, zero subprocess,
zero model calls. The subprocess execution step lives in judge.py and is
the ONLY place fetched code actually runs, and only ever in a fresh
``python3 -`` process with a timeout (see judge.py / task instructions §c).
"""
import ast

# Tight stdlib whitelist: pure computation, no filesystem/process/network/
# reflection access. Excludes (per task spec) os, sys, subprocess, socket,
# shutil, ctypes, importlib, and anything reflection-flavoured.
SAFE_IMPORT_WHITELIST = frozenset({
    "math", "re", "string", "functools", "itertools", "collections",
    "operator", "textwrap", "unicodedata", "statistics", "fractions",
    "decimal", "bisect", "heapq", "array", "cmath",
})

# Builtins that let sandboxed code escape the sandbox or hit the outside
# world even with no unsafe import in sight. Blocked regardless of context.
FORBIDDEN_BUILTINS = frozenset({
    "eval", "exec", "open", "__import__", "getattr", "setattr", "delattr",
    "compile", "globals", "locals", "vars", "input", "exit", "quit",
    "breakpoint", "memoryview", "help",
})


class Rejected(Exception):
    """Carries a short machine-readable reason a candidate was gated out."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def top_level_imports(tree: ast.Module) -> dict:
    """Map bound name -> root module for every top-level import statement.

    ``import os.path as p`` binds ``p`` -> ``os``; ``from math import sqrt``
    binds ``sqrt`` -> ``math``. A star import binds nothing resolvable, so
    callers must reject it separately (see ``has_star_import``).
    """
    bound = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                bound[bound_name] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound_name = alias.asname or alias.name
                bound[bound_name] = node.module.split(".")[0]
    return bound


def has_star_import(tree: ast.Module) -> bool:
    """True if any top-level ``from X import *`` is present."""
    return any(isinstance(node, ast.ImportFrom)
               and any(a.name == "*" for a in node.names)
               for node in tree.body)


def top_level_functions(tree: ast.Module) -> dict:
    """Map name -> node for every top-level (sync or async) function def."""
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    return {n.name: n for n in tree.body if isinstance(n, defs)}


def top_level_assigns(tree: ast.Module) -> dict:
    """Map simple assigned name -> node for top-level ``NAME = ...`` lines.

    Only plain ``Name`` targets are considered (a classic "lookup table"
    constant like ``ROMAN = {'I': 1, ...}``); tuple/attribute targets are
    skipped as out of scope for a free-function snippet.
    """
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node
    return out


def node_source(source_lines: list, node: ast.AST) -> str:
    """Slice a node's own source lines (1-based lineno, inclusive end)."""
    return "\n".join(source_lines[node.lineno - 1:node.end_lineno])


def uses_forbidden_builtin(node: ast.AST) -> str | None:
    """Return the offending name if the subtree touches a forbidden builtin.

    Checks both bare names (``eval(...)``) and dunder-ish attribute access
    (``x.__class__``) as a best-effort sandbox-escape trip wire — see the
    REPORT's safety section for what this does NOT catch.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in FORBIDDEN_BUILTINS:
            return n.id
        if isinstance(n, ast.Attribute) and n.attr.startswith("__") \
                and n.attr.endswith("__") and n.attr not in {"__name__"}:
            return n.attr
    return None


def rename_def(source: str, old_name: str, new_name: str) -> str:
    """Rename ``def old_name(`` -> ``def new_name(`` on its own first line.

    Foreign code is rarely named exactly what the spec calls it (E9's
    whole premise is fuzzy name matching), but the anchor assert always
    calls the spec's canonical name, so the accepted def must wear that
    name before it is executed.
    """
    lines = source.splitlines()
    prefix_sync, prefix_async = f"def {old_name}(", f"async def {old_name}("
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(prefix_sync):
            indent = line[:len(line) - len(stripped)]
            lines[i] = f"{indent}def {new_name}(" + stripped[len(prefix_sync):]
            return "\n".join(lines)
        if stripped.startswith(prefix_async):
            indent = line[:len(line) - len(stripped)]
            lines[i] = (f"{indent}async def {new_name}("
                        + stripped[len(prefix_async):])
            return "\n".join(lines)
    return source  # unchanged if the def line was not found verbatim


if __name__ == "__main__":
    tree = ast.parse("import os\nfrom math import sqrt as s\ndef f():\n    pass\n")
    imports = top_level_imports(tree)
    assert imports == {"os": "os", "s": "math"}, imports
    assert not has_star_import(tree)
    assert has_star_import(ast.parse("from os import *\n"))
    funcs = top_level_functions(tree)
    assert set(funcs) == {"f"}
    assigns = top_level_assigns(ast.parse("X = 1\nY, Z = 2, 3\n"))
    assert set(assigns) == {"X"}, assigns  # tuple target skipped
    danger = ast.parse("def f():\n    return eval('1')\n")
    assert uses_forbidden_builtin(danger.body[0]) == "eval"
    safe = ast.parse("def f(x):\n    return x + 1\n")
    assert uses_forbidden_builtin(safe.body[0]) is None
    renamed = rename_def("def romanToInt(s):\n    return 0\n", "romanToInt",
                          "roman_to_int")
    assert renamed.startswith("def roman_to_int(s):"), renamed
    print("smoke OK")
