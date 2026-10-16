"""Name analysis for the edit vertical: bound names and missing helpers.

``module_level_names`` gives what a file binds at module level, and
``find_missing_helper`` names a helper the target calls but nothing
defines, with a parameter list built from the call (zero model calls).
A file with a star import never has a missing helper: any unknown name
can come from that import.
"""
import ast

from threetoks.code import gates
from threetoks.code.edit.spans import FUNC_TYPES, function_text, parse_source

PARAM_PREFIX = "arg"
STAR_PARAM = "*args"
DOUBLE_STAR_PARAM = "**kwargs"
STAR_NAME = "*"
_BLOCK_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")


def module_level_names(source: str) -> set[str]:
    """Give every name a file binds at module level.

    The edit vertical gives these names to find_missing_helper and to the
    splice gates as known names. Imports, defs, classes and assignments
    count, also inside top-level if/try/for/with blocks and in tuple
    targets. So an imported or constant name is never a missing helper.

    Args:
        source: str, the file text.

    Returns:
        set[str]: the bound names, empty when the file does not parse.
    """
    tree = parse_source(source)
    if tree is None:
        return set()
    names: set[str] = set()
    for node in _module_statements(tree.body):
        names |= _bound_names(node)
    return names


def _module_statements(body: list):
    """Yield the module-level statements, also those in compound blocks.

    The bodies of defs and classes are not module level, so they are
    skipped.

    Args:
        body: list, statements (or except handlers / match cases).

    Returns:
        Iterator[ast.AST]: each statement in source order.
    """
    for node in body:
        yield node
        if isinstance(node, FUNC_TYPES + (ast.ClassDef,)):
            continue
        for field in _BLOCK_FIELDS:
            yield from _module_statements(getattr(node, field, []))


def _bound_names(node: ast.AST) -> set[str]:
    """Give the names one statement binds in its own scope.

    Args:
        node: ast.AST, a module-level statement or except handler.

    Returns:
        set[str]: the bound names.
    """
    if isinstance(node, ast.Import):
        return {a.asname or a.name.split(".")[0] for a in node.names}
    if isinstance(node, ast.ImportFrom):
        return {a.asname or a.name for a in node.names if a.name != "*"}
    if isinstance(node, FUNC_TYPES + (ast.ClassDef,)):
        return {node.name}
    if isinstance(node, ast.ExceptHandler):
        return {node.name} if node.name else set()
    return {name.id for target in _targets(node) for name in ast.walk(target)
            if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)}


def _targets(node: ast.AST) -> list[ast.AST]:
    """Give the assignment targets of one statement.

    Args:
        node: ast.AST, a module-level statement.

    Returns:
        list[ast.AST]: targets of assignments, for loops and with items.
    """
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For,
                         ast.AsyncFor)):
        return [node.target]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [item.optional_vars for item in node.items
                if item.optional_vars is not None]
    return []


def has_star_import(source: str) -> bool:
    """Tell if a file has a "from module import *" statement.

    find_missing_helper and the splice gates use it. With a star import
    any unknown name can come from that module, so the undefined-name
    test cannot be trusted for that file.

    Args:
        source: str, the file text.

    Returns:
        bool: True when the file parses and has a star import.
    """
    tree = parse_source(source)
    return tree is not None and _has_star(tree)


def _has_star(tree: ast.AST) -> bool:
    """Tell if a parsed tree has a star import.

    Args:
        tree: ast.AST, a parsed file.

    Returns:
        bool: True when an ImportFrom imports "*".
    """
    return any(isinstance(node, ast.ImportFrom)
               and any(alias.name == STAR_NAME for alias in node.names)
               for node in ast.walk(tree))


def find_missing_helper(file_source: str, lineno: int,
                        known_names: set[str]) -> tuple[str, str] | None:
    """Name a helper that a function calls but nothing defines.

    The edit vertical calls this before insert-after, so the model only
    writes the helper body. It uses the undefined-name gate, then builds
    the parameters from the first call: positional arguments become
    arg1..argN, a starred argument becomes *args, keyword arguments
    become parameters of the same name, and **mapping becomes **kwargs.
    A file with a star import gives None: the name can come from it.

    Args:
        file_source: str, the file text (LF), from the symbol index.
        lineno: int, the def line of the target (Symbol.lineno).
        known_names: set[str], names the file defines (module_level_names).

    Returns:
        tuple[str, str] | None: (helper name, parameter text), or None.
    """
    tree = parse_source(file_source)
    if tree is None or _has_star(tree):
        return None
    target = next((node for node in ast.walk(tree) if isinstance(
        node, FUNC_TYPES) and node.lineno == lineno), None)
    if target is None:
        return None
    missing = gates.undefined_names(function_text(file_source, target),
                                    known_names)
    for node in ast.walk(target):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in missing:
            return node.func.id, _helper_params(node)
    return None


def _helper_params(call: ast.Call) -> str:
    """Build a parameter list that accepts one call.

    Args:
        call: ast.Call, the first call of the missing helper.

    Returns:
        str: parameter text such as "arg1, *args, key, **kwargs".
    """
    keywords = [kw.arg for kw in call.keywords if kw.arg is not None]
    params = _positional_params(call.args, set(keywords)) + keywords
    if any(kw.arg is None for kw in call.keywords):
        params.append(DOUBLE_STAR_PARAM)
    return ", ".join(params)


def _positional_params(args: list[ast.expr], taken: set[str]) -> list[str]:
    """Name the positional parameters for the arguments of one call.

    Args:
        args: list[ast.expr], the positional arguments of the call.
        taken: set[str], keyword names the call also uses.

    Returns:
        list[str]: arg1..argN, ending with *args after a starred argument.
    """
    params: list[str] = []
    number = 0
    for arg in args:
        if isinstance(arg, ast.Starred):
            return params + [STAR_PARAM]
        number += 1
        while f"{PARAM_PREFIX}{number}" in taken:
            number += 1
        params.append(f"{PARAM_PREFIX}{number}")
    return params


if __name__ == "__main__":
    missing_src = ("def average(nums):\n    total = 0\n"
                   "    for n in nums:\n        total = _safe_add(total, n)\n"
                   "    return total / len(nums)\n")
    assert find_missing_helper(missing_src, 1, {"average"}) == \
        ("_safe_add", "arg1, arg2")
    starred = "from m import *\n" + missing_src
    assert has_star_import(starred)
    assert find_missing_helper(starred, 2, {"average"}) is None
    assert module_level_names("import os\nif os:\n    a, b = 1, 2\n") == \
        {"os", "a", "b"}
    print("smoke OK")
