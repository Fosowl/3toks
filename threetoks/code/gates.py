"""Deterministic gates: the free judge that never calls the model.

Every check here is ast analysis or a sandboxed subprocess — zero tokens.
The undefined-name gate is deliberately conservative (it treats every
Store-context name anywhere in the body as defined) so it never rejects
correct code; it only catches a body that references a name that exists
nowhere. See docs/DESIGN-coding-agent.md §3.
"""
import ast
import builtins
import subprocess

BODY_INDENT = "    "
_LEADING_WS = " \t"
SUBPROCESS_TIMEOUT_S = 5
_BUILTINS = frozenset(dir(builtins))
_ERROR_MARKER = "Error:"


def docstring(text: str, indent: str = BODY_INDENT) -> str:
    """A docstring line that is a valid string literal for ANY text.

    repr() escapes embedded quotes and newlines, so a goal or contract
    containing ``\"\"\"`` cannot break out and desync the module — the
    render invariant (module always parses) depends on this.
    """
    return indent + repr(text)


def parses(source: str) -> bool:
    """True when source is syntactically valid Python (null-byte safe)."""
    try:
        ast.parse(source)
        return True
    except (SyntaxError, ValueError):
        return False


def reconstruct(name: str, args: str, completion: str) -> str:
    """Full source = the prefill def line plus the model's body text.

    The prefill already supplies the first line's four-space indent, so a
    stray leading space from the model is stripped — otherwise the first
    body line lands at five spaces and later lines (or an inserted
    docstring) at four, an IndentationError that corrupts the module.
    """
    return f"def {name}({args}):\n{BODY_INDENT}{completion.lstrip(_LEADING_WS)}"


def function_def(source: str, name: str) -> ast.AST | None:
    """The top-level def named `name` (sync or async), or None."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):     # ValueError: null bytes
        return None
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    return next((node for node in tree.body
                 if isinstance(node, defs) and node.name == name), None)


def function_source(name: str, args: str, completion: str) -> str | None:
    """Reconstruct the body and trim to only the named function's lines.

    A model completion can run past the function into top-level code
    (stray ``x = {...}`` assignments). Slicing to the def's own line span
    keeps each method self-contained, so it cannot inject module globals a
    sibling would then silently depend on, and the undefined-name gate
    stays sound.
    """
    raw = reconstruct(name, args, completion)
    node = function_def(raw, name)
    if node is None:
        return None
    return "\n".join(raw.splitlines()[node.lineno - 1:node.end_lineno])


def is_placeholder(source: str, name: str) -> bool:
    """True when the body is a non-implementation the model must not ship.

    A body that is empty, or only a docstring, or exactly ``pass`` / ``...``
    / ``raise NotImplementedError`` — or the return-shaped dodges ``return``,
    ``return None``, ``return ...``, ``return NotImplementedError`` — is the
    model dodging the work. Such a body passes every other gate (it parses
    and imports), so it needs its own check, else it lands as a fake "done"
    method.
    """
    node = function_def(source, name)
    if node is None:
        return False
    body = node.body
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]                       # drop the docstring
    if not body:
        return True
    return len(body) == 1 and _is_empty_statement(body[0])


def _is_empty_statement(node: ast.AST) -> bool:
    """True for pass, bare ..., raising or returning NotImplementedError,
    and the no-op returns (bare / None / Ellipsis)."""
    if isinstance(node, ast.Pass):
        return True
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
            and node.value.value is Ellipsis:
        return True
    if isinstance(node, ast.Raise):
        return _is_not_implemented(node.exc)
    if isinstance(node, ast.Return):
        return node.value is None or _is_noop_constant(node.value) \
            or _is_not_implemented(node.value)
    return False


def _is_not_implemented(node: ast.AST | None) -> bool:
    """True for ``NotImplementedError`` bare or called."""
    if isinstance(node, ast.Call):
        node = node.func
    return isinstance(node, ast.Name) and node.id == "NotImplementedError"


def _is_noop_constant(node: ast.AST) -> bool:
    """True for the literal ``None`` or ``...``."""
    return isinstance(node, ast.Constant) \
        and (node.value is None or node.value is Ellipsis)


def signature_matches(source: str, name: str, args: str) -> bool:
    """True when the parsed def's argument names match the plan's."""
    node = function_def(source, name)
    if node is None:
        return False
    got = [arg.arg for arg in node.args.args]
    want = [a.strip().split(":")[0].split("=")[0].strip()
            for a in args.split(",") if a.strip()]
    return got == want


def undefined_names(source: str, allowed: set[str]) -> list[str]:
    """Load-context names that are defined nowhere reachable (conservative)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    defined = {n.id for n in ast.walk(tree)
               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    defined |= {a.arg for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                for a in n.args.args}
    known = defined | _BUILTINS | allowed
    loads = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(loads - known)


def ensure_docstring(source: str, contract: str) -> str:
    """Return source with a contract docstring inserted if it has none.

    Falls back to the untouched source if injection would not parse, so
    this never hands back broken code (belt-and-suspenders atop repr()).
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return source
    node = tree.body[0] if tree.body else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return source
    if ast.get_docstring(node) is not None:
        return source
    lines = source.splitlines()
    header = node.lineno                       # 1-based def line count
    injected = "\n".join(lines[:header] + [docstring(contract)] + lines[header:])
    return injected if parses(injected) else source


def execute(source: str) -> tuple[bool, str]:
    """Execute source in a fresh python subprocess; (ok, full stderr).

    The runner's blame mapping needs the whole traceback (its ``line N``
    references), so this returns stderr untrimmed; ``_run`` keeps the
    single-line view the per-method gates report.
    """
    try:
        done = subprocess.run(["python3", "-"], input=source, text=True,
                              capture_output=True, timeout=SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    return done.returncode == 0, done.stderr


def _run(source: str) -> tuple[bool, str]:
    """Execute source in a fresh python subprocess; (ok, first_error)."""
    ok, stderr = execute(source)
    return ok, first_error(stderr)


def first_error(stderr: str) -> str:
    """Last traceback line naming an exception, else the last line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in reversed(lines):
        if _ERROR_MARKER in line:
            return line
    return lines[-1] if lines else ""


def import_ok(module_source: str) -> tuple[bool, str]:
    """True when the rendered module imports/defines without raising."""
    return _run(module_source)


def execute_asserts(module_source: str,
                    asserts: list[str]) -> tuple[bool, str]:
    """Run asserts against the module; (all passed, full stderr)."""
    if not asserts:
        return True, ""
    return execute(module_source + "\n" + "\n".join(asserts))


def run_asserts(module_source: str, asserts: list[str]) -> tuple[bool, str]:
    """Run asserts against the module; (all passed, first error line)."""
    ok, stderr = execute_asserts(module_source, asserts)
    return ok, first_error(stderr)


if __name__ == "__main__":
    src = reconstruct("clamp", "value, low, high", "return max(low, min(high, value))")
    assert function_def(src, "clamp") is not None
    stray = reconstruct("f", "x", " return x")           # stray leading space
    assert ast.parse(ensure_docstring(stray, "doc")) and stray.count("\n    ") == 1
    trimmed = function_source("f", "x", "return x\n\nLEAK = {1: 2}")
    assert trimmed == "def f(x):\n    return x", trimmed   # top-level code dropped
    assert undefined_names(trimmed, set()) == []           # gate now sound
    assert is_placeholder("def f(x):\n    pass", "f")
    assert is_placeholder("def f(x):\n    raise NotImplementedError", "f")
    assert is_placeholder("def f(x):\n    raise NotImplementedError('todo')", "f")
    assert is_placeholder("def f(x):\n    ...", "f")
    assert is_placeholder("def f(x):\n    'just a docstring'", "f")
    assert is_placeholder("def f(x):\n    return NotImplementedError", "f")
    assert is_placeholder("def f(x):\n    return NotImplementedError('x')", "f")
    assert is_placeholder("def f(x):\n    return", "f")
    assert is_placeholder("def f(x):\n    return None", "f")
    assert is_placeholder("def f(x):\n    return ...", "f")
    assert not is_placeholder("def f(x):\n    return x * 2", "f")
    assert not is_placeholder("def f(x):\n    'doc'\n    return x", "f")
    assert not is_placeholder("def f(x):\n    return 0", "f")
    assert not is_placeholder("def f(x):\n    print(x)\n    return None", "f")
    assert signature_matches(src, "clamp", "value, low, high")
    assert not signature_matches(src, "clamp", "value, low")
    assert undefined_names("def f(x):\n    return helper(x)\n", set()) == ["helper"]
    assert undefined_names("def f(x):\n    return helper(x)\n", {"helper"}) == []
    documented = ensure_docstring("def f(x):\n    return x\n", "return x")
    assert "return x" in documented and ast.parse(documented)
    hostile = ensure_docstring("def f(x):\n    return x\n", 'has """ triple')
    assert ast.parse(hostile), "triple-quote contract must not break the source"
    assert function_def("return x\x00", "f") is None      # null byte, no crash
    module = "def add(a, b):\n    return a + b\n"
    assert run_asserts(module, ["assert add(1, 2) == 3"]) == (True, "")
    bad_ok, bad_err = run_asserts(module, ["assert add(1, 2) == 4"])
    assert not bad_ok and "AssertionError" in bad_err, (bad_ok, bad_err)
    assert import_ok("def f():\n    return 1\n")[0]
    assert not import_ok("import nonexistent_pkg_xyz\n")[0]
    print("smoke OK")
