"""Deterministic gates: the free judge that never calls the model.

Every check here is ast analysis or a child process run through
``threetoks.code.sandbox`` — zero tokens. That child gets isolated
interpreter flags, a temporary cwd, an allowlisted environment, a
timeout and (POSIX) process-group kill and resource limits; it can still
use the network and the user's files, so it is not a security sandbox.
The undefined-name gate is deliberately conservative (it treats every
binding anywhere in the body as defined) so it never rejects correct
code; it only catches a body that references a name that exists
nowhere. See docs/DESIGN-coding-agent.md §3.
"""
import ast
import builtins
from pathlib import Path

from threetoks.code import sandbox

BODY_INDENT = "    "
_LEADING_WS = " \t"
SUBPROCESS_TIMEOUT_S = sandbox.DEFAULT_TIMEOUT_S
TIMEOUT_MESSAGE = "TimeoutExpired"
_BUILTINS = frozenset(dir(builtins))
_KNOWN_DUNDERS = frozenset({"__file__", "__name__", "__doc__",
                            "__builtins__"})
_COMPILE_ERRORS = (SyntaxError, ValueError, OverflowError, RecursionError,
                   MemoryError)
_ERROR_MARKER = "Error:"
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_TYPE_PARAMS = tuple(getattr(ast, kind) for kind in
                     ("TypeVar", "ParamSpec", "TypeVarTuple")
                     if hasattr(ast, kind))
_NAMED_BINDERS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                  ast.ExceptHandler, ast.MatchAs, ast.MatchStar) + _TYPE_PARAMS


def docstring(text: str, indent: str = BODY_INDENT) -> str:
    """A docstring line that is a valid string literal for ANY text.

    repr() escapes embedded quotes and newlines, so a goal or contract
    containing ``\"\"\"`` cannot break out and desync the module — the
    render invariant (module always parses) depends on this.
    """
    return indent + repr(text)


def parses(source: str) -> bool:
    """True when source compiles as a Python module (null-byte safe).

    The vertical and the render invariant use this before they store
    code. compile() is stricter than ast.parse: it also rejects
    ``return`` outside a function and bad ``nonlocal``. Nothing runs.

    Args:
        source: module text, from the caller.

    Returns:
        bool: True when compile() succeeds.
    """
    try:
        compile(source, "<gate>", "exec", dont_inherit=True)
        return True
    except _COMPILE_ERRORS:
        return False


def reconstruct(name: str, args: str, completion: str) -> str:
    """Full source = the prefill def line plus the model's body text.

    The implement prefill ends at the newline after the def line, so the
    model writes the indent itself. A prefill that also gave the four
    spaces made qwen3.8 stop at once (empty answers) when the body's
    natural first token carried its own indent. The first line is
    re-indented to exactly four spaces here, so a model that writes
    three or five still lines up with later lines (or with an inserted
    docstring) instead of raising IndentationError.
    """
    return f"def {name}({args}):\n{BODY_INDENT}{completion.lstrip(_LEADING_WS)}"


def renormalize_indent(completion: str) -> str:
    """Re-anchor a completion's continuation lines at one body level.

    qwen-family models sometimes indent EVERY continuation line one column
    off (E8 live runs: a docstring then a whole body at 5 spaces against
    the prefill's 4), which ``reconstruct``'s first-line strip cannot fix.
    Dedents the continuation block so its minimum indent is exactly
    ``BODY_INDENT``, preserving deeper relative nesting.

    Only ever use this as a *fallback* after the raw completion failed to
    reconstruct: a correct body whose continuation lines are all
    legitimately deeper (e.g. one opening ``for``/``if`` line, everything
    else nested inside it) would be corrupted by an unconditional shift.
    """
    lines = completion.splitlines()
    if len(lines) <= 1:
        return completion
    first, rest = lines[0], lines[1:]
    indents = [len(l) - len(l.lstrip(" ")) for l in rest if l.strip()]
    if not indents:
        return completion
    shift = len(BODY_INDENT) - min(indents)
    if shift == 0:
        return completion
    fixed = [first]
    for line in rest:
        if not line.strip():
            fixed.append(line)
            continue
        current = len(line) - len(line.lstrip(" "))
        fixed.append(" " * max(0, current + shift) + line.lstrip(" "))
    return "\n".join(fixed)


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

    A completion that fails to reconstruct as-is gets one free second
    chance with its continuation-line indentation re-anchored (see
    ``renormalize_indent``) — recovering the E8-observed whole-body drift
    without ever touching a completion that was already valid.
    """
    source = _trimmed_function(name, args, completion)
    if source is not None:
        return source
    return _trimmed_function(name, args, renormalize_indent(completion))


def _trimmed_function(name: str, args: str, completion: str) -> str | None:
    """One reconstruct-and-trim attempt; None when it does not parse."""
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


def zero_arg_callable(args: str) -> bool:
    """True when ``def f(args)`` can be called as ``f()``.

    That covers no parameters at all AND parameters that all carry
    defaults (``def say_hello(name="World")``) — the script-entry and
    smoke-call rule, so a deliverable like that still gets its
    ``__main__`` guard and its run output.
    """
    if not args.strip():
        return True
    try:
        tree = ast.parse(f"def _probe({args}):\n    pass")
    except (SyntaxError, ValueError):
        return False
    spec = tree.body[0].args
    required = len(spec.posonlyargs) + len(spec.args) - len(spec.defaults)
    kwonly_missing = sum(1 for d in spec.kw_defaults if d is None)
    return required <= 0 and kwonly_missing == 0


def has_annotations(args: str) -> bool:
    """True when a planned argument list carries a type annotation.

    The method store uses this to know if the module needs
    ``from __future__ import annotations``. MethodStore.render_module
    calls it with each planned argument text. It parses the text as a
    def and looks for an annotation on any parameter.

    Args:
        args (str): the argument text of a plan line, e.g.
            ``"fig: plt.Figure, name: str"``, from the method record.

    Returns:
        bool: True when one parameter or more is annotated; False for
        plain names and for text that does not parse.
    """
    try:
        tree = ast.parse(f"def _probe({args}):\n    pass")
    except _COMPILE_ERRORS:
        return False
    spec = tree.body[0].args
    params = [*spec.posonlyargs, *spec.args, *spec.kwonlyargs,
              spec.vararg, spec.kwarg]
    return any(param is not None and param.annotation is not None
               for param in params)


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
    """Load-context names that are defined nowhere reachable (conservative).

    The implement gate and the edit splice gate call this to reject a
    body that uses a name that exists nowhere. Every binding anywhere in
    the tree counts as defined: assignments, all argument kinds of defs
    and lambdas, def/class names, imports, ``except ... as`` names,
    match captures, type parameters, the module dunders and
    ``__builtins__``. Names used only inside type annotations do not
    count: the rendered module never evaluates them (``from __future__
    import annotations``), so ``fig: plt.Figure`` needs no import.

    Args:
        source: function or module text, from the caller.
        allowed: extra known names, e.g. sibling method names.

    Returns:
        list[str]: sorted unknown names; [] when source does not parse.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    known = _BUILTINS | _KNOWN_DUNDERS | allowed
    for node in ast.walk(tree):
        known = known | _names_bound_by(node)
    hinted = _annotation_node_ids(tree)
    loads = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
             and id(n) not in hinted}
    return sorted(loads - known)


def _annotation_node_ids(tree: ast.AST) -> set[int]:
    """Give the ids of the ast nodes in hints that never run in a call.

    undefined_names uses this to skip names that are only type hints.
    It receives the parsed source. It collects the parameter and return
    hints of the top-level defs (the rendered module defers them with
    ``from __future__ import annotations``; an edit keeps the file's own
    header) and the hints of annotated assignments (never evaluated in a
    function body). Hints of a nested def run when the outer function
    runs, so their names stay checked.

    Args:
        tree (ast.AST): the parsed source, from undefined_names.

    Returns:
        set[int]: ``id()`` of each node that sits in a skipped hint.
    """
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    tops = [node for node in tree.body if isinstance(node, defs)]
    roots = [node.annotation for node in ast.walk(tree)
             if isinstance(node, ast.AnnAssign)]
    roots += [top.returns for top in tops]
    roots += [param.annotation for top in tops
              for param in ast.walk(top.args) if isinstance(param, ast.arg)]
    return {id(inner) for root in roots if root is not None
            for inner in ast.walk(root)}


def _names_bound_by(node: ast.AST) -> set[str]:
    """The names that one ast node binds in its scope.

    undefined_names calls this for every node of the tree.

    Args:
        node: any ast node, from ast.walk.

    Returns:
        set[str]: bound names; empty for a node that binds nothing.
    """
    bound: set[str] = set()
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        bound.add(node.id)
    if isinstance(node, _SCOPES):
        bound |= _argument_names(node.args)
    if isinstance(node, _NAMED_BINDERS) and node.name:
        bound.add(node.name)
    if isinstance(node, ast.MatchMapping) and node.rest:
        bound.add(node.rest)
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        bound |= _import_bound_names(node)
    return bound


def _argument_names(spec: ast.arguments) -> set[str]:
    """Every parameter name of a def or lambda, of all kinds.

    _names_bound_by calls this for FunctionDef, AsyncFunctionDef, Lambda.

    Args:
        spec: the node's ``args`` field.

    Returns:
        set[str]: positional-only, normal, *args, keyword-only, **kwargs.
    """
    params = [*spec.posonlyargs, *spec.args, *spec.kwonlyargs,
              spec.vararg, spec.kwarg]
    return {param.arg for param in params if param is not None}


def _import_bound_names(node: ast.Import | ast.ImportFrom) -> set[str]:
    """Every name one import statement binds.

    _names_bound_by calls this for Import and ImportFrom nodes.

    Args:
        node: the import statement.

    Returns:
        set[str]: alias or first dotted part; star imports bind nothing.
    """
    if isinstance(node, ast.Import):
        return {a.asname or a.name.split(".")[0] for a in node.names}
    return {a.asname or a.name for a in node.names if a.name != "*"}


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


def execute(source: str, cwd: Path | None = None, *,
            interpreter: str | None = None) -> tuple[bool, str]:
    """Execute source in an isolated child process; (ok, full stderr).

    The vertical, the runner and the oracle call this. The runner's
    blame mapping needs the whole traceback (its ``line N`` references),
    so this returns stderr untrimmed; ``_run`` keeps the single-line view
    the per-method gates report.

    Args:
        source: module text plus check statements, from the caller.
        cwd: working directory for the child; None gives a temp dir.
        interpreter: python to run; None uses sys.executable.

    Returns:
        tuple[bool, str]: success flag and the full stderr.
    """
    ok, _, stderr = execute_capture(source, cwd, interpreter=interpreter)
    return ok, stderr


def execute_capture(source: str, cwd: Path | None = None, *,
                    interpreter: str | None = None) -> tuple[bool, str, str]:
    """Execute source in an isolated child process; (ok, stdout, stderr).

    The runner calls this so that what the delivered script prints is
    shown, never silently discarded. The run goes through
    ``sandbox.run_python`` (isolated flags, allowlisted environment,
    SUBPROCESS_TIMEOUT_S limit).

    Args:
        source: module text plus check statements, from the caller.
        cwd: working directory for the child; None gives a temp dir.
        interpreter: python to run (the project python for user code);
            None uses the harness interpreter sys.executable.

    Returns:
        tuple[bool, str, str]: success flag, stdout, stderr; a timeout
        gives stderr TIMEOUT_MESSAGE.
    """
    run = sandbox.run_python(source, cwd, SUBPROCESS_TIMEOUT_S, interpreter)
    if run.timed_out:
        return False, run.stdout, TIMEOUT_MESSAGE
    return run.ok, run.stdout, run.stderr


def _run(source: str) -> tuple[bool, str]:
    """Execute source in an isolated child process; (ok, first_error)."""
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
    drifted = ('"""doc"""\n     start = x + 1\n     return start')  # 5-space body
    recovered = function_source("f", "x", drifted)
    assert recovered is not None and "\n     " not in recovered, recovered
    loop_body = "for n in x:\n        total += n\n        count += 1"
    kept = function_source("f", "x", loop_body)             # legit deep nesting
    assert kept is not None and "\n        total" in kept, kept
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
    weather = ("def main():\n    import requests\n"
               "    return requests.get('http://x').text\n")
    assert undefined_names(weather, set()) == []       # import binds requests
    aliased = ("def f():\n    from json import loads as parse\n"
               "    return parse('1')\n")
    assert undefined_names(aliased, set()) == []
    nested = ("def main():\n    def say_hello():\n        print('hi')\n"
              "    say_hello()\n")
    assert undefined_names(nested, set()) == []       # nested def binds
    assert zero_arg_callable("")
    assert zero_arg_callable('name="World"')
    assert zero_arg_callable("a=1, *rest, b=2, **kw")
    assert not zero_arg_callable("name")
    assert not zero_arg_callable("a, b=2")
    assert not zero_arg_callable("*, required_kw")
    assert not zero_arg_callable("((broken")
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
    ran_ok, out, err = execute_capture("print('hello')\n")
    assert ran_ok and out == "hello\n" and err == "", (ran_ok, out, err)
    for bound in ("f = lambda kv: kv[1]\n",
                  "try:\n    pass\nexcept Exception as err:\n    print(err)\n",
                  "def f():\n    def g(*, k):\n        return k\n",
                  "match 1:\n    case [x, *rest]:\n        print(x, rest)\n",
                  "print(__file__, __name__, __doc__)\n"):
        assert undefined_names(bound, set()) == [], bound
    assert not parses("return 1\n")                     # compile is stricter
    print("smoke OK")
