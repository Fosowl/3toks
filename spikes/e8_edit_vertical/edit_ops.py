"""Edit machinery owned by the harness: splice a span, gate, resample.

Three operations, all applied to an ast-selected span (1-based, inclusive
line numbers): replace-span, insert-after-span, delete-span. Delete never
calls the model (the harness already knows the exact statement span once
the model has picked it from a menu -- see vertical.py). Replace and
insert-after ask for only the delta: a stateless micro-prompt in the style
of threetoks/code/nodes.py's ImplementNode -- goal + verbatim current span
+ a little surrounding context, prefill'd with the def line, tight token
cap, resampled blind (temps 0.0/0.0/0.4) on any gate failure.

The gate that matters most for the E8 thesis: after every splice the
harness re-parses the WHOLE file. A splice that breaks ast.parse is
rejected for free (no model call spent judging it) and the policy's own
retry ladder (threetoks/policy.py) resamples -- this file never re-derives
that ladder, it just reports each attempt's candidate as valid/invalid.
"""
import ast
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_DIR.parent.parent
for _p in (str(SPIKE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from threetoks.code import gates  # noqa: E402  (path bootstrap above)
from threetoks.nodes import Decision  # noqa: E402

OP_REPLACE = "replace"
OP_INSERT_AFTER = "insert_after"
OP_DELETE = "delete"
CONTEXT_LINES = 3
BODY_INDENT_WIDTH = 4

GENERATE_SYSTEM = ("You are a Python programmer fixing one function in an "
                   "existing file. Write exactly the requested code. Output "
                   "only code, no explanations, no markdown. Never leave it "
                   "unfinished: no pass, no ..., no raise NotImplementedError.")
IMPLEMENT_STOP = ("\ndef ", "\nclass ", "\nif __name__", "\n@", "\nprint(")
GENERATE_MAX_TOKENS = 200


# ------------------------------------------------------------ pure splices

def replace_span(source: str, start: int, end: int, replacement: str) -> str:
    """Swap lines [start, end] (1-based, inclusive) for replacement text."""
    lines = source.splitlines()
    new_lines = lines[:start - 1] + replacement.splitlines() + lines[end:]
    return "\n".join(new_lines) + "\n"


def insert_after_span(source: str, end: int, new_text: str) -> str:
    """Insert new_text as new lines immediately after line `end`."""
    lines = source.splitlines()
    block = ["", *new_text.splitlines(), ""]
    new_lines = lines[:end] + block + lines[end:]
    return "\n".join(new_lines) + "\n"


def delete_span(source: str, start: int, end: int) -> str:
    """Remove lines [start, end] (1-based, inclusive)."""
    lines = source.splitlines()
    new_lines = lines[:start - 1] + lines[end:]
    return "\n".join(new_lines) + "\n"


def span_text(source: str, start: int, end: int) -> str:
    """The verbatim text of one span -- copied into prompts, never
    paraphrased."""
    lines = source.splitlines()
    return "\n".join(lines[start - 1:end])


def context_lines(source: str, start: int, end: int,
                   n: int = CONTEXT_LINES) -> tuple[str, str]:
    """Up to n lines of file text immediately before/after a span."""
    lines = source.splitlines()
    before = lines[max(0, start - 1 - n):start - 1]
    after = lines[end:end + n]
    return "\n".join(before), "\n".join(after)


def statement_spans(source: str, func_name: str) -> list[tuple[int, int]]:
    """1-based (start, end) span of each top-level statement in a
    function's body -- the unit a "delete the offending line" menu offers.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == func_name:
            return [(stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno))
                    for stmt in node.body]
    return []


def find_missing_helper(source: str, func_name: str,
                        known_names: set[str]) -> tuple[str, str] | None:
    """Deterministically name+arity a helper a function calls but never
    defines -- zero model calls to figure out the signature.

    Reuses threetoks.code.gates.undefined_names (the code vertical's own
    undefined-name gate) to find the missing callee, then counts the
    positional args of its first call site to synthesize a parameter list
    (a, b, c, ...). Only the *body* is left for the model to write.
    """
    missing = gates.undefined_names(source, known_names)
    if not missing:
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    target = next((n for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == func_name), None)
    if target is None:
        return None
    for node in ast.walk(target):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in missing:
            params = ", ".join("abcdefgh"[i] for i in range(len(node.args)))
            return node.func.id, params
    return None


# ------------------------------------------------------------ generation node

class GenerateSpanNode:
    """Ask for the replacement/insertion body only -- never the whole file.

    Duck-typed the same way threetoks/code/nodes.py's ImplementNode is:
    kind/system/prefill/max_tokens/stop/render/parse, so it flows through
    the existing Policy retry ladder untouched. `splice_fn` closes over the
    file's current source and the target span; it applies the splice, then
    re-parses the WHOLE file -- a candidate that breaks ast.parse comes
    back invalid, which is exactly what makes the ladder resample.
    """

    kind = "generate_span"
    tag = "generate_span"
    system = GENERATE_SYSTEM
    stop = IMPLEMENT_STOP
    max_tokens = GENERATE_MAX_TOKENS

    def __init__(self, instruction: str, prefill: str, splice_fn,
                 temperature: float | None = None):
        self.question = instruction
        self.prefill = prefill
        self._splice_fn = splice_fn
        self.temperature = temperature

    def render(self, perm: tuple[int, ...]) -> str:
        return self.question

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        candidate = self._splice_fn(text)
        return Decision(self.kind, candidate, text, valid=candidate is not None)


def renormalize_indent(text: str) -> str:
    """Fix continuation-line indent drift in a multi-line completion.

    threetoks/code/gates.reconstruct already strips a stray leading space
    on the completion's FIRST line (E6: otherwise it lands at 5 spaces
    against the prefill's 4 and corrupts the module). Live E8 runs turned
    up the same drift one level down: qwen2.5:1.5b-instruct sometimes
    indents EVERY continuation line at 5 spaces instead of 4 (a docstring
    line, then a run of body lines, all shifted by one column), which
    `.lstrip()` on the whole string does not touch because it only trims
    the start of the string, not the start of each line. This dedents to
    the block's minimum indent and re-adds exactly one 4-space level,
    preserving any deeper relative nesting (an `if`/`for` body keeps its
    extra indent) -- a whitespace-consistency gate, not a content change.
    """
    lines = text.splitlines()
    if len(lines) <= 1:
        return text
    first, rest = lines[0], lines[1:]
    indents = [len(l) - len(l.lstrip(" ")) for l in rest if l.strip()]
    if not indents:
        return text
    shift = BODY_INDENT_WIDTH - min(indents)
    if shift == 0:
        return text
    fixed = [first]
    for line in rest:
        if not line.strip():
            fixed.append(line)
            continue
        current = len(line) - len(line.lstrip(" "))
        fixed.append(" " * max(0, current + shift) + line.lstrip(" "))
    return "\n".join(fixed)


def replace_splice_fn(file_source: str, start: int, end: int,
                      name: str, args: str):
    """Build the splice_fn for a replace-span edit on function `name`.

    Rejects (for free, no model call spent) a candidate that: doesn't
    parse as `name(args)`, is a placeholder dodge (pass/NotImplementedError
    cheats, reusing the code vertical's own gate), or whose splice breaks
    ast.parse on the WHOLE file.
    """
    def fn(text: str) -> str | None:
        new_source = gates.function_source(name, args, renormalize_indent(text))
        if new_source is None or gates.is_placeholder(new_source, name):
            return None
        candidate = replace_span(file_source, start, end, new_source)
        return candidate if gates.parses(candidate) else None
    return fn


def insert_after_splice_fn(file_source: str, end: int, name: str, args: str):
    """Build the splice_fn for an insert-after-span edit (a brand-new
    function, e.g. a helper the target function calls but never defines).
    """
    def fn(text: str) -> str | None:
        new_source = gates.function_source(name, args, renormalize_indent(text))
        if new_source is None or gates.is_placeholder(new_source, name):
            return None
        candidate = insert_after_span(file_source, end, new_source)
        return candidate if gates.parses(candidate) else None
    return fn


if __name__ == "__main__":
    src = "def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y\n"
    spans = statement_spans(src, "f")
    assert spans == [(2, 2)], spans

    replaced = replace_span(src, 1, 2, "def f(x):\n    return x * 2")
    assert "return x * 2" in replaced and ast.parse(replaced)

    inserted = insert_after_span(src, 2, "def h(z):\n    return z")
    assert "def h(z):" in inserted and "def g(y):" in inserted
    assert ast.parse(inserted)

    deleted = delete_span(src, 2, 2)
    assert "return x + 1" not in deleted

    fn = replace_splice_fn(src, 1, 2, "f", "x")
    ok = fn("return x * 3")
    assert ok is not None and "return x * 3" in ok
    bad = fn("return (((")            # breaks ast.parse -> rejected free
    assert bad is None
    cheat = fn("raise NotImplementedError")
    assert cheat is None               # placeholder dodge rejected

    missing_src = ("def average(nums):\n    total = 0\n"
                  "    for n in nums:\n        total = _safe_add(total, n)\n"
                  "    return total / len(nums)\n")
    helper = find_missing_helper(missing_src, "average", {"average"})
    assert helper == ("_safe_add", "a, b"), helper

    ins_fn = insert_after_splice_fn(missing_src, 5, "_safe_add", "a, b")
    with_helper = ins_fn("return a + b")
    assert with_helper is not None
    ast.parse(with_helper)

    # live-observed drift: every continuation line indented one column too
    # deep (5 spaces, not 4) -- a real qwen2.5:1.5b-instruct completion.
    drifted = ('"""doc"""\n     start = (page - 1) * size\n'
              "     end = start + size\n     return items[start:end]")
    fixed = renormalize_indent(drifted)
    assert fixed.count("\n     ") == 0, fixed     # no more 5-space lines
    assert fixed.count("\n    ") == 3, fixed        # all re-anchored at 4
    fn2 = replace_splice_fn(src, 1, 2, "f", "x")
    assert fn2(drifted) is not None, "renormalized body should now parse"
    # a body with real nested indentation keeps its relative depth
    nested = "if x:\n         return 1\n     return 0"    # 9 / 5 spaces
    fixed_nested = renormalize_indent(nested)
    assert "\n    return 0" in fixed_nested          # outer -> 4
    assert "\n        return 1" in fixed_nested       # inner stays +4 deeper
    print("smoke OK")
