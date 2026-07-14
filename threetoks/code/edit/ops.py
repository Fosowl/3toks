"""Edit machinery owned by the harness: splice a span, gate, resample.

Three operations over ast-selected spans (1-based, inclusive lines):
replace-span, insert-after-span, delete-span. Delete never calls the
model. Replace and insert-after ask for only the delta via a stateless
micro-prompt (GenerateSpanNode), gated for free: the candidate must parse
as the named function, must not be a placeholder dodge, must not repeat a
body already seen this episode (the oscillation guard — E8 scenario 4
burned a repair round regenerating the original bug verbatim), and the
spliced WHOLE file must still ast.parse. Any gate failure invalidates the
attempt so Policy's existing retry ladder resamples blind.
"""
import ast

from threetoks.code import gates
from threetoks.code.nodes import IMPLEMENT_MAX_TOKENS, IMPLEMENT_STOP
from threetoks.nodes import Decision

CONTEXT_LINES = 3
FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)

GENERATE_SYSTEM = ("You are a Python programmer fixing one function in an "
                   "existing file. Write exactly the requested code. Output "
                   "only code, no explanations, no markdown. Never leave it "
                   "unfinished: no pass, no ..., no raise NotImplementedError.")
# The stop/cap contract is the greenfield ImplementNode's — one source of
# truth, so an anti-cheat tweak there propagates here.
GENERATE_STOP = IMPLEMENT_STOP
GENERATE_MAX_TOKENS = IMPLEMENT_MAX_TOKENS


def replace_span(source: str, start: int, end: int, replacement: str) -> str:
    """Swap lines [start, end] for the replacement text."""
    lines = source.splitlines()
    return "\n".join(lines[:start - 1] + replacement.splitlines()
                     + lines[end:]) + "\n"


def insert_after_span(source: str, end: int, new_text: str) -> str:
    """Insert new_text as a blank-line-separated block after line ``end``."""
    lines = source.splitlines()
    block = ["", *new_text.splitlines(), ""]
    return "\n".join(lines[:end] + block + lines[end:]) + "\n"


def delete_span(source: str, start: int, end: int) -> str:
    """Remove lines [start, end]."""
    lines = source.splitlines()
    return "\n".join(lines[:start - 1] + lines[end:]) + "\n"


def span_text(source: str, start: int, end: int) -> str:
    """The verbatim text of one span — copied into prompts, never
    paraphrased."""
    return "\n".join(source.splitlines()[start - 1:end])


def context_lines(source: str, start: int, end: int,
                  n: int = CONTEXT_LINES) -> tuple[str, str]:
    """Up to n lines of file text immediately before/after a span."""
    lines = source.splitlines()
    before = lines[max(0, start - 1 - n):start - 1]
    return "\n".join(before), "\n".join(lines[end:end + n])


def find_def(source: str, qualname: str) -> ast.AST | None:
    """The def node for a top-level function or a "Class.method" qualname."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    if "." not in qualname:
        return next((n for n in tree.body if isinstance(n, FUNC_TYPES)
                     and n.name == qualname), None)
    class_name, method = qualname.split(".", 1)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == class_name), None)
    if cls is None:
        return None
    return next((n for n in cls.body if isinstance(n, FUNC_TYPES)
                 and n.name == method), None)


def def_args(source: str, qualname: str) -> str:
    """The argument list of a def as plan-style text ("a, b"), or ""."""
    node = find_def(source, qualname)
    if node is None:
        return ""
    return ", ".join(a.arg for a in node.args.args)


def indent_block(text: str, indent: str) -> str:
    """Prefix every non-blank line with ``indent`` (method-body splices)."""
    if not indent:
        return text
    return "\n".join(indent + line if line.strip() else line
                     for line in text.splitlines())


def statement_spans(source: str, qualname: str) -> list[tuple[int, int]]:
    """1-based (start, end) of each statement in a def's body — the unit a
    "delete the offending line" menu offers. Works for methods too."""
    node = find_def(source, qualname)
    if node is None:
        return []
    return [(stmt.lineno, getattr(stmt, "end_lineno", stmt.lineno))
            for stmt in node.body]


def module_level_names(source: str) -> set[str]:
    """Every name a file binds at top level: imports, defs, classes,
    assignments.

    The missing-helper inference must treat all of these as defined — a
    call to a ``from helpers import normalize`` name or a module constant
    is NOT a missing helper, and inferring insert-after for one would
    shadow the real binding.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
        elif isinstance(node, FUNC_TYPES + (ast.ClassDef,)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) \
                and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def find_missing_helper(source: str, func_name: str,
                        known_names: set[str]) -> tuple[str, str] | None:
    """Deterministically name+arity a helper a function calls but never
    defines — zero model calls to figure out the signature.

    Reuses the undefined-name gate to find the missing callee, then counts
    the first call site's positional args to synthesize a parameter list
    (a, b, c, ...). Only the *body* is left for the model to write.
    """
    missing = gates.undefined_names(source, known_names)
    target = gates.function_def(source, func_name)
    if not missing or target is None:
        return None
    for node in ast.walk(target):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in missing:
            params = ", ".join("abcdefgh"[i] for i in range(len(node.args)))
            return node.func.id, params
    return None


class GenerateSpanNode:
    """Ask for the replacement/insertion body only — never the whole file.

    Duck-typed like the code vertical's ImplementNode (kind/system/prefill/
    max_tokens/stop/render/parse) so it flows through the existing Policy
    retry ladder untouched. ``splice_fn`` closes over the file's current
    source and target span; it gates the candidate and applies the splice.
    An attempt any gate rejects parses as invalid, which is exactly what
    makes the ladder resample.
    """

    kind = "generate_span"
    tag = "generate_span"
    system = GENERATE_SYSTEM
    stop = GENERATE_STOP
    max_tokens = GENERATE_MAX_TOKENS

    def __init__(self, instruction: str, prefill: str, splice_fn,
                 temperature: float | None = None):
        self.question = instruction
        self.prefill = prefill
        self._splice_fn = splice_fn
        self.temperature = temperature

    def render(self, perm: tuple[int, ...]) -> str:
        """The instruction; request context lives in the episode."""
        return self.question

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Value is (candidate_file, function_source), or invalid."""
        spliced = self._splice_fn(text)
        return Decision(self.kind, spliced, text, valid=spliced is not None)


def replace_splice_fn(file_source: str, start: int, end: int, name: str,
                      args: str, seen_hashes: set[int] | None = None,
                      indent: str = ""):
    """Splice_fn for a replace-span edit on function ``name``.

    Rejects for free a candidate that: does not parse as ``name(args)``,
    is a placeholder dodge, repeats an already-seen body (oscillation), or
    whose splice breaks ast.parse on the whole file. ``indent`` re-indents
    the generated def for method targets (the model always writes at
    column 0; the harness owns file indentation). Returns
    (candidate_file, function_source) on success, else None.
    """
    def fn(text: str) -> tuple[str, str] | None:
        new_source = _gated_function(name, args, text, seen_hashes)
        if new_source is None:
            return None
        candidate = replace_span(file_source, start, end,
                                 indent_block(new_source, indent))
        return (candidate, new_source) if gates.parses(candidate) else None
    return fn


def insert_after_splice_fn(file_source: str, end: int, name: str, args: str,
                           seen_hashes: set[int] | None = None):
    """Splice_fn for inserting a brand-new top-level function after line
    ``end`` (e.g. a helper the target calls but never defines)."""
    def fn(text: str) -> tuple[str, str] | None:
        new_source = _gated_function(name, args, text, seen_hashes)
        if new_source is None:
            return None
        candidate = insert_after_span(file_source, end, new_source)
        return (candidate, new_source) if gates.parses(candidate) else None
    return fn


def _gated_function(name: str, args: str, text: str,
                    seen_hashes: set[int] | None) -> str | None:
    """Reconstruct + placeholder + oscillation gates; None on any reject."""
    new_source = gates.function_source(name, args, text)
    if new_source is None or gates.is_placeholder(new_source, name):
        return None
    if seen_hashes is not None and hash(new_source) in seen_hashes:
        return None
    return new_source


if __name__ == "__main__":
    src = "def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y\n"
    assert statement_spans(src, "f") == [(2, 2)]
    assert "return x * 2" in replace_span(src, 1, 2, "def f(x):\n    return x * 2")
    inserted = insert_after_span(src, 2, "def h(z):\n    return z")
    assert "def h(z):" in inserted and ast.parse(inserted)
    assert "return x + 1" not in delete_span(src, 2, 2)
    assert span_text(src, 2, 2) == "    return x + 1"

    fn = replace_splice_fn(src, 1, 2, "f", "x")
    candidate, new_source = fn("return x * 3")
    assert "return x * 3" in candidate and new_source.startswith("def f(x):")
    assert fn("return (((") is None                 # breaks parse -> free reject
    assert fn("raise NotImplementedError") is None  # placeholder dodge

    seen = {hash("def f(x):\n    return x + 1")}    # the original buggy body
    guarded = replace_splice_fn(src, 1, 2, "f", "x", seen)
    assert guarded("return x + 1") is None          # oscillation guard fires
    assert guarded("return x - 1") is not None

    missing_src = ("def average(nums):\n    total = 0\n"
                   "    for n in nums:\n        total = _safe_add(total, n)\n"
                   "    return total / len(nums)\n")
    assert find_missing_helper(missing_src, "average", {"average"}) == \
        ("_safe_add", "a, b")
    ins = insert_after_splice_fn(missing_src, 5, "_safe_add", "a, b")
    with_helper, _ = ins("return a + b")
    ast.parse(with_helper)

    classy = ("class Box:\n    def get(self):\n        return self.v + 1\n\n"
              "    def other(self):\n        return 0\n")
    assert find_def(classy, "Box.get").name == "get"
    assert def_args(classy, "Box.get") == "self"
    assert statement_spans(classy, "Box.get") == [(3, 3)]
    method_fn = replace_splice_fn(classy, 2, 3, "get", "self", indent="    ")
    fixed_file, method_src = method_fn("return self.v - 1")
    assert "    def get(self):" in fixed_file       # re-indented into the class
    assert method_src.startswith("def get(self):")  # model-side stays col 0
    ast.parse(fixed_file)
    print("smoke OK")
