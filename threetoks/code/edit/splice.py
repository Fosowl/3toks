"""Splice functions for the edit vertical: rebuild, gate, resample.

Replace and insert-after ask the model for only a function body
(GenerateSpanNode). The harness keeps the original def header verbatim,
puts back a docstring the model dropped, and gates each candidate for
free: the function must parse, must not be a placeholder dodge, must not
use names the file does not define (skipped for a star-import file),
must not repeat a version that already failed (compared by
``canonical_key``), and the spliced WHOLE file must compile. A candidate
equal to the original function is reported as ``UNCHANGED``. Any gate
failure makes the attempt invalid, so Policy's retry ladder samples
again.
"""
import ast

from threetoks.code import gates
from threetoks.code.edit.names import has_star_import
from threetoks.code.edit.spans import (FUNC_TYPES, LEADING_WS, body_prefill,
                                       dedent_block, indent_block,
                                       insert_after_span, is_docstring,
                                       leading_indent, replace_span,
                                       span_text, without_docstring)
from threetoks.code.nodes import IMPLEMENT_MAX_TOKENS, IMPLEMENT_STOP
from threetoks.nodes import Decision

UNCHANGED = "<unchanged>"
_INDENTED_WRAPPER = "if True:\n"

GENERATE_SYSTEM = ("You are a Python programmer fixing one function in an "
                   "existing file. Write exactly the requested code. Output "
                   "only code, no explanations, no markdown. Never leave it "
                   "unfinished: no pass, no ..., no raise NotImplementedError.")
# The stop/cap contract is the greenfield ImplementNode's — one source of
# truth, so an anti-cheat tweak there propagates here.
GENERATE_STOP = IMPLEMENT_STOP
GENERATE_MAX_TOKENS = IMPLEMENT_MAX_TOKENS


def _single_def(function_source: str) -> ast.AST | None:
    """Parse text that holds exactly one def (maybe indented).

    canonical_key and the splice gates call this. Indented text (a
    method span) is parsed inside an "if True:" wrapper.

    Args:
        function_source: str, the text of one function.

    Returns:
        ast.AST | None: the def node, or None when it does not parse.
    """
    indented = function_source[:1] in LEADING_WS
    text = _INDENTED_WRAPPER + function_source if indented else function_source
    try:
        body = ast.parse(text).body
    except (SyntaxError, ValueError):
        return None
    if indented and body:
        body = body[0].body
    if len(body) != 1 or not isinstance(body[0], FUNC_TYPES):
        return None
    return body[0]


def canonical_key(function_source: str) -> str:
    """Give a key that is equal for functions with the same code.

    The splice functions use it to detect UNCHANGED, and the edit vertical
    adds it to ``tried`` after a candidate fails. Whitespace, comments
    and the docstring do not change the key.

    Args:
        function_source: str, the text of one def (no decorators needed).

    Returns:
        str: ast.dump of the def without its docstring, "" on parse error.
    """
    node = _single_def(function_source)
    if node is None:
        return ""
    node.body = without_docstring(node.body)
    return ast.dump(node)


class GenerateSpanNode:
    """Ask for one function body only — never the whole file.

    Duck-typed like the code vertical's ImplementNode (kind/system/prefill/
    max_tokens/stop/render/parse), so it flows through the Policy retry
    ladder. ``splice_fn`` closes over the file source and the target span;
    it gates the candidate and applies the splice. A rejected attempt is
    an invalid Decision, and a truncated completion is invalid too
    (``reject_truncated``), so the ladder samples again.
    """

    kind = "generate_span"
    tag = "generate_span"
    system = GENERATE_SYSTEM
    stop = GENERATE_STOP
    max_tokens = GENERATE_MAX_TOKENS
    reject_truncated = True

    def __init__(self, instruction: str, prefill: str, splice_fn,
                 temperature: float | None = None):
        """Keep the prompt parts and the splice function.

        Args:
            instruction: str, the micro-prompt from the edit vertical.
            prefill: str, body_prefill(header) from the edit vertical.
            splice_fn: callable, from replace_splice_fn or
                insert_after_splice_fn.
            temperature: float | None, an override for the retry ladder.
        """
        self.question = instruction
        self.prefill = prefill
        self._splice_fn = splice_fn
        self.temperature = temperature

    def render(self, perm: tuple[int, ...]) -> str:
        """Give the instruction; the request context is in the episode.

        Args:
            perm: tuple[int, ...], the option order from Policy (unused).

        Returns:
            str: the instruction.
        """
        return self.question

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Turn the model text into a Decision through the splice function.

        Args:
            text: str, the model completion after the prefill.
            perm: tuple[int, ...], the option order from Policy (unused).

        Returns:
            Decision: value is (candidate_file, function_source) or
            UNCHANGED; invalid when the splice function returned None.
        """
        value = self._splice_fn(text)
        return Decision(self.kind, value, text, valid=value is not None)


def replace_splice_fn(file_source: str, start: int, end: int, header: str,
                      indent: str, known_names: set[str], tried: set[str]):
    """Make the splice function for a replace edit of one function.

    The edit vertical gives it to GenerateSpanNode. The new function is
    the verbatim header plus the model body; the original docstring goes
    back in when the model drops it. Decorators above ``start`` stay.
    Names the original function already uses unresolved stay allowed.
    In a file with a star import the undefined-name gate is skipped.

    Args:
        file_source: str, the file text (LF), from the symbol index.
        start: int, the def line of the target (1-based, no decorators).
        end: int, the last line of the target (inclusive).
        header: str, from def_header.
        indent: str, the indent of the original def line.
        known_names: set[str], names the file defines.
        tried: set[str], canonical keys of candidates that already failed.

    Returns:
        callable: fn(model_text) -> (candidate_file, new_function_source),
        UNCHANGED, or None on any gate reject.
    """
    original = dedent_block(span_text(file_source, start, end), indent)
    allowed = None if has_star_import(file_source) else \
        known_names | set(gates.undefined_names(original, known_names))
    old_key = canonical_key(original)
    doc = _docstring_text(original)

    def fn(text: str) -> tuple[str, str] | str | None:
        """Gate one model body and splice it into the file.

        Args:
            text: str, the model completion after body_prefill(header).

        Returns:
            tuple[str, str] | str | None: (candidate_file, new_function),
            UNCHANGED, or None on a gate reject.
        """
        new_source = _with_docstring(_function_from(header, text), doc)
        if new_source is None:
            return None
        if canonical_key(new_source) == old_key:
            return UNCHANGED
        if not _passes_gates(new_source, allowed, tried):
            return None
        candidate = replace_span(file_source, start, end,
                                 indent_block(new_source, indent))
        return (candidate, new_source) if gates.parses(candidate) else None
    return fn


def insert_after_splice_fn(file_source: str, end: int, name: str, args: str,
                           known_names: set[str], tried: set[str]):
    """Make the splice function that adds a new top-level function.

    The edit vertical uses it for a helper the target calls but nothing
    defines (see find_missing_helper). The gates are the same as for
    replace_splice_fn; it never gives UNCHANGED.

    Args:
        file_source: str, the file text (LF), from the symbol index.
        end: int, the line after which the helper goes (1-based).
        name: str, the helper name.
        args: str, the parameter text from find_missing_helper.
        known_names: set[str], names the file defines.
        tried: set[str], canonical keys of candidates that already failed.

    Returns:
        callable: fn(model_text) -> (candidate_file, new_function_source)
        or None on any gate reject.
    """
    header = f"def {name}({args}):"
    allowed = None if has_star_import(file_source) else known_names

    def fn(text: str) -> tuple[str, str] | None:
        """Gate one model helper body and insert it into the file.

        Args:
            text: str, the model completion after body_prefill(header).

        Returns:
            tuple[str, str] | None: (candidate_file, new_function), or
            None on a gate reject.
        """
        new_source = _function_from(header, text)
        if new_source is None or not _passes_gates(new_source, allowed,
                                                   tried):
            return None
        candidate = insert_after_span(file_source, end, new_source)
        return (candidate, new_source) if gates.parses(candidate) else None
    return fn


def _function_from(header: str, text: str) -> str | None:
    """Rebuild a function from its header and the model body.

    The body follows body_prefill(header). When the raw body does not
    parse, one more try uses gates.renormalize_indent (the qwen
    one-column drift).

    Args:
        header: str, the def header at column 0.
        text: str, the model completion after the prefill.

    Returns:
        str | None: the function text, cut after the def, or None.
    """
    source = _trimmed_def(body_prefill(header) + text.lstrip(LEADING_WS))
    if source is not None:
        return source
    fixed = gates.renormalize_indent(text)
    return _trimmed_def(body_prefill(header) + fixed.lstrip(LEADING_WS))


def _trimmed_def(raw: str) -> str | None:
    """Cut a rebuilt function after its def, dropping trailing code.

    Model text can run on into top-level code; that code is dropped.

    Args:
        raw: str, the header plus the model text.

    Returns:
        str | None: the def lines only, or None when it does not parse.
    """
    try:
        body = ast.parse(raw).body
    except (SyntaxError, ValueError):
        return None
    if not body or not isinstance(body[0], FUNC_TYPES):
        return None
    return "\n".join(raw.split("\n")[:body[0].end_lineno])


def _docstring_text(function_source: str) -> str:
    """Give the verbatim docstring statement of a function.

    Args:
        function_source: str, the original function at column 0.

    Returns:
        str: the docstring source text, "" when there is none.
    """
    node = _single_def(function_source)
    if node is None or not node.body or not is_docstring(node.body[0]):
        return ""
    return ast.get_source_segment(function_source, node.body[0]) or ""


def _with_docstring(function_source: str | None, doc: str) -> str | None:
    """Put the original docstring back when the model dropped it.

    Args:
        function_source: str | None, the rebuilt function (column 0).
        doc: str, from _docstring_text; "" means nothing to add.

    Returns:
        str | None: the function with a docstring, or the input unchanged
        when it has one already or the result would not parse.
    """
    node = None if function_source is None else _single_def(function_source)
    if not doc or node is None or is_docstring(node.body[0]):
        return function_source
    lines = function_source.split("\n")
    at = node.body[0].lineno - 1
    indent = leading_indent(lines[at])
    merged = "\n".join(lines[:at] + (indent + doc).split("\n") + lines[at:])
    return merged if _single_def(merged) is not None else function_source


def _passes_gates(new_source: str, allowed: set[str] | None,
                  tried: set[str]) -> bool:
    """Run the free function gates on a rebuilt function.

    Args:
        new_source: str, the rebuilt function (column 0, parses).
        allowed: set[str] | None, names the function may use without
            defining; None skips the name gate (star import in the file).
        tried: set[str], canonical keys of candidates that already failed.

    Returns:
        bool: True when it is not a placeholder, uses no unknown name,
        and was not tried before.
    """
    name = _single_def(new_source).name
    if gates.is_placeholder(new_source, name):
        return False
    if allowed is not None and gates.undefined_names(new_source, allowed):
        return False
    return canonical_key(new_source) not in tried


if __name__ == "__main__":
    fancy = ("class C:\n    @staticmethod\n    async def run(a, b=2, *rest,"
             "\n                  key: int = 0, **kw) -> int:\n"
             '        """Run it."""\n        return a\n')
    from threetoks.code.edit.spans import def_header
    header = def_header(fancy, "C.run", 3)
    fn = replace_splice_fn(fancy, 3, 6, header, "    ", set(), set())
    candidate, new_source = fn("return a + b")
    assert "    @staticmethod\n    async def run(" in candidate
    assert '"""Run it."""' in new_source             # docstring re-inserted
    assert fn("return a") == UNCHANGED
    assert fn("return (((") is None                 # free parse reject
    assert fn("raise NotImplementedError") is None  # placeholder dodge
    assert fn("return nowhere(a)") is None          # undefined name
    tried = {canonical_key(new_source)}
    assert replace_splice_fn(fancy, 3, 6, header, "    ", set(), tried)(
        "return a + b") is None                     # already failed

    missing_src = ("def average(nums):\n    total = 0\n"
                   "    for n in nums:\n        total = _safe_add(total, n)\n"
                   "    return total / len(nums)\n")
    ins = insert_after_splice_fn(missing_src, 5, "_safe_add", "arg1, arg2",
                                 set(), set())
    with_helper, _ = ins("return arg1 + arg2")
    ast.parse(with_helper)
    print("smoke OK")
