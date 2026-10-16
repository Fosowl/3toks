"""Span helpers for the edit vertical: line spans, def lookup, headers.

Edits work on ast-selected spans (1-based, inclusive lines). All helpers
split text with ``split("\\n")``, so their line numbers agree with ast
line numbers and the file keeps its trailing newline. ``def_header``
gives the verbatim def header (async, defaults, annotations), and
``find_def`` / ``statement_spans`` tell same-named defs apart by line.
"""
import ast
import io
import tokenize

from threetoks.code import gates

CONTEXT_LINES = 3
FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
BODY_INDENT = gates.BODY_INDENT
LEADING_WS = " \t"
_OPENERS = ("(", "[", "{")
_CLOSERS = (")", "]", "}")


def replace_span(source: str, start: int, end: int, replacement: str) -> str:
    """Replace the lines of one span with new text.

    The splice functions and the edit vertical use this to put a new
    function into a file. It keeps every other line, and the trailing
    newline of the file, exactly as they were.

    Args:
        source: str, the file text (LF), from the symbol index.
        start: int, first line to replace (1-based, from ast).
        end: int, last line to replace (inclusive, from ast).
        replacement: str, the new lines, without a trailing newline.

    Returns:
        str: the new file text.
    """
    lines = source.split("\n")
    return "\n".join(lines[:start - 1] + replacement.split("\n")
                     + lines[end:])


def insert_after_span(source: str, end: int, new_text: str) -> str:
    """Insert a new block after one line, with one blank line before it.

    The insert-after splice function uses this to add a missing helper.
    It adds a blank line after the block only when code follows it.

    Args:
        source: str, the file text (LF), from the symbol index.
        end: int, the line after which the block goes (1-based).
        new_text: str, the block, without a trailing newline.

    Returns:
        str: the new file text.
    """
    lines = source.split("\n")
    rest = lines[end:]
    block = ["", *new_text.split("\n")]
    if rest and rest[0].strip():
        block.append("")
    return "\n".join(lines[:end] + block + rest)


def delete_span(source: str, start: int, end: int) -> str:
    """Remove the lines of one span.

    The edit vertical uses this for the delete operation. The result can
    fail to parse; the caller must gate it.

    Args:
        source: str, the file text (LF), from the symbol index.
        start: int, first line to remove (1-based).
        end: int, last line to remove (inclusive).

    Returns:
        str: the new file text.
    """
    lines = source.split("\n")
    return "\n".join(lines[:start - 1] + lines[end:])


def span_text(source: str, start: int, end: int) -> str:
    """Give the verbatim text of one span.

    The edit vertical copies this text into prompts and menus. It is
    never paraphrased.

    Args:
        source: str, the file text (LF).
        start: int, first line (1-based).
        end: int, last line (inclusive).

    Returns:
        str: the lines of the span, joined with LF.
    """
    return "\n".join(source.split("\n")[start - 1:end])


def context_lines(source: str, start: int, end: int,
                  n: int = CONTEXT_LINES) -> tuple[str, str]:
    """Give the file lines just before and just after a span.

    The edit vertical shows them in the replace prompt.

    Args:
        source: str, the file text (LF).
        start: int, first line of the span (1-based).
        end: int, last line of the span (inclusive).
        n: int, how many lines to give on each side.

    Returns:
        tuple[str, str]: the text before and the text after the span.
    """
    lines = source.split("\n")
    before = lines[max(0, start - 1 - n):start - 1]
    return "\n".join(before), "\n".join(lines[end:end + n])


def parse_source(source: str) -> ast.Module | None:
    """Parse file text, or give None when it is not valid Python.

    Args:
        source: str, the file text.

    Returns:
        ast.Module | None: the tree, None on a syntax or null-byte error.
    """
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def find_def(source: str, qualname: str,
             lineno: int | None = None) -> ast.AST | None:
    """Find the def node of a top-level function or of a method.

    The span helpers and the edit vertical use this to locate a target.
    Same-named defs (a property setter, @overload, alternates) are told
    apart by ``lineno``; without it the first def is given.

    Args:
        source: str, the file text.
        qualname: str, "name" or "Class.method", from the symbol index.
        lineno: int | None, the def line (Symbol.lineno), or None.

    Returns:
        ast.AST | None: the FunctionDef/AsyncFunctionDef, or None.
    """
    tree = parse_source(source)
    if tree is None:
        return None
    defs = _named_defs(tree, qualname)
    if lineno is not None:
        defs = [node for node in defs if node.lineno == lineno]
    return defs[0] if defs else None


def _named_defs(tree: ast.Module, qualname: str) -> list[ast.AST]:
    """Give every def of a module with this qualname, in source order.

    Args:
        tree: ast.Module, the parsed file.
        qualname: str, "name" or "Class.method".

    Returns:
        list[ast.AST]: the matching FunctionDef/AsyncFunctionDef nodes.
    """
    if "." not in qualname:
        return [n for n in tree.body
                if isinstance(n, FUNC_TYPES) and n.name == qualname]
    class_name, method = qualname.split(".", 1)
    return [n for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == class_name
            for n in cls.body
            if isinstance(n, FUNC_TYPES) and n.name == method]


def function_text(source: str, node: ast.AST) -> str:
    """Give the text of one def at column 0, without its decorators.

    Args:
        source: str, the file text (LF).
        node: ast.AST, a def node parsed from source.

    Returns:
        str: the lines from the def line to the end of the def, dedented.
    """
    text = span_text(source, node.lineno, node.end_lineno)
    return dedent_block(text, leading_indent(text))


def def_header(source: str, qualname: str, lineno: int) -> str | None:
    """Give the verbatim def header of a function, at column 0.

    The edit vertical puts this header in the generate prefill, and
    replace_splice_fn puts it in front of the model body. So async,
    defaults, annotations, keyword-only and star parameters survive an
    edit. Decorators are not part of the header. A multi-line header
    stays multi-line. For "def f(x): return x" the header is "def f(x):".

    Args:
        source: str, the file text (LF).
        qualname: str, "name" or "Class.method", from the symbol index.
        lineno: int, the def line (Symbol.lineno); it selects one of
            same-named defs (property setter, @overload).

    Returns:
        str | None: the header, ending with ":", or None when not found.
    """
    node = find_def(source, qualname, lineno)
    if node is None:
        return None
    segment = function_text(source, node)
    colon = _header_end(segment)
    if colon is None:
        return None
    row, col = colon
    header_lines = segment.split("\n")[:row]
    header_lines[-1] = header_lines[-1][:col]
    return "\n".join(header_lines)


def _header_end(segment: str) -> tuple[int, int] | None:
    """Find the colon that ends a def header.

    def_header calls this with the def text at column 0. Colons inside
    brackets (lambda defaults, annotations, type parameters) do not count.

    Args:
        segment: str, text that starts with the def keyword line.

    Returns:
        tuple[int, int] | None: (1-based row, column after the colon).
    """
    depth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(segment).readline):
            if tok.string in _OPENERS:
                depth += 1
            elif tok.string in _CLOSERS:
                depth -= 1
            elif tok.string == ":" and depth == 0:
                return tok.end
    except (tokenize.TokenError, SyntaxError):
        return None
    return None


def body_prefill(header: str) -> str:
    """Give the generate prefill for a def header.

    The edit vertical uses this as GenerateSpanNode.prefill. The splice
    functions rebuild the function the same way, so the two must agree.

    Args:
        header: str, from def_header or "def name(args):".

    Returns:
        str: the header, a newline, and the body indent.
    """
    return f"{header}\n{BODY_INDENT}"


def indent_block(text: str, indent: str) -> str:
    """Add an indent to every non-blank line (method splices).

    The splice functions call this to put a column-0 function back at
    the indent of the original def.

    Args:
        text: str, the column-0 function text.
        indent: str, the indent of the original def line.

    Returns:
        str: the indented text.
    """
    if not indent:
        return text
    return "\n".join(indent + line if line.strip() else line
                     for line in text.split("\n"))


def dedent_block(text: str, indent: str) -> str:
    """Remove an indent from every line that starts with it.

    This is the inverse of indent_block. The splice functions, def_header
    and the edit vertical use it to bring a method to column 0.

    Args:
        text: str, function text at the def's indent.
        indent: str, the indent of the def line (leading_indent).

    Returns:
        str: the text at column 0.
    """
    if not indent:
        return text
    return "\n".join(line[len(indent):] if line.startswith(indent) else line
                     for line in text.split("\n"))


def leading_indent(line: str) -> str:
    """Give the spaces and tabs at the start of a line.

    The splice helpers and the edit vertical use it to learn the indent
    of a def line.

    Args:
        line: str, one line of source (with more lines, the first counts).

    Returns:
        str: the leading whitespace, "" at column 0.
    """
    return line[:len(line) - len(line.lstrip(LEADING_WS))]


def statement_spans(source: str, qualname: str,
                    lineno: int) -> list[tuple[int, int]]:
    """Give the line span of each statement in a def's body.

    The edit vertical offers these spans in its "delete the wrong line"
    menu. The docstring is never one of them. Methods work too.

    Args:
        source: str, the file text.
        qualname: str, "name" or "Class.method", from the symbol index.
        lineno: int, the def line (Symbol.lineno); it selects one of
            same-named defs.

    Returns:
        list[tuple[int, int]]: 1-based (start, end) per statement.
    """
    node = find_def(source, qualname, lineno)
    if node is None:
        return []
    return [(stmt.lineno, stmt.end_lineno)
            for stmt in without_docstring(node.body)]


def is_docstring(stmt: ast.AST) -> bool:
    """Tell if a statement is a string literal (a docstring).

    Args:
        stmt: ast.AST, the first statement of a body.

    Returns:
        bool: True for a bare string expression.
    """
    return isinstance(stmt, ast.Expr) \
        and isinstance(stmt.value, ast.Constant) \
        and isinstance(stmt.value.value, str)


def without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Give a def body without its docstring statement.

    Args:
        body: list[ast.stmt], the body of a parsed def.

    Returns:
        list[ast.stmt]: the body, minus a leading docstring.
    """
    if body and is_docstring(body[0]):
        return body[1:]
    return body


if __name__ == "__main__":
    src = "def f(x):\n    return x + 1\n\n\ndef g(y):\n    return y\n"
    assert statement_spans(src, "f", 1) == [(2, 2)]
    assert replace_span(src, 1, 2, "def f(x):\n    return 2").endswith("y\n")
    inserted = insert_after_span(src, 2, "def h(z):\n    return z")
    assert "def h(z):" in inserted and ast.parse(inserted)
    assert "return x + 1" not in delete_span(src, 2, 2)
    assert span_text(src, 2, 2) == "    return x + 1"

    fancy = ("class C:\n    @staticmethod\n    async def run(a, b=2, *rest,"
             "\n                  key: int = 0, **kw) -> int:\n"
             '        """Run it."""\n        return a\n')
    header = def_header(fancy, "C.run", 3)
    assert header.startswith("async def run(a, b=2") and header.endswith(":")
    assert def_header("def f(x): return x\n", "f", 1) == "def f(x):"
    assert statement_spans(fancy, "C.run", 3) == [(6, 6)]   # no docstring

    setter = ("class B:\n    @property\n    def w(self):\n"
              "        return self._w\n\n    @w.setter\n"
              "    def w(self, value):\n        self._w = value\n")
    assert def_header(setter, "B.w", 7) == "def w(self, value):"
    assert statement_spans(setter, "B.w", 7) == [(8, 8)]
    assert dedent_block(indent_block("a\n\nb", "    "), "    ") == "a\n\nb"
    print("smoke OK")
