"""What each edit operation needs for one accepted target, for free.

The edit vertical creates one TargetOperations when it accepts a target
and asks it: which operations are viable, which helper is missing
(insert-after), which statements may be deleted (delete), and the
GenerateSpanNode for replace or insert-after. Everything is computed
from the indexed file text: zero model calls and no disk access.
"""
import math

from threetoks.code import gates
from threetoks.code.edit import nav
from threetoks.code.edit.names import find_missing_helper, module_level_names
from threetoks.code.edit.search import target_code
from threetoks.code.edit.splice import (GENERATE_MAX_TOKENS,
                                        GenerateSpanNode,
                                        insert_after_splice_fn,
                                        replace_splice_fn)
from threetoks.code.edit.spans import (body_prefill, context_lines,
                                       def_header, delete_span,
                                       leading_indent, span_text,
                                       statement_spans)
from threetoks.nodes import MenuNode

OP_REPLACE, OP_INSERT_AFTER, OP_DELETE = "replace", "insert_after", "delete"
OPERATION_LABELS = {
    OP_REPLACE: "replace the target function's body with corrected code",
    OP_INSERT_AFTER: "insert new helper code right after the target function",
    OP_DELETE: "delete the offending line(s) inside the target function",
}
OPERATION_QUESTION = "How should this be fixed?"
NO_HEADER = "could not read the def line of the target"
NO_HELPER = "insert-after: no undefined helper to add"
# Replace token cap: the rewrite may grow the original span.
CHARS_PER_TOKEN = 3
GROWTH_FACTOR = 1.5
MAX_GENERATE_TOKENS = 1200


class TargetOperations:
    """The free facts and generate nodes of one accepted edit target."""

    def __init__(self, index, target):
        """Read the target's file text and indentation from the index.

        The edit vertical creates one per accepted target.

        Args:
            index: SymbolIndex, the corpus index (cached file text).
            target: Symbol, the accepted function or method.
        """
        self.index = index
        self.target = target
        self.source = index.source_of(target.file)
        def_line = self.source.split("\n")[target.lineno - 1]
        self.indent = leading_indent(def_line)

    def known_names(self) -> set[str]:
        """Give the names the target file defines or imports.

        Returns:
            set[str]: module-level bindings plus same-file symbol names.
        """
        same_file = {s.name for s in self.index.symbols
                     if s.file == self.target.file}
        return module_level_names(self.source) | same_file

    def missing_helper(self) -> tuple[str, str] | None:
        """Give (name, args) of a helper the target calls but nobody defines.

        The def is found by its line, so same-named defs do not collide;
        a file with a star import never has a missing helper.

        Returns:
            tuple[str, str] | None: the helper, or None.
        """
        return find_missing_helper(self.source, self.target.lineno,
                                   self.known_names())

    def deletable_spans(self, tried: set) -> list[tuple[int, int]]:
        """List the statement spans that may be deleted.

        A span must be untried, below the def line (a one-liner's only
        statement shares it), and leave a file that still parses.

        Args:
            tried: set[tuple[int, int]], spans already deleted and failed.
        Returns:
            list[tuple[int, int]]: 1-based (start, end) spans.
        """
        target = self.target
        spans = statement_spans(self.source, target.qualname, target.lineno)
        return [s for s in spans
                if s not in tried and s[0] > target.lineno
                and gates.parses(delete_span(self.source, *s))]

    def viable(self, tried_deletes: set) -> list[str]:
        """List the operations that can run against this target.

        The edit vertical offers only these in its operation menu:
        insert-after needs a missing helper, delete needs a deletable
        statement. An impossible option only wastes a menu pick.

        Args:
            tried_deletes: set[tuple[int, int]], spans already tried.
        Returns:
            list[str]: operation ids, OP_REPLACE first.
        """
        viable = [OP_REPLACE]
        if self.missing_helper() is not None:
            viable.append(OP_INSERT_AFTER)
        if self.deletable_spans(tried_deletes):
            viable.append(OP_DELETE)
        return viable

    def delete_menu(self, tried: set):
        """Give the paged menu over the deletable statements.

        Args:
            tried: set[tuple[int, int]], spans already deleted and failed.
        Returns:
            DeleteMenu | None: None when no statement may be deleted.
        """
        spans = self.deletable_spans(tried)
        if not spans:
            return None
        labels = nav.uniquify([nav.clip(span_text(self.source, *s))
                               for s in spans])
        return DeleteMenu(f"Which line inside {self.target.name}() is "
                          "wrong?", dict(zip(labels, spans)))

    def without(self, span: tuple[int, int]) -> str:
        """Give the file text with one statement span deleted.

        Args:
            span: tuple[int, int], 1-based (start, end).
        Returns:
            str: the delete candidate (LF).
        """
        return delete_span(self.source, *span)

    def generate_node(self, operation: str,
                      tried: set) -> tuple[GenerateSpanNode | None, str]:
        """Give the generate node of replace or insert-after.

        The edit vertical calls this for every generation attempt.

        Args:
            operation: str, OP_REPLACE or OP_INSERT_AFTER.
            tried: set[str], canonical keys of candidates that failed.
        Returns:
            tuple[GenerateSpanNode | None, str]: (node, qualname the
            candidate changes or adds), or (None, why no node exists).
        """
        if operation == OP_REPLACE:
            node = self._replace_node(tried)
            return (node, self.target.qualname) if node else (None, NO_HEADER)
        helper = self.missing_helper()
        if helper is None:
            return None, NO_HELPER
        return self._insert_node(helper, tried), helper[0]

    def _replace_node(self, tried: set) -> GenerateSpanNode | None:
        """Give the replace node, prefilled with the verbatim def header.

        Args:
            tried: set[str], canonical keys of candidates that failed.
        Returns:
            GenerateSpanNode | None: None when the header is unreadable.
        """
        target, source = self.target, self.source
        header = def_header(source, target.qualname, target.lineno)
        if header is None:
            return None
        splice = replace_splice_fn(
            source, target.lineno, target.end_lineno, header, self.indent,
            self.known_names(), tried)
        before, after = context_lines(source, target.lineno,
                                      target.end_lineno)
        code = target_code(self.index, target)
        node = GenerateSpanNode(_replace_instruction(code, before, after),
                                body_prefill(header), splice)
        node.max_tokens = _token_cap(len(code))
        return node

    def _insert_node(self, helper: tuple[str, str],
                     tried: set) -> GenerateSpanNode:
        """Give the insert-after node for one missing helper.

        Args:
            helper: tuple[str, str], (name, args) from missing_helper.
            tried: set[str], canonical keys of candidates that failed.
        Returns:
            GenerateSpanNode: the node.
        """
        name, args = helper
        splice = insert_after_splice_fn(
            self.source, self._insert_line(), name, args,
            self.known_names() | {name}, tried)
        instruction = (
            f"Existing function (verbatim, do not repeat it):\n"
            f"{target_code(self.index, self.target)}\n"
            f"Write the missing helper function {name}({args}) that this "
            "code calls but that is never defined. Do not use pass or "
            "raise NotImplementedError — write a real body.")
        return GenerateSpanNode(
            instruction, body_prefill(f"def {name}({args}):"), splice)

    def _insert_line(self) -> int:
        """Give the line after which a helper lands (top level).

        Returns:
            int: the target's last line, or its class's last line.
        """
        if self.target.kind != "method":
            return self.target.end_lineno
        class_name = self.target.qualname.split(".", 1)[0]
        owner = self.index.symbol_at(self.target.file, class_name)
        return owner.end_lineno if owner else self.target.end_lineno


def operation_menu(viable: list[str]) -> MenuNode:
    """Give the operation menu over the viable operations.

    The edit vertical asks it when the operation is not free.

    Args:
        viable: list[str], operation ids from TargetOperations.viable.
    Returns:
        MenuNode: one option per operation, plus escape.
    """
    return MenuNode(OPERATION_QUESTION,
                    [OPERATION_LABELS[op] for op in viable], escape=True)


def operation_of(label: str) -> str:
    """Map an operation menu label back to its operation id.

    Args:
        label: str, one of the OPERATION_LABELS values (a menu answer).
    Returns:
        str: the operation id.
    """
    return next(op for op, text in OPERATION_LABELS.items() if text == label)


class DeleteMenu(nav.LevelChoice):
    """A paged statement menu that also maps each label to its span."""

    def __init__(self, question: str, spans: dict):
        """Build the menu over the span labels.

        Only TargetOperations.delete_menu creates one.

        Args:
            question: str, the menu question.
            spans: dict[str, tuple[int, int]], label -> 1-based span.
        """
        super().__init__(question, list(spans))
        self.spans = spans


def _replace_instruction(span: str, before: str, after: str) -> str:
    """Build the replace micro-prompt from the span and its context.

    Only TargetOperations.replace_node calls this. The request is already
    the TASK line of the fresh generate episode.

    Args:
        span: str, the target source at column 0.
        before: str, file lines just before the span.
        after: str, file lines just after the span.
    Returns:
        str: the instruction text.
    """
    lines = ["Current function (verbatim):", span]
    if before:
        lines += ["Lines just before it:", before]
    if after:
        lines += ["Lines just after it:", after]
    lines.append("Write this function with a corrected, complete body. Do "
                 "not use pass or raise NotImplementedError — write a real "
                 "fix.")
    return "\n".join(lines)


def _token_cap(span_chars: int) -> int:
    """Size the replace token cap to the target function.

    Only TargetOperations.replace_node calls this. A long function needs
    more tokens than the default cap, else every answer is cut off and
    rejected.

    Args:
        span_chars: int, the length of the original function text.
    Returns:
        int: between GENERATE_MAX_TOKENS and MAX_GENERATE_TOKENS.
    """
    wanted = math.ceil(span_chars / CHARS_PER_TOKEN * GROWTH_FACTOR)
    return min(MAX_GENERATE_TOKENS, max(GENERATE_MAX_TOKENS, wanted))


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from threetoks.code.edit.index import SymbolIndex

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "m.py").write_text(
            "class Box:\n    def total(self, xs):\n        n = 0\n"
            "        for x in xs:\n            n = _acc(n, x)\n"
            "        return n\n")
        index = SymbolIndex(root)
        ops = TargetOperations(index, index.symbol_at("m.py", "Box.total"))
        assert ops.indent == "    "
        assert ops.missing_helper() == ("_acc", "arg1, arg2")
        assert ops.deletable_spans(set()), "statements can be deleted"
        assert ops.viable(set()) == [OP_REPLACE, OP_INSERT_AFTER, OP_DELETE]
        replace, edited = ops.generate_node(OP_REPLACE, set())
        assert replace.prefill.startswith("def total(self, xs):")
        assert edited == "Box.total"
        insert, helper = ops.generate_node(OP_INSERT_AFTER, set())
        assert insert.prefill.startswith("def _acc(arg1, arg2):")
        assert helper == "_acc"
        assert len(ops.delete_menu(set()).spans) == 3
        menu = operation_menu(ops.viable(set()))
        assert operation_of(menu.options[-1]) == OP_DELETE
        assert _token_cap(10) == GENERATE_MAX_TOKENS
        assert _token_cap(10 ** 6) == MAX_GENERATE_TOKENS
    print("smoke OK")
