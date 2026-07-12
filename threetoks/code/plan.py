"""Loose parser for the model's method plan.

E5c showed a 1.5b will not obey a rigid ``name(args): purpose`` line — it
drops parentheses, uses ``-`` or ``=`` for the separator, and echoes list
numbering. So the harness parses leniently rather than forcing the model
to be strict (docs/DESIGN-coding-agent.md §2/§3): recover a name, optional
args, a purpose, and an optional ``e.g. f(x)==y`` example per line.
"""
import ast
import re
from dataclasses import dataclass

MAX_METHODS = 4
_LEAD = re.compile(r"^\s*(?:\d+\s*[.)]\s*)?(?:def\s+)?")
_NAME = re.compile(r"([a-z_][a-z0-9_]*)")
_ARGS = re.compile(r"\(([^)]*)\)")
_SEP = re.compile(r"^\s*[:=\-]\s*")
_EXAMPLE = re.compile(r"e\.g\.?\s*(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedMethod:
    """One method recovered from a plan line."""
    name: str
    args: str
    contract: str
    advisory_assert: str | None = None


def parse_plan(text: str) -> list[ParsedMethod]:
    """Parse plan lines into methods; drop unparsable lines, dedupe names."""
    methods: list[ParsedMethod] = []
    seen: set[str] = set()
    for line in text.splitlines():
        parsed = _parse_line(line)
        if parsed is not None and parsed.name not in seen:
            seen.add(parsed.name)
            methods.append(parsed)
    return methods[:MAX_METHODS]


def _parse_line(line: str) -> ParsedMethod | None:
    """Recover a method from a declaration line; skip prose.

    A line qualifies only if it carries a structural signal — parentheses
    or a ``:``/``-``/``=`` separator after the name — so free prose that
    happens to start with a word is not mistaken for a method.
    """
    rest = _LEAD.sub("", line, count=1)
    name_match = _NAME.match(rest)
    if name_match is None:
        return None
    tail = rest[name_match.end():]
    args_match = _ARGS.match(tail)
    if args_match is not None:
        tail = tail[args_match.end():]
    if args_match is None and _SEP.match(tail) is None:
        return None
    args = args_match.group(1).strip() if args_match else ""
    contract, advisory = _split_example(_SEP.sub("", tail).strip())
    return ParsedMethod(name_match.group(1), args,
                        contract or f"implement {name_match.group(1)}", advisory)


def _split_example(purpose: str) -> tuple[str, str | None]:
    """Split a purpose into (contract, advisory assert) on an e.g. clause."""
    match = _EXAMPLE.search(purpose)
    if match is None:
        return purpose, None
    contract = purpose[:match.start()].strip(" ;,")
    return contract, _as_assert(match.group(1).strip(" .;"))


def _as_assert(expr: str) -> str | None:
    """Wrap an ``f(x)==y`` example as an assert, if it is a comparison."""
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):     # ValueError: null bytes
        return None
    if isinstance(tree.body, ast.Compare) and \
            any(isinstance(op, ast.Eq) for op in tree.body.ops):
        return f"assert {expr}"
    return None


if __name__ == "__main__":
    plan = parse_plan(
        "FUNCTIONS:\n"
        "1. count_vowels(text): counts vowels; e.g. count_vowels('hi')==1\n"
        "2. normalize - lowercase the text\n"
        "3. tokenize(text) = split into words\n"
        "1. count_vowels(text): duplicate name dropped\n"
        "just prose with no identifier start !!!")
    names = [m.name for m in plan]
    assert names == ["count_vowels", "normalize", "tokenize"], names
    assert plan[0].advisory_assert == "assert count_vowels('hi')==1", plan[0]
    assert plan[1].args == "" and plan[1].contract == "lowercase the text"
    assert plan[2].args == "text"
    assert _as_assert("f(1)") is None            # not a comparison
    assert len(parse_plan("\n".join(f"m{i}(): x" for i in range(9)))) == MAX_METHODS
    print("smoke OK")
