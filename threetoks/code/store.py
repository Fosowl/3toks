"""The MethodStore: the harness-owned model of the file being written.

An ordered list of method records; the .py module is rendered
deterministically from it. A method with no accepted body renders as a
NotImplementedError stub, so the module is always importable — never a
half-written function. Declaration order is implementation order, so the
render is also the dependency order (see docs/DESIGN-coding-agent.md §3).
"""
from dataclasses import dataclass, field

from threetoks.code.gates import docstring

STATUS_PLANNED = "planned"
STATUS_TESTED = "tested"
STATUS_STUBBED = "stubbed"
STUB_BODY = "    raise NotImplementedError"


@dataclass
class MethodRecord:
    """One planned method and everything the harness learns about it."""
    name: str
    args: str
    contract: str
    example: str | None = None      # trusted assert expr, or None
    body: str | None = None         # accepted source, or None
    status: str = STATUS_PLANNED
    verified: bool = False          # passed its asserts (not just gated)
    attempts: int = 0
    repairs: int = 0                # times the runner unlocked it to fix
    asserts: list[str] = field(default_factory=list)
    seen_hashes: set[int] = field(default_factory=set)


class MethodStore:
    """Ordered method records plus the deterministic module render."""

    def __init__(self, goal: str):
        self.goal = goal
        self.methods: list[MethodRecord] = []

    def add(self, name: str, args: str, contract: str,
            example: str | None = None) -> MethodRecord:
        """Append a planned method; ignore a duplicate name."""
        if any(method.name == name for method in self.methods):
            return self._by_name(name)
        record = MethodRecord(name, args, contract, example)
        self.methods.append(record)
        return record

    def _by_name(self, name: str) -> MethodRecord:
        """The record with this name (assumes it exists)."""
        return next(m for m in self.methods if m.name == name)

    def names(self) -> list[str]:
        """All planned method names, in declaration order."""
        return [method.name for method in self.methods]

    def bodied_except(self, index: int) -> list[MethodRecord]:
        """Every record with a body other than the one at ``index``.

        During a repair round an early method is revisited while later
        siblings already have bodies; those are callable too, so the
        prompt context should list them all, not just the earlier ones.
        """
        return [m for i, m in enumerate(self.methods)
                if i != index and m.body is not None]

    def signature_lines(self, records: list[MethodRecord]) -> str:
        """Signature + contract lines for prompt context, one per method."""
        if not records:
            return "(none)"
        return "\n".join(f"{m.name}({m.args}): {m.contract}" for m in records)

    def render_module(self) -> str:
        """The full .py text: module docstring then every method or stub."""
        header = docstring(self.goal, "")
        blocks = [self._render_one(method) for method in self.methods]
        return "\n\n\n".join([header, *blocks]) + "\n" if blocks else header + "\n"

    def render_script(self, entry: str | None) -> str:
        """The module plus a ``__main__`` guard calling ``entry``.

        This is the *deliverable* render only: gates and the runner keep
        using the guard-free ``render_module`` so candidate imports never
        execute the entry while siblings are still stubs. ``None`` (no
        runnable entry) renders the plain module.
        """
        module = self.render_module()
        if entry is None:
            return module
        return f'{module}\n\nif __name__ == "__main__":\n    {entry}()\n'

    def _render_one(self, method: MethodRecord) -> str:
        """A method's accepted body, or a NotImplementedError stub."""
        if method.body is not None:
            return method.body.rstrip()
        return (f"def {method.name}({method.args}):\n"
                f"{docstring(method.contract)}\n{STUB_BODY}")


if __name__ == "__main__":
    store = MethodStore("count words in a file")
    store.add("read_text", "path", "read a file into a string")
    store.add("read_text", "path", "dup ignored")
    counts = store.add("tally", "words", "count each word", 'tally(["a","a"])=={"a":2}')
    counts.body = "def tally(words):\n    return {}"
    counts.status = STATUS_TESTED
    module = store.render_module()
    assert store.names() == ["read_text", "tally"], store.names()
    assert "raise NotImplementedError" in module      # read_text stubbed
    assert "return {}" in module                       # tally bodied
    import ast
    ast.parse(module)                                  # always importable
    assert store.render_script(None) == module         # no entry, no guard
    script = store.render_script("tally")
    assert script.endswith('if __name__ == "__main__":\n    tally()\n'), script
    ast.parse(script)                                  # invariant holds
    assert store.signature_lines(store.methods).count("\n") == 1
    hostile = MethodStore('goal with """ triple')
    hostile.add("f", "x", 'contract with """ triple')
    ast.parse(hostile.render_module())                 # invariant holds
    print("smoke OK")
