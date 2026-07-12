"""Deterministic symbol index: one ast pass over a corpus of .py files.

Editing existing code is selection-dominated (spike E8). Before any model
call, the harness already knows where every function, class, and method
lives and where each name is called. This index is built for zero tokens;
"where is X" is answered free when the name is unique in the corpus, and
an ambiguous name costs at most one menu decision (see vertical.py).
"""
import ast
from dataclasses import dataclass
from pathlib import Path

FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Symbol:
    """One top-level function/class or one method, with its file span."""
    name: str        # bare name, e.g. "slice_page" or "validate"
    qualname: str    # "slice_page" or "Paginator.slice_page"
    file: str        # posix path relative to the corpus root
    lineno: int      # 1-based, first line of the def/class
    end_lineno: int  # 1-based, inclusive last line
    kind: str        # "function" | "method" | "class"
    doc: str = ""    # first docstring line, "" when absent


class SymbolIndex:
    """Every top-level def/class + method in a corpus, plus call sites."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.symbols: list[Symbol] = []
        # callee bare-name -> [(file, lineno), ...] for every ast.Call
        self.call_sites: dict[str, list[tuple[str, int]]] = {}
        self._files: dict[str, str] = {}   # relative path -> source text
        self._build()

    def _build(self) -> None:
        """One parse per file; unparseable files are indexed as text only."""
        for path in sorted(self.root.rglob("*.py")):
            rel = path.relative_to(self.root).as_posix()
            source = path.read_text()
            self._files[rel] = source
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                continue
            self._collect_defs(tree, rel)
            self._collect_calls(tree, rel)

    def _collect_defs(self, tree: ast.AST, rel: str) -> None:
        """Record top-level functions, classes, and each class's methods."""
        for node in tree.body:
            if isinstance(node, FUNC_TYPES):
                self.symbols.append(_symbol(node, node.name, rel, "function"))
            elif isinstance(node, ast.ClassDef):
                self.symbols.append(_symbol(node, node.name, rel, "class"))
                for sub in node.body:
                    if isinstance(sub, FUNC_TYPES):
                        qual = f"{node.name}.{sub.name}"
                        self.symbols.append(_symbol(sub, qual, rel, "method"))

    def _collect_calls(self, tree: ast.AST, rel: str) -> None:
        """Record every call site keyed by the callee's bare name."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee_name(node.func)
            if callee:
                self.call_sites.setdefault(callee, []).append((rel, node.lineno))

    def find_by_name(self, name: str) -> list[Symbol]:
        """Every symbol with this bare name — 0, 1 (free), or many (menu)."""
        return [s for s in self.symbols if s.name == name]

    def source_of(self, rel_file: str) -> str:
        """The current on-disk text of one indexed file."""
        return (self.root / rel_file).read_text()

    def top_level_names(self, rel_file: str) -> list[tuple[str, str]]:
        """(name, kind) per top-level def/class in one file, source order."""
        return [(s.name, s.kind) for s in self.symbols
                if s.file == rel_file and s.qualname == s.name]

    def methods_of(self, rel_file: str, class_name: str) -> list[str]:
        """Method names of one class, in source order."""
        prefix = f"{class_name}."
        return [s.name for s in self.symbols
                if s.file == rel_file and s.qualname.startswith(prefix)]

    def symbol_at(self, rel_file: str, qualname: str) -> Symbol | None:
        """The symbol with this qualname in this file, or None."""
        return next((s for s in self.symbols
                     if s.file == rel_file and s.qualname == qualname), None)


def _symbol(node: ast.AST, qualname: str, rel: str, kind: str) -> Symbol:
    """Build a Symbol from a def/class node, capturing its doc first line."""
    doc = (ast.get_docstring(node) or "").split("\n", 1)[0]
    return Symbol(node.name, qualname, rel, node.lineno, node.end_lineno,
                  kind, doc)


def _callee_name(func_node: ast.AST) -> str | None:
    """The bare name of a call target: ``f()`` or ``obj.f()`` -> "f"."""
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        return func_node.attr
    return None


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a").mkdir()
        (root / "a" / "one.py").write_text(
            'def foo(x):\n    """Add via helper."""\n    return helper(x)\n\n\n'
            "def helper(x):\n    return x + 1\n\n\n"
            "class Thing:\n    def bar(self):\n        return 1\n")
        (root / "a" / "two.py").write_text("def foo(y):\n    return y\n")

        index = SymbolIndex(root)
        assert {s.name for s in index.symbols} == {"foo", "helper", "Thing", "bar"}
        assert len(index.find_by_name("foo")) == 2           # ambiguous
        assert len(index.find_by_name("helper")) == 1        # unique -> free
        assert index.find_by_name("foo")[0].doc == "Add via helper."
        assert index.call_sites["helper"] == [("a/one.py", 3)]
        assert index.methods_of("a/one.py", "Thing") == ["bar"]
        assert index.top_level_names("a/one.py") == [
            ("foo", "function"), ("helper", "function"), ("Thing", "class")]
        assert index.symbol_at("a/one.py", "Thing.bar").kind == "method"
        assert index.symbol_at("a/one.py", "nope") is None
    print("smoke OK")
