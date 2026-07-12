"""Deterministic symbol index: one ast pass over a corpus of .py files.

E8 thesis: editing existing code is selection-dominated. Before any model
call, the harness should already know exactly where every function, class,
and method lives, and where each name gets called. This module builds that
index for zero tokens (pure ast) and answers "where is X" for free when the
name is unique in the corpus -- only an ambiguous name costs one menu
decision (see nav.py / vertical.py for the model-facing side).
"""
import ast
from dataclasses import dataclass
from pathlib import Path

FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class Symbol:
    """One top-level function/class or one method, with its file span."""
    name: str          # bare name, e.g. "slice_page" or "validate"
    qualname: str       # "slice_page" or "Paginator.slice_page"
    file: str            # posix path relative to the corpus root
    lineno: int           # 1-based, first line of the def/class
    end_lineno: int        # 1-based, inclusive last line
    kind: str                # "function" | "method" | "class"


class SymbolIndex:
    """Every top-level def/class + method in a corpus, plus call sites.

    Built once per corpus root with a plain ast.walk -- no model call is
    ever involved in construction or in an unambiguous lookup.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.symbols: list[Symbol] = []
        # callee bare-name -> [(file, lineno), ...] for every ast.Call found
        self.call_sites: dict[str, list[tuple[str, int]]] = {}
        self._files: dict[str, str] = {}   # relative path -> source text
        self._build()

    def _build(self) -> None:
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
        for node in tree.body:
            if isinstance(node, FUNC_TYPES):
                self.symbols.append(Symbol(node.name, node.name, rel,
                                           node.lineno, node.end_lineno,
                                           "function"))
            elif isinstance(node, ast.ClassDef):
                self.symbols.append(Symbol(node.name, node.name, rel,
                                           node.lineno, node.end_lineno,
                                           "class"))
                for sub in node.body:
                    if isinstance(sub, FUNC_TYPES):
                        qual = f"{node.name}.{sub.name}"
                        self.symbols.append(Symbol(sub.name, qual, rel,
                                                   sub.lineno, sub.end_lineno,
                                                   "method"))

    def _collect_calls(self, tree: ast.AST, rel: str) -> None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = _callee_name(node.func)
            if callee:
                self.call_sites.setdefault(callee, []).append((rel, node.lineno))

    # ------------------------------------------------------------ lookups

    def find_by_name(self, name: str) -> list[Symbol]:
        """Every symbol with this bare name -- 0, 1 (free), or many (menu)."""
        return [s for s in self.symbols if s.name == name]

    def source_of(self, rel_file: str) -> str:
        """The verbatim current text of one indexed file."""
        return self._files.get(rel_file) or (self.root / rel_file).read_text()

    def top_level_names(self, rel_file: str) -> list[tuple[str, str]]:
        """(name, kind) for every top-level function/class in one file, in
        source order -- the raw material for a "def" navigation menu."""
        return [(s.name, s.kind) for s in self.symbols if s.file == rel_file
                and s.qualname == s.name]

    def methods_of(self, rel_file: str, class_name: str) -> list[str]:
        """Method names of one class, in source order."""
        prefix = f"{class_name}."
        return [s.name for s in self.symbols
                if s.file == rel_file and s.qualname.startswith(prefix)]


def _callee_name(func_node: ast.AST) -> str | None:
    """The bare name of a call target: ``f()`` or ``obj.f()`` -> "f"."""
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        return func_node.attr
    return None


def list_folders(root: Path) -> list[str]:
    """Immediate subdirectories of root that contain at least one .py file."""
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.rglob("*.py")):
            out.append(child.name)
    return out


def list_files(folder: Path) -> list[str]:
    """.py filenames directly inside one folder, sorted."""
    return sorted(p.name for p in folder.iterdir()
                 if p.is_file() and p.suffix == ".py")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a").mkdir()
        (root / "a" / "one.py").write_text(
            "def foo(x):\n    return helper(x)\n\n\n"
            "def helper(x):\n    return x + 1\n\n\n"
            "class Thing:\n    def bar(self):\n        return 1\n")
        (root / "a" / "two.py").write_text("def foo(y):\n    return y\n")

        index = SymbolIndex(root)
        names = {s.name for s in index.symbols}
        assert names == {"foo", "helper", "Thing", "bar"}, names
        assert len(index.find_by_name("foo")) == 2          # ambiguous
        assert len(index.find_by_name("helper")) == 1        # unique -> free
        assert index.find_by_name("helper")[0].file == "a/one.py"
        assert index.call_sites["helper"] == [("a/one.py", 2)]
        assert index.methods_of("a/one.py", "Thing") == ["bar"]
        assert set(list_folders(root)) == {"a"}
        assert list_files(root / "a") == ["one.py", "two.py"]
        top = index.top_level_names("a/one.py")
        assert top == [("foo", "function"), ("helper", "function"),
                       ("Thing", "class")], top
    print("smoke OK")
