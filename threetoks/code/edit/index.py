"""Deterministic symbol index: one ast pass over a corpus of .py files.

Editing existing code is selection-dominated (spike E8). Before any model
call, the harness already knows where every function, class, and method
lives, what each one calls, and where each name is called. This index is
built for zero tokens; rank.py orders its symbols into a shortlist and
the edit vertical asks the model to choose and confirm.
"""
import ast
import os
from dataclasses import dataclass
from pathlib import Path

from threetoks.code.edit.encoding import decode_source

FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
# Directories that are never user code; hidden dirs are skipped too.
SKIP_DIRS = frozenset({"__pycache__", "venv", "node_modules", "build",
                       "dist", "site-packages"})
EGG_INFO_SUFFIX = ".egg-info"
PYTHON_SUFFIX = ".py"
MAX_INDEXED_FILES = 500     # a corpus bigger than this is indexed partially
CALLABLE_KINDS = ("function", "method")


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
    """Every top-level def/class + method in a corpus, plus call data.

    Public data: ``symbols`` (source order per file), ``files`` (every
    indexed rel path, sorted), ``call_sites`` (callee bare name ->
    [(file, lineno)]), ``truncated`` (True when MAX_INDEXED_FILES bit).
    """

    def __init__(self, root: Path):
        """Build the index once; no later disk read.

        The edit vertical, the code agent, and the oracle create one index
        per request. They give the corpus root. This walks the tree,
        decodes and parses each .py file, and keeps its text in memory.

        Args:
            root: Path, the corpus root from Services.files_root or a test.
        """
        self.root = Path(root)
        self.symbols: list[Symbol] = []
        self.files: list[str] = []
        self.call_sites: dict[str, list[tuple[str, int]]] = {}
        self.truncated = False
        self._sources: dict[str, str] = {}
        self._calls: dict[Symbol, list[tuple[str, int]]] = {}
        self._build()

    def _build(self) -> None:
        """Index every decodable .py file under root, up to the cap.

        Only __init__ calls this. It reads each file found by
        _python_files, keeps the text, and collects defs and calls when
        the file parses. Unparseable files are kept as text only.
        """
        for path in _python_files(self.root):
            if len(self._sources) >= MAX_INDEXED_FILES:
                self.truncated = True
                break
            source = _read_python(path)
            if source is None:
                continue
            rel = path.relative_to(self.root).as_posix()
            self._sources[rel] = source
            self._parse_file(rel, source)
        self.files = sorted(self._sources)

    def _parse_file(self, rel: str, source: str) -> None:
        """Collect defs and calls of one decoded file.

        Only _build calls this, with the cached text of one file. A file
        that does not parse adds no symbols.

        Args:
            rel: str, posix path relative to root.
            source: str, the decoded file text.
        """
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return
        self._collect_defs(tree, rel)
        self._collect_calls(tree, rel)

    def _collect_defs(self, tree: ast.Module, rel: str) -> None:
        """Record top-level functions, classes, and each class's methods.

        Only _parse_file calls this with a parsed module. It adds one
        Symbol per def/class and remembers the calls inside each.

        Args:
            tree: ast.Module, the parsed file.
            rel: str, posix path relative to root.
        """
        for node in tree.body:
            if isinstance(node, FUNC_TYPES):
                self._add(node, node.name, rel, "function")
            elif isinstance(node, ast.ClassDef):
                self._add_class(node, rel)

    def _add_class(self, node: ast.ClassDef, rel: str) -> None:
        """Store one class and each of its methods.

        Only _collect_defs calls this, once per top-level class.

        Args:
            node: ast.ClassDef, the class node.
            rel: str, posix path relative to root.
        """
        self._add(node, node.name, rel, "class")
        for sub in node.body:
            if isinstance(sub, FUNC_TYPES):
                self._add(sub, f"{node.name}.{sub.name}", rel, "method")

    def _add(self, node: ast.AST, qualname: str, rel: str, kind: str) -> None:
        """Store one symbol and the calls inside its span.

        Only _collect_defs calls this, once per def or class node.

        Args:
            node: ast.AST, a FunctionDef, AsyncFunctionDef, or ClassDef.
            qualname: str, "name" or "Class.method".
            rel: str, posix path relative to root.
            kind: str, "function", "method", or "class".
        """
        symbol = _symbol(node, qualname, rel, kind)
        self.symbols.append(symbol)
        self._calls[symbol] = _calls_in(node)

    def _collect_calls(self, tree: ast.AST, rel: str) -> None:
        """Record every call site keyed by the callee's bare name.

        Only _parse_file calls this. The oracle reads ``call_sites``.

        Args:
            tree: ast.AST, the parsed file.
            rel: str, posix path relative to root.
        """
        for name, lineno in _calls_in(tree):
            self.call_sites.setdefault(name, []).append((rel, lineno))

    def find_by_name(self, name: str) -> list[Symbol]:
        """Every symbol with this bare name — 0, 1 (free), or many (menu)."""
        return [s for s in self.symbols if s.name == name]

    def source_of(self, rel_file: str) -> str:
        """Give the text of one file as it was when the index was built.

        The vertical, rank.py, and the code agent call this with a path
        from a Symbol. It reads the cache only, so it never touches disk.

        Args:
            rel_file: str, posix path relative to root.
        Returns:
            str: the decoded text (LF newlines), "" for an unknown path.
        """
        return self._sources.get(rel_file, "")

    def methods_of(self, rel_file: str, class_name: str) -> list[str]:
        """Give the method names of one class.

        The code agent calls this to outline a class in a navigate answer.

        Args:
            rel_file: str, posix path relative to root.
            class_name: str, the bare class name.
        Returns:
            list[str]: method names in source order.
        """
        prefix = f"{class_name}."
        return [s.name for s in self.symbols
                if s.file == rel_file and s.qualname.startswith(prefix)]

    def symbol_at(self, rel_file: str, qualname: str) -> Symbol | None:
        """The symbol with this qualname in this file, or None."""
        return next((s for s in self.symbols
                     if s.file == rel_file and s.qualname == qualname), None)

    def callees_of(self, symbol: Symbol) -> list[Symbol]:
        """List the functions and methods that one symbol calls.

        The edit vertical calls this to offer "the bug is in a function
        this one calls". It resolves each called bare name, in source
        order, without duplicates and without the symbol itself.

        Args:
            symbol: Symbol, an indexed symbol (from this index).
        Returns:
            list[Symbol]: resolved callees, source order.
        """
        found: list[Symbol] = []
        for name, _ in self._calls.get(symbol, []):
            callee = self._resolve(symbol, name)
            if callee is not None and callee != symbol and callee not in found:
                found.append(callee)
        return found

    def _resolve(self, caller: Symbol, name: str) -> Symbol | None:
        """Find the one function or method a bare call name means.

        Only callees_of calls this. Order: a method of the caller's own
        class, then a unique name in the same file, then a unique name in
        the whole corpus.

        Args:
            caller: Symbol, the symbol that makes the call.
            name: str, the bare callee name from the call.
        Returns:
            Symbol | None: the callee, or None when not resolvable.
        """
        same_file = [s for s in self.find_by_name(name)
                     if s.kind in CALLABLE_KINDS and s.file == caller.file]
        owner = caller.qualname.rpartition(".")[0]
        same_class = [s for s in same_file
                      if owner and s.qualname == f"{owner}.{name}"]
        if same_class:
            return same_class[0]
        if len(same_file) == 1:
            return same_file[0]
        corpus = [s for s in self.find_by_name(name)
                  if s.kind in CALLABLE_KINDS]
        return corpus[0] if len(corpus) == 1 and not same_file else None


def _python_files(root: Path):
    """Yield every .py file under root, junk directories pruned.

    Only SymbolIndex._build calls this. It walks with os.walk and removes
    hidden, SKIP_DIRS, egg-info, and symlinked directories before descent
    (a symlink could point outside the root). Symlinked files are skipped.

    Args:
        root: Path, the corpus root.
    Returns:
        Iterator[Path]: file paths in a stable sorted order.
    """
    for folder, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if _keep_dir(Path(folder) / d))
        paths = (Path(folder) / name for name in sorted(filenames))
        yield from (path for path in paths if _is_python_file(path))


def _is_python_file(path: Path) -> bool:
    """Tell if the walk indexes one file.

    Only _python_files calls this for each file it finds.

    Args:
        path: Path, one file under the root.
    Returns:
        bool: True for a .py file that is not a symlink.
    """
    return path.name.endswith(PYTHON_SUFFIX) and not path.is_symlink()


def _keep_dir(path: Path) -> bool:
    """Tell if the walk may enter one directory.

    Only _python_files calls this for each subdirectory it finds.

    Args:
        path: Path, the subdirectory.
    Returns:
        bool: False for hidden, junk, egg-info, or symlinked directories.
    """
    name = path.name
    if name.startswith(".") or name in SKIP_DIRS:
        return False
    return not name.endswith(EGG_INFO_SUFFIX) and not path.is_symlink()


def _read_python(path: Path) -> str | None:
    """Decode one Python file the way the interpreter does.

    Only SymbolIndex._build calls this. decode_source honours a BOM
    and a coding cookie and gives LF newlines.

    Args:
        path: Path, one .py file.
    Returns:
        str | None: the text, or None when the file cannot be read or
        decoded.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return decode_source(raw)


def _symbol(node: ast.AST, qualname: str, rel: str, kind: str) -> Symbol:
    """Build a Symbol from a def/class node, capturing its doc first line."""
    doc = (ast.get_docstring(node) or "").split("\n", 1)[0]
    return Symbol(node.name, qualname, rel, node.lineno, node.end_lineno,
                  kind, doc)


def _calls_in(node: ast.AST) -> list[tuple[str, int]]:
    """List the bare callee names called inside one node.

    _add and _collect_calls call this with a def, class, or module node.

    Args:
        node: ast.AST, any parsed node.
    Returns:
        list[tuple[str, int]]: (bare name, lineno), source order.
    """
    calls = [sub for sub in ast.walk(node) if isinstance(sub, ast.Call)]
    calls.sort(key=lambda call: (call.lineno, call.col_offset))
    named = [(_callee_name(call.func), call.lineno) for call in calls]
    return [(name, lineno) for name, lineno in named if name]


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
            "class Thing:\n    def bar(self):\n        return self.baz()\n\n"
            "    def baz(self):\n        return foo(1)\n")
        (root / "a" / "two.py").write_text("def foo(y):\n    return y\n")
        (root / "a" / "empty.py").write_text("X = 1\n")

        index = SymbolIndex(root)
        assert {s.name for s in index.symbols} == {"foo", "helper", "Thing",
                                                   "bar", "baz"}
        assert index.files == ["a/empty.py", "a/one.py", "a/two.py"]
        assert len(index.find_by_name("foo")) == 2           # ambiguous
        assert index.find_by_name("foo")[0].doc == "Add via helper."
        assert index.call_sites["helper"] == [("a/one.py", 3)]
        foo = index.symbol_at("a/one.py", "foo")
        bar = index.symbol_at("a/one.py", "Thing.bar")
        assert [s.name for s in index.callees_of(foo)] == ["helper"]
        assert [s.qualname for s in index.callees_of(bar)] == ["Thing.baz"]
        assert index.methods_of("a/one.py", "Thing") == ["bar", "baz"]
        (root / "a" / "one.py").unlink()
        assert "def helper" in index.source_of("a/one.py")   # cached text
        assert index.source_of("nope.py") == ""

        (root / "node_modules").mkdir()
        (root / "node_modules" / "junk.py").write_text("def junky():\n    pass\n")
        (root / ".venv").mkdir()
        (root / ".venv" / "secret.py").write_text("def hush():\n    pass\n")
        (root / "bad.py").write_bytes(b"def bad():\n    return '\xff'\n")
        pruned = SymbolIndex(root)
        names = {s.name for s in pruned.symbols}
        assert not names & {"junky", "hush", "bad"}
        assert not pruned.truncated
    print("smoke OK")
