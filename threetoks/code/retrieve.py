"""Retrieval-as-repair: fetch a classic function instead of stubbing it.

Spike E9 measured this pipeline at 7/8 on classic functions (including
both E6 generation failures) with zero false accepts on invented names:
search the public web, extract candidate defs from ``<pre>``/``<code>``
blocks and raw files, gate them with a TIGHT whitelist (foreign code gets
none of the trust the harness's own generations get), and accept only a
candidate that passes the trusted anchor asserts in a subprocess. No
model call anywhere — the subprocess is the judge.

Scope discipline, all E9-measured:
- REPAIR-ONLY: generation wins 83% first-try (E5); the network is spent
  exclusively on methods that exhausted their attempts and would stub.
- ANCHOR-REQUIRED: without a trusted example there is no free judge for
  foreign code, so retrieval never fires (also the E9 lesson: prefer two
  anchors — one anchor let a boundary-broken is_prime through).
- OFF BY DEFAULT (``[code] retrieval`` in config.ini): executing and
  embedding internet code has licensing and sandboxing implications the
  accepted-snippet provenance header only documents, not solves.

Everything network-flavoured imports lazily (requests/bs4 are the ``web``
extra; a bare install must still import this module).
"""
import ast
import difflib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from threetoks.code import gates

# Tight stdlib whitelist: pure computation only. Foreign code never gets
# filesystem / process / network / reflection access.
SAFE_IMPORT_WHITELIST = frozenset({
    "math", "re", "string", "functools", "itertools", "collections",
    "operator", "textwrap", "unicodedata", "statistics", "fractions",
    "decimal", "bisect", "heapq", "array", "cmath",
})

# Builtins that reach outside pure computation, blocked wholesale.
FORBIDDEN_BUILTINS = frozenset({
    "eval", "exec", "open", "__import__", "getattr", "setattr", "delattr",
    "compile", "globals", "locals", "vars", "input", "exit", "quit",
    "breakpoint", "memoryview", "help",
})

MIN_FUZZY_RATIO = 0.72
MAX_HELPER_HOPS = 2
MAX_URLS_PER_CALL = 6
MAX_CANDIDATES_TRIED = 8
FETCH_TIMEOUT_S = 8
_BUILTIN_NAMES = frozenset(dir(__builtins__)) if not isinstance(
    __builtins__, dict) else frozenset(__builtins__)


# ------------------------------------------------------------ extraction

def extract_candidates(file_text: str, name: str) -> list[str]:
    """Standalone, renamed, safety-gated snippets matching ``name``.

    Best name-match tier first (exact, substring, fuzzy). Every snippet
    is self-contained: whitelisted imports plus any sibling helper or
    constant the def needs (up to MAX_HELPER_HOPS), with the def renamed
    to ``name`` so the anchor asserts can call it.
    """
    try:
        tree = ast.parse(file_text)
    except (SyntaxError, ValueError):
        return []
    lines = file_text.splitlines()
    matched = _name_matched_defs(tree, name)
    snippets = []
    for node in matched:
        snippet = _assemble(node, name, tree, lines)
        if snippet is not None:
            snippets.append(snippet)
    return snippets


def _name_matched_defs(tree: ast.Module, name: str) -> list[ast.AST]:
    """Top-level defs whose name matches, best tier first."""
    scored = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        tier = _match_tier(node.name, name)
        if tier is not None:
            scored.append((tier, node))
    scored.sort(key=lambda pair: pair[0])
    return [node for _, node in scored]


def _match_tier(candidate: str, wanted: str) -> int | None:
    """0 exact, 1 substring, 2 fuzzy (>= MIN_FUZZY_RATIO), None no match."""
    cand, target = _normalize(candidate), _normalize(wanted)
    if cand == target:
        return 0
    if target in cand or cand in target:
        return 1
    if difflib.SequenceMatcher(None, cand, target).ratio() >= MIN_FUZZY_RATIO:
        return 2
    return None


def _normalize(name: str) -> str:
    return re.sub(r"[_\-]", "", name).lower()


def _assemble(node: ast.AST, canonical: str, tree: ast.Module,
              lines: list[str]) -> str | None:
    """One def -> standalone renamed snippet, or None on any gate reject."""
    if _uses_forbidden_builtin(node) or _has_star_import(tree):
        return None
    imports = _top_level_imports(tree)
    functions = _top_level_functions(tree)
    assigns = _top_level_assigns(tree)
    prelude: list[str] = []
    seen = {node.name}
    frontier = [(free, MAX_HELPER_HOPS) for free in _free_names(node)]
    while frontier:
        needed, depth = frontier.pop()
        if needed in seen:
            continue
        seen.add(needed)
        piece = _resolve(needed, imports, functions, assigns, lines, depth)
        if piece == "REJECT":
            return None
        if piece is None:
            continue
        prelude.append(piece)
        if needed in functions and depth > 0:
            frontier.extend((free, depth - 1)
                            for free in _free_names(functions[needed]))
    body = _rename_def(_node_source(lines, node), node.name, canonical)
    snippet = "\n".join([*prelude, body])
    return snippet if gates.parses(snippet) else None


def _resolve(name: str, imports: dict, functions: dict, assigns: dict,
             lines: list[str], depth: int) -> str | None:
    """Source text that binds ``name``, None if unknown, "REJECT" if unsafe."""
    if name in imports:
        module = imports[name]
        return f"import {module}" if module in SAFE_IMPORT_WHITELIST \
            else "REJECT"
    if name in assigns and depth > 0:
        return _node_source(lines, assigns[name])
    if name in functions and depth > 0:
        helper = functions[name]
        return "REJECT" if _uses_forbidden_builtin(helper) \
            else _node_source(lines, helper)
    return None


def _top_level_imports(tree: ast.Module) -> dict:
    """Bound name -> ROOT module for every top-level import."""
    bound = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.split(".")[0]] = \
                    alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name != "*":
                    bound[alias.asname or alias.name] = \
                        node.module.split(".")[0]
    return bound


def _has_star_import(tree: ast.Module) -> bool:
    return any(isinstance(node, ast.ImportFrom)
               and any(a.name == "*" for a in node.names)
               for node in tree.body)


def _top_level_functions(tree: ast.Module) -> dict:
    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _top_level_assigns(tree: ast.Module) -> dict:
    """Plain ``NAME = ...`` constants (lookup tables etc.)."""
    return {n.targets[0].id: n for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)}


def _uses_forbidden_builtin(node: ast.AST) -> bool:
    """Bare forbidden-builtin names plus a dunder-access trip wire.

    Best effort, not a hardened sandbox — the subprocess timeout and the
    whitelist carry the real weight; this catches the obvious escapes.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in FORBIDDEN_BUILTINS:
            return True
        if isinstance(n, ast.Attribute) and n.attr.startswith("__") \
                and n.attr.endswith("__") and n.attr != "__name__":
            return True
    return False


def _free_names(node: ast.AST) -> set[str]:
    """Load-context names the def does not bind itself (conservative)."""
    args = node.args
    bound = {a.arg for a in [*args.args, *args.posonlyargs, *args.kwonlyargs]}
    bound |= {v.arg for v in (args.vararg, args.kwarg) if v}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and n is not node:
            bound.add(n.name)
    loads = {n.id for n in ast.walk(node)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loads - bound - _BUILTIN_NAMES


def _rename_def(source: str, old: str, new: str) -> str:
    """Rename the def line to the canonical name the anchors call."""
    lines = source.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        for prefix in (f"def {old}(", f"async def {old}("):
            if stripped.startswith(prefix):
                indent = line[:len(line) - len(stripped)]
                renamed = prefix.replace(f"{old}(", f"{new}(")
                lines[i] = indent + renamed + stripped[len(prefix):]
                return "\n".join(lines)
    return source


def _node_source(lines: list[str], node: ast.AST) -> str:
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


# ------------------------------------------------------------ fetching

def html_code_blocks(html: str) -> list[str]:
    """Text of every ``<pre>``/``<code>`` block (the E9 winning path —
    every accepted benchmark snippet came from an explained-code page,
    none from raw GitHub files)."""
    from bs4 import BeautifulSoup                    # web extra, lazy
    soup = BeautifulSoup(html, "html.parser")
    blocks = [tag.get_text() for tag in soup.find_all(["pre", "code"])]
    return [b for b in blocks if "def " in b]


def github_raw_url(url: str) -> str | None:
    """github.com blob URL -> raw.githubusercontent.com (not bot-walled)."""
    parsed = urlparse(url)
    if parsed.netloc != "github.com" or "/blob/" not in parsed.path:
        return None
    owner_repo, blob_path = parsed.path.lstrip("/").split("/blob/", 1)
    return f"https://raw.githubusercontent.com/{owner_repo}/{blob_path}"


class Retriever:
    """Callable retrieval-as-repair hook for CodeVertical.

    ``provider`` is any SearchProvider (the paced FallbackProvider chain);
    ``cache_path`` persists accepted snippets so the second miss on the
    same classic costs no network at all.
    """

    def __init__(self, provider, cache_path: str | None = None,
                 max_urls: int = MAX_URLS_PER_CALL,
                 max_candidates: int = MAX_CANDIDATES_TRIED):
        self.provider = provider
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_urls = max_urls
        self.max_candidates = max_candidates
        self._cache = self._load_cache()

    def __call__(self, name: str, args: str, contract: str,
                 examples: list[str]) -> str | None:
        """A snippet passing every anchor, or None (an honest miss)."""
        if not examples:
            return None
        if name in self._cache:
            return self._cache[name]
        asserts = [f"assert {example}" for example in examples]
        tried = 0
        for url, text in self._fetched_texts(name, contract):
            for block in self._code_blocks(url, text):
                for snippet in extract_candidates(block, name):
                    if tried >= self.max_candidates:
                        return None
                    tried += 1
                    ok, _ = gates.run_asserts(snippet, asserts)
                    if ok:
                        return self._accept(name, snippet, url)
        return None

    def _fetched_texts(self, name: str, contract: str):
        """(url, page text) for each search hit, capped and best-effort."""
        fetched = 0
        for query in (f'python "def {name}"',
                      f"python {name} function {contract}"):
            for result in self._search(query):
                if fetched >= self.max_urls:
                    return
                text = self._fetch(github_raw_url(result.url) or result.url)
                if text is not None:
                    fetched += 1
                    yield result.url, text

    def _search(self, query: str) -> list:
        try:
            return self.provider.search(query) or []
        except Exception:
            return []

    def _fetch(self, url: str) -> str | None:
        import requests                              # web extra, lazy
        from threetoks.web.search import BROWSER_USER_AGENT
        try:
            response = requests.get(
                url, headers={"User-Agent": BROWSER_USER_AGENT},
                timeout=FETCH_TIMEOUT_S)
            response.raise_for_status()
            return response.text
        except Exception:
            return None

    def _code_blocks(self, url: str, text: str) -> list[str]:
        """The candidate code units one fetched document yields."""
        if url.endswith(".py") or "raw.githubusercontent.com" in url:
            return [text]
        return html_code_blocks(text)

    def _accept(self, name: str, snippet: str, url: str) -> str:
        """Stamp provenance, cache, and return the winning snippet."""
        stamped = (f"# retrieved from {url}\n"
                   "# license unreviewed — verify before redistribution\n"
                   f"{snippet}")
        self._cache[name] = stamped
        self._save_cache()
        return stamped

    def _load_cache(self) -> dict:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=1))


if __name__ == "__main__":
    fixture = (
        "ROMANS = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500,"
        " 'M': 1000}\n\n"
        "def romanToInt(s):\n"
        "    total = 0\n"
        "    for i in range(len(s)):\n"
        "        value = ROMANS[s[i]]\n"
        "        if i + 1 < len(s) and ROMANS[s[i + 1]] > value:\n"
        "            total -= value\n"
        "        else:\n"
        "            total += value\n"
        "    return total\n")
    [snippet] = extract_candidates(fixture, "roman_to_int")
    assert "def roman_to_int(" in snippet and "ROMANS" in snippet
    ok, err = gates.run_asserts(snippet,
                                ['assert roman_to_int("MCMXCIV") == 1994'])
    assert ok, err

    unsafe = "import os\n\ndef roman_to_int(s):\n    return os.getpid()\n"
    assert extract_candidates(unsafe, "roman_to_int") == []
    golfed = "def roman_to_int(s):\n    return eval('0')\n"
    assert extract_candidates(golfed, "roman_to_int") == []
    assert extract_candidates("def f(:\n", "roman_to_int") == []

    assert github_raw_url("https://github.com/a/b/blob/main/x.py") == \
        "https://raw.githubusercontent.com/a/b/main/x.py"
    assert github_raw_url("https://example.com/a/b/blob/main/x.py") is None

    class _Provider:
        def search(self, query, max_results=8):
            return []

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "snippets.json"
        retriever = Retriever(_Provider(), cache_path=str(cache))
        assert retriever("roman_to_int", "s", "convert numerals",
                         ['roman_to_int("X") == 10']) is None   # honest miss
        retriever._accept("roman_to_int", snippet, "fixture://x")
        warm = Retriever(_Provider(), cache_path=str(cache))
        cached = warm("roman_to_int", "s", "convert numerals",
                      ['roman_to_int("X") == 10'])
        assert cached and "retrieved from fixture://x" in cached
    print("smoke OK")
