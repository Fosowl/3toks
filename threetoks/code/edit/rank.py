"""Rank indexed symbols against a request: exact names first, then BM25F.

The lexical score only ORDERS candidates; the model makes the choice
(edit vertical shortlist menu, code agent navigate menu). Scoring is a
field-weighted, length-normalised BM25 (BM25F): a long function with the
request words scattered through it does not beat a short function whose
name and docstring say the same thing. No regular expressions.
"""
import math
from collections import Counter

from threetoks.code.edit.index import CALLABLE_KINDS
from threetoks.code.edit.nav import PAGE_SIZE

SHORTLIST_SIZE = 2 * PAGE_SIZE      # > one page, so paging is reachable
EDITABLE_KINDS = CALLABLE_KINDS
IDENTIFIER_CHARS = "_."             # kept by exact_hits: Class.method
# Field weights: name x3, docstring x2, path x1, body x1.
FIELD_WEIGHTS = {"name": 3.0, "doc": 2.0, "path": 1.0, "body": 1.0}
BM25_K1 = 1.2
BM25_B = 0.75
MIN_TOKEN_CHARS = 2
MIN_STEM_CHARS = 3
PYTHON_SUFFIX = ".py"
# Suffixes stripped by the light stemmer, tried in this order.
STEM_SUFFIXES = ("ing", "ers", "er", "ed", "es", "s")
STOPWORDS = frozenset(
    "a an the it its is are was were be been to of on in at by for with as "
    "and or not no this that these those there here what which who when "
    "where why how do does did can could should would will shall may must "
    "i me my we our you your he she they them his her their so than then "
    "but if else elif while from into out up down about all any some each "
    "only also just very too please instead return self cls def class "
    "import pass raise try except finally none true false print len int "
    "str float list dict set range py fix fixes fixed bug bugs broken wrong "
    "code function functions method methods file make makes".split())


def tokens(text: str) -> list[str]:
    """Turn free text or code into comparable search words.

    rank.shortlist calls this for the request and every symbol field. It
    splits on non-alphanumerics, snake_case, and camelCase, lowercases,
    drops stopwords and 1-char words, and applies a light suffix stem.

    Args:
        text: str, a request, a qualname, a path, a docstring, or code.
    Returns:
        list[str]: stemmed words in text order (repeats kept).
    """
    parts: list[str] = []
    for chunk in words(text):
        parts.extend(_camel_parts(chunk))
    stemmed = [_stem(part.lower()) for part in parts]
    return [word for word in stemmed
            if len(word) >= MIN_TOKEN_CHARS and word not in STOPWORDS]


def words(text: str, keep: str = "") -> list[str]:
    """Split text into words without a regular expression.

    tokens, exact_hits, code/route.py, and oracle.py call this; it is the
    only word splitter of the code agent. Letters, digits, and the
    characters in ``keep`` stay; every other character splits.

    Args:
        text: str, any text (request, code, path).
        keep: str, extra characters that belong to a word (e.g. "_").
    Returns:
        list[str]: the words, original case, in text order.
    """
    cleaned = "".join(char if char.isalnum() or char in keep else " "
                      for char in text)
    return cleaned.split()


def _camel_parts(chunk: str) -> list[str]:
    """Split one alphanumeric run at camelCase boundaries.

    Only tokens calls this. "parseHTTPHeader" gives parse, HTTP, Header.

    Args:
        chunk: str, letters and digits only.
    Returns:
        list[str]: the parts, original case.
    """
    parts: list[str] = []
    start = 0
    for pos in range(1, len(chunk)):
        prev, char = chunk[pos - 1], chunk[pos]
        nxt = chunk[pos + 1] if pos + 1 < len(chunk) else ""
        lower_to_upper = prev.islower() and char.isupper()
        acronym_end = prev.isupper() and char.isupper() and nxt.islower()
        if lower_to_upper or acronym_end:
            parts.append(chunk[start:pos])
            start = pos
    parts.append(chunk[start:])
    return parts


def _stem(word: str) -> str:
    """Strip one common English suffix so word forms match.

    Only tokens calls this. parse, parser, parsing, parsed, parses all
    become "pars"; entries becomes "entry".

    Args:
        word: str, one lowercase word.
    Returns:
        str: the stem (never shorter than MIN_STEM_CHARS).
    """
    if word.endswith("ies") and len(word) - 2 >= MIN_STEM_CHARS:
        return word[:-3] + "y"
    for suffix in STEM_SUFFIXES:
        if suffix == "s" and word.endswith("ss"):
            break
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM_CHARS:
            word = word[:-len(suffix)]
            break
    if word.endswith("e") and len(word) - 1 >= MIN_STEM_CHARS:
        word = word[:-1]
    return word


def exact_hits(index, request: str, kinds=EDITABLE_KINDS) -> list:
    """Find the symbols the request names verbatim.

    The edit vertical and the code agent call this first: a unique hit
    needs no model call. "Class.method" in the request selects only that
    qualname; else a bare name (case-sensitive, whole word) selects every
    symbol with that name.

    Args:
        index: SymbolIndex, the corpus index.
        request: str, the user's request text.
        kinds: tuple[str, ...], the Symbol kinds that may match.
    Returns:
        list[Symbol]: matches in index order.
    """
    dotted, bare = _named_identifiers(request)
    allowed = [s for s in index.symbols if s.kind in kinds]
    by_qualname = [s for s in allowed if s.qualname in dotted]
    if by_qualname:
        return by_qualname
    return [s for s in allowed if s.name in bare]


def _named_identifiers(request: str) -> tuple[set[str], set[str]]:
    """Collect the identifiers the request writes out.

    Only exact_hits calls this. File names ("mod.py") are ignored.

    Args:
        request: str, the user's request text.
    Returns:
        tuple[set[str], set[str]]: (dotted names, bare identifiers).
    """
    dotted: set[str] = set()
    bare: set[str] = set()
    for word in (raw.strip(".") for raw in words(request, IDENTIFIER_CHARS)):
        if not word or word.endswith(PYTHON_SUFFIX):
            continue
        parts = word.split(".")
        bare.update(part for part in parts if part.isidentifier())
        if len(parts) > 1:
            dotted.add(word)
    return dotted, bare


def shortlist(index, request: str, limit: int = SHORTLIST_SIZE,
              kinds=EDITABLE_KINDS) -> list:
    """Give the best candidate symbols for a request, best first.

    The edit vertical and the code agent call this when no unique exact
    name hit exists. Exact hits come first, then symbols with a BM25F
    score above zero; both groups are ordered by score, highest first.

    Args:
        index: SymbolIndex, the corpus index.
        request: str, the user's request text.
        limit: int, the maximum list length.
        kinds: tuple[str, ...], the Symbol kinds that may be listed.
    Returns:
        list[Symbol]: at most ``limit`` symbols.
    """
    allowed = [s for s in index.symbols if s.kind in kinds]
    query = set(tokens(request))
    scores = dict(zip(allowed, _bm25f(index, allowed, query)))
    hits = exact_hits(index, request, kinds)
    named = set(hits)
    rest = [s for s in allowed if s not in named and scores[s] > 0]
    ranked_hits = sorted(hits, key=scores.get, reverse=True)   # stable
    return (ranked_hits + sorted(rest, key=scores.get, reverse=True))[:limit]


def _bm25f(index, symbols: list, query: set[str]) -> list[float]:
    """Score every symbol against the query words with BM25F.

    Only shortlist calls this. Each field's term counts are weighted and
    divided by that field's relative length before saturation.

    Args:
        index: SymbolIndex, gives the body text through source_of.
        symbols: list[Symbol], the candidates.
        query: set[str], stemmed request words.
    Returns:
        list[float]: one score per symbol, same order.
    """
    if not symbols or not query:
        return [0.0] * len(symbols)
    docs = [_fields(index, symbol) for symbol in symbols]
    averages = _average_lengths(docs)
    idf = _idf(docs, query)
    return [_score(doc, idf, averages) for doc in docs]


def _fields(index, symbol) -> dict[str, Counter]:
    """Count the words of each scored field of one symbol.

    Only _bm25f calls this.

    Args:
        index: SymbolIndex, gives the body text.
        symbol: Symbol, one candidate.
    Returns:
        dict[str, Counter]: field name -> word counts.
    """
    lines = index.source_of(symbol.file).split("\n")
    body = "\n".join(lines[symbol.lineno - 1:symbol.end_lineno])
    texts = {"name": symbol.qualname, "doc": symbol.doc,
             "path": symbol.file, "body": body}
    return {field: Counter(tokens(text)) for field, text in texts.items()}


def _average_lengths(docs: list[dict[str, Counter]]) -> dict[str, float]:
    """Mean word count of each field over all candidates.

    Only _bm25f calls this. A field that is empty everywhere averages 1.

    Args:
        docs: list[dict[str, Counter]], from _fields.
    Returns:
        dict[str, float]: field name -> mean length (never 0).
    """
    return {field: max(1.0, sum(sum(doc[field].values()) for doc in docs)
                       / len(docs))
            for field in FIELD_WEIGHTS}


def _idf(docs: list[dict[str, Counter]], query: set[str]) -> dict[str, float]:
    """Inverse document frequency of each query word.

    Only _bm25f calls this. A word in fewer candidates weighs more.

    Args:
        docs: list[dict[str, Counter]], from _fields.
        query: set[str], stemmed request words.
    Returns:
        dict[str, float]: word -> idf (always positive).
    """
    total = len(docs)
    idf = {}
    for word in query:
        having = sum(1 for doc in docs
                     if any(word in counts for counts in doc.values()))
        idf[word] = math.log(1 + (total - having + 0.5) / (having + 0.5))
    return idf


def _score(doc: dict[str, Counter], idf: dict[str, float],
           averages: dict[str, float]) -> float:
    """BM25F score of one candidate.

    Only _bm25f calls this.

    Args:
        doc: dict[str, Counter], the candidate's field counts.
        idf: dict[str, float], query word -> idf.
        averages: dict[str, float], mean field lengths.
    Returns:
        float: the score, 0.0 when no query word occurs.
    """
    score = 0.0
    for word, weight in idf.items():
        freq = 0.0
        for field, field_weight in FIELD_WEIGHTS.items():
            length = sum(doc[field].values()) / averages[field]
            norm = 1 - BM25_B + BM25_B * length
            freq += field_weight * doc[field][word] / norm
        score += weight * freq / (BM25_K1 + freq)
    return score


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from threetoks.code.edit.index import SymbolIndex

    assert tokens("parseHTTPHeader") == ["pars", "http", "head"]
    assert tokens("the parser") == tokens("parsing") == tokens("parse")
    assert tokens("slice_page entries") == ["slic", "pag", "entry"]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "m.py").write_text(
            'def parse_header(text):\n    """Parse the header line."""\n'
            "    return text\n\n\ndef helper(x):\n    return x\n\n\n"
            "class Box:\n    def open(self):\n        return 1\n")
        index = SymbolIndex(root)
        assert [s.name for s in shortlist(index, "the parser is off")] == \
            ["parse_header"]
        assert shortlist(index, "parsing broke in helper")[0].name == "helper"
        assert [s.qualname for s in exact_hits(index, "fix Box.open()")] == \
            ["Box.open"]
        assert exact_hits(index, "Box is wrong") == []     # classes excluded
    print("smoke OK")
