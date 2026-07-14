"""Provenance-tracked note store.

Notes are verbatim sentences the model chose (via PICK_MANY over sentence
indices), stored with their source URL and sentence index. They are
hallucination-free by construction: the harness copies content, the model
only points at it (see docs/DESIGN.md §8). The store is append-only and
skips exact-duplicate texts; ``render`` produces compact numbered lines
for the rolling log.
"""
import re
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_MAX_NOTES = 30
NEAR_DUP_PREFIX_CHARS = 60


@dataclass(frozen=True)
class Note:
    """One saved quote with its provenance."""

    text: str
    source_url: str
    sentence_idx: int


def _host_of(url: str) -> str:
    """Return the hostname of a URL, or the URL itself if unparsable."""
    host = urlparse(url).netloc
    return host or url


def rank_by_overlap(texts: list[str], query: str, top_k: int) -> list[str]:
    """The ``top_k`` texts sharing the most 4+-letter words with ``query``.

    Deterministic and free: answer fallbacks use it to quote the notes
    closest to the task instead of whichever happened to be stored first.
    Ties keep the original order (stable sort); a query with no usable
    keywords falls back to the head of the list.
    """
    keywords = set(re.findall(r"\w{4,}", query.lower()))
    if not keywords:
        return texts[:top_k]

    def overlap(text: str) -> int:
        lowered = text.lower()
        return sum(word in lowered for word in keywords)

    return sorted(texts, key=overlap, reverse=True)[:top_k]


class NoteStore:
    """Append-only collection of provenance-tracked notes."""

    def __init__(self):
        """Start with no notes."""
        self._notes: list[Note] = []
        self._seen_texts: set[str] = set()
        self._seen_prefixes: set[str] = set()

    def add(self, text: str, source_url: str, sentence_idx: int) -> None:
        """Append a note, skipping exact and near-duplicate texts.

        Near-dup drop: a note whose first ``NEAR_DUP_PREFIX_CHARS``
        lowercased characters match an existing note is dropped, so
        boilerplate that reappears with a trailing tweak does not clog
        the store before curation.
        """
        prefix = text[:NEAR_DUP_PREFIX_CHARS].lower()
        if text in self._seen_texts or prefix in self._seen_prefixes:
            return
        self._seen_texts.add(text)
        self._seen_prefixes.add(prefix)
        self._notes.append(Note(text, source_url, sentence_idx))

    def entries(self) -> list[Note]:
        """Stored notes in insertion order (public accessor)."""
        return list(self._notes)

    def select(self, indices: list[int]) -> "NoteStore":
        """New store with the 1-based ``indices`` notes, in that order.

        The given index order becomes the new store order (so a curation
        ranking is preserved), and provenance carries over unchanged.
        Out-of-range indices are skipped.
        """
        picked = NoteStore()
        for index in indices:
            if 1 <= index <= len(self._notes):
                note = self._notes[index - 1]
                picked.add(note.text, note.source_url, note.sentence_idx)
        return picked

    def texts(self) -> list[str]:
        """Note texts in insertion order (for extractive answering)."""
        return [note.text for note in self._notes]

    def count(self) -> int:
        """Return the number of stored notes."""
        return len(self._notes)

    def render(self, max_notes: int = DEFAULT_MAX_NOTES) -> str:
        """Format up to ``max_notes`` notes as numbered source-tagged lines."""
        lines = [
            f"[{index}] {note.text} (source: {_host_of(note.source_url)})"
            for index, note in enumerate(self._notes[:max_notes], start=1)
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    store = NoteStore()
    long_a = "The capital city of the French Republic is the city of Paris, a hub."
    long_b = "The capital city of the French Republic is the city of Paris, and big."
    store.add(long_a, "https://en.wikipedia.org/wiki/Paris", 3)
    store.add(long_a, "https://other.com/x", 1)  # exact dupe
    store.add(long_b, "https://other.com/y", 2)  # near-dup: shared 60-char head
    store.add("It has 2.1 million people.", "https://en.wikipedia.org/wiki/Paris", 7)
    assert store.count() == 2, store.count()
    rendered = store.render()
    assert rendered.startswith(f"[1] {long_a} "
                               "(source: en.wikipedia.org)"), rendered
    assert "[2] It has 2.1 million people." in rendered, rendered
    ranked = store.select([2, 1])  # reorder: preserves index order
    assert ranked.count() == 2, ranked.count()
    assert ranked.entries()[0].text == "It has 2.1 million people.", ranked
    closest = rank_by_overlap(store.texts(), "how many people live there?", 1)
    assert closest == ["It has 2.1 million people."], closest
    assert rank_by_overlap(["a", "b"], "??", 1) == ["a"]  # no keywords: head
    print("smoke OK")
