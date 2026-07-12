"""Cross-session memory: what each agent was asked and how it answered.

A dict keyed by agent name; each entry keeps the query, the final answer,
and a global ordinal so recency survives a JSON round-trip. The store is
dumped to JSON when a session ends and loaded again at startup. Quality
is enforced at the edges, all deterministically: ``worth_remembering``
refuses empty / "(no answer)" / judge-rejected answers, ``remember``
replaces same-query duplicates and caps each agent's history, and
``relevant`` shortlists selector candidates by keyword overlap — no
overlap means no model call at all.

After the router picks an agent, :func:`select_memories` shows the model
the shortlisted queries (with clipped answers) and lets it point at up to
a few that give useful context (or none) — one PickManyNode decision, the
same "point at numbered items" shape the note-taking verticals use. The
picks land in ``services.recalled``; :func:`render_recall` turns them
into the context block agents log into their episodes.
"""
import json
import re
from pathlib import Path

from threetoks.nodes import PickManyNode
from threetoks.render import Episode

RECENT_LIMIT = 32
MAX_PER_AGENT = 64
MAX_MEMORY_PICKS = 3
SHOW_LIMIT = 8
QUERY_CLIP = 100
SELECTOR_ANSWER_CLIP = 60
RECALL_ANSWER_CLIP = 160
NO_ANSWER = "(no answer)"
SELECTOR_PREFIX = ("You pick which earlier questions give useful background "
                   "for a new request. Choose only ones that truly relate.")
_KEYWORD_RE = re.compile(r"\w{4,}")
_STOPWORDS = frozenset(
    "what when where which this that these those does have been from "
    "with about would could should there their tell them then".split())


def _keywords(text: str) -> set[str]:
    """Lowercased 4+ char words minus function words — cheap relevance."""
    return {word for word in _KEYWORD_RE.findall(text.lower())
            if word not in _STOPWORDS}


def _flat(text, clip: int) -> str:
    """One line, clipped: prompt lists must never gain stray newlines."""
    return " ".join(str(text).split())[:clip]


def worth_remembering(outcome: dict) -> bool:
    """Deterministic gate: keep only exchanges a future task can build on.

    Empty answers, the "(no answer)" placeholder, and answers a judge
    rejected (``judged_good`` False) teach a future recall nothing —
    remembering them turns memory into a junk feed.
    """
    answer = str(outcome.get("answer") or "").strip()
    if not answer or answer == NO_ANSWER:
        return False
    return outcome.get("judged_good") is not False


class MemoryStore:
    """Per-agent history of (query, answer) with recency across a reload."""

    def __init__(self, agents: dict | None = None, ordinal: int = 0):
        self._by_agent: dict[str, list[dict]] = agents or {}
        self._ordinal = ordinal

    def remember(self, agent: str, query: str, answer: str) -> None:
        """Append one (query, answer) under an agent, stamped for recency.

        The store stays bounded: an older entry with the same query
        (case-folded) is replaced, and each agent keeps only its newest
        ``MAX_PER_AGENT`` exchanges.
        """
        entries = self._by_agent.setdefault(agent, [])
        key = query.strip().casefold()
        entries[:] = [entry for entry in entries
                      if entry["query"].strip().casefold() != key]
        entries.append({"n": self._ordinal, "query": query, "answer": answer})
        self._ordinal += 1
        del entries[:-MAX_PER_AGENT]

    def recent(self, limit: int = RECENT_LIMIT) -> list[dict]:
        """The most recent entries across all agents, newest first."""
        flat = [dict(entry, agent=agent)
                for agent, entries in self._by_agent.items()
                for entry in entries]
        flat.sort(key=lambda entry: entry["n"], reverse=True)
        return flat[:limit]

    def count(self) -> int:
        """Total remembered exchanges across every agent."""
        return sum(len(entries) for entries in self._by_agent.values())

    def relevant(self, task: str, limit: int = SHOW_LIMIT) -> list[dict]:
        """Entries sharing a keyword with the task, best first.

        Free deterministic pre-filter for the recall selector: the whole
        store is scored by keyword overlap against query plus answer
        text, ties break newest-first, and no overlap at all means no
        candidates — and therefore no model call. A task with no usable
        keywords at all ("who is he?") falls back to the newest entries:
        recency is the only signal an underspecified follow-up gives.
        """
        task_words = _keywords(task)
        if not task_words:
            return self.recent(limit)
        scored = []
        for entry in self.recent(self.count()):
            words = _keywords(entry["query"]) | _keywords(str(entry["answer"]))
            overlap = len(task_words & words)
            if overlap:
                scored.append((overlap, entry["n"], entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [entry for _, _, entry in scored[:limit]]

    def as_dict(self) -> dict:
        """The JSON-serialisable form: the ordinal plus the per-agent lists."""
        return {"ordinal": self._ordinal, "agents": self._by_agent}

    def dump(self, path: str) -> None:
        """Write the whole memory to JSON, creating parent folders."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), ensure_ascii=False,
                                     indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "MemoryStore":
        """Load memory from JSON; broken files or entries degrade gracefully.

        Every entry is coerced back to the store shape (string query and
        answer, int ordinal; hopeless entries dropped) so a hand-edited
        or legacy file can never crash remember/recall later.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            agents = {}
            for name, entries in (data.get("agents") or {}).items():
                if isinstance(entries, list):
                    cleaned = [e for e in map(_clean_entry, entries) if e]
                    if cleaned:
                        agents[str(name)] = cleaned
            ordinal = int(data.get("ordinal", 0))
        except (OSError, json.JSONDecodeError, TypeError,
                ValueError, AttributeError):
            return cls()
        return cls(agents=agents, ordinal=ordinal)


def _clean_entry(raw) -> dict | None:
    """Coerce one loaded JSON entry to the store shape; None if hopeless."""
    if not isinstance(raw, dict):
        return None
    try:
        entry = {"n": int(raw.get("n", 0)),
                 "query": str(raw.get("query", "")),
                 "answer": str(raw.get("answer", ""))}
    except (TypeError, ValueError):
        return None
    return entry if entry["query"].strip() else None


def select_memories(task: str, store: MemoryStore, policy,
                    max_picks: int = MAX_MEMORY_PICKS) -> list[dict]:
    """Let the model pick memories from a keyword-shortlisted set.

    ``store.relevant`` builds the candidate list for free; the model only
    ever sees entries that share a keyword with the task, so an empty
    shortlist (or an empty store) costs zero model calls. Returns the
    chosen entries (each a {n, query, answer, agent} dict), or an empty
    list when nothing relevant is picked.
    """
    candidates = store.relevant(task)
    if not candidates:
        return []
    node = _selector_node(task, candidates, max_picks)
    decision = policy.decide(_selector_episode(task, candidates), node)
    picked = decision.value if decision.valid else []
    return [candidates[index - 1] for index in picked
            if 1 <= index <= len(candidates)]


def render_recall(recalled: list[dict]) -> str | None:
    """Context block agents log into their episode, or None when empty.

    Verbatim clipped (query -> answer) lines — the recall counterpart of
    the TARGET SO FAR line: folded once into the episode history, never
    mutated afterwards.
    """
    if not recalled:
        return None
    lines = [f"- {_flat(entry['query'], QUERY_CLIP)} -> "
             f"{_flat(entry['answer'], RECALL_ANSWER_CLIP)}"
             for entry in recalled]
    return "EARLIER ANSWERS (past sessions):\n" + "\n".join(lines)


def _selector_episode(task: str, candidates: list[dict]) -> Episode:
    """A fresh episode showing numbered queries with clipped answers."""
    episode = Episode(SELECTOR_PREFIX, task)
    numbered = "\n".join(
        f"[{i}] {_flat(entry['query'], QUERY_CLIP)} -> "
        f"{_flat(entry['answer'], SELECTOR_ANSWER_CLIP)}"
        for i, entry in enumerate(candidates, 1))
    episode.open_observation(f"EARLIER QUESTIONS:\n{numbered}")
    return episode


def _selector_node(task: str, candidates: list[dict],
                   max_picks: int) -> PickManyNode:
    """The PickManyNode asking which earlier queries help, or 0 for none."""
    node = PickManyNode(
        f"Which earlier questions help with: {task}? "
        f"Reply up to {max_picks} numbers, or 0 if none fit.",
        len(candidates), max_picks=max_picks,
        items=[_flat(entry["query"], QUERY_CLIP) for entry in candidates])
    node.tag = "recall"
    return node


if __name__ == "__main__":
    import tempfile

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    store = MemoryStore()
    store.remember("web", "population of France", "about 68 million")
    store.remember("code", "reverse a string", "def rev(s): return s[::-1]")
    assert store.count() == 2
    assert store.recent()[0]["query"] == "reverse a string"  # newest first

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "mem.json")
        store.dump(path)
        again = MemoryStore.load(path)
        assert again.count() == 2 and again.recent()[0]["query"] == "reverse a string"
        again.remember("web", "capital of Japan", "Tokyo")  # ordinal continues
        assert again.recent()[0]["query"] == "capital of Japan"
    assert MemoryStore.load("/nonexistent/mem.json").count() == 0

    class _Pick:
        def __init__(self, text):
            self.text = text

        def complete(self, model, raw_prompt, opts):
            return GenResult(self.text, 5, 2, 0.0, "stop")

    policy = Policy(_Pick(" 1"), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    chosen = select_memories("help me reverse text", store, policy)
    assert len(chosen) == 1 and chosen[0]["query"] == "reverse a string", chosen
    assert select_memories("zebra migration", store, policy) == []  # no overlap
    assert select_memories("x", MemoryStore(), policy) == []  # empty store, no call

    assert worth_remembering({"answer": "Paris"})
    assert not worth_remembering({"answer": ""})
    assert not worth_remembering({"answer": NO_ANSWER})
    assert not worth_remembering({"answer": "junk", "judged_good": False})
    store.remember("web", "population of France", "68.2 million now")
    assert store.count() == 2  # same query replaced, not duplicated
    block = render_recall([{"query": "population of France",
                            "answer": "about 68 million"}])
    assert block.startswith("EARLIER ANSWERS") and "-> about 68" in block
    assert render_recall([]) is None
    print("smoke OK")
