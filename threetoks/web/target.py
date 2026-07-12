"""Target belief: what the research target appears to be, across rounds.

The deep-research loop's blind spot (seen in live traces): a rejected
round proposed its next query knowing only the task and the tried-query
strings, so the 1.5b paraphrased the task forever ("X project details",
"X project overview", ...) and never used what opened pages had revealed
about the target. This module gives the loop a tiny belief state:

- ``TargetBelief``: one taxonomy label picked from a 1-token menu plus a
  verbatim note fragment as anchor. The belief settles once the model
  has picked the same label twice in total; a settled belief costs no
  further calls.
- ``strategy_queries``: deterministic query composition from the belief
  (site: filters, profile/social angles, the anchor keyword), so the next
  query becomes a 1-digit menu pick over harness-built strings instead of
  free text the model has to write.
- ``too_similar``: content-word overlap check that rejects paraphrase
  requeries free of charge.
"""
import re
from dataclasses import dataclass, replace

from threetoks.nodes import ESCAPE, MenuNode, PickManyNode
from threetoks.policy import Policy
from threetoks.render import Episode

BELIEF_PREFIX = "You classify what a research subject is, from evidence notes."
ANCHOR_PREFIX = "You point at the note that best identifies a research subject."
TAXONOMY = (
    "a github user or open-source developer",
    "a software project, library, or product",
    "a company or organization",
    "a person or public figure",
    "a place, event, or general topic",
)
CONFIRMATIONS_TO_SETTLE = 2
BELIEF_NOTES_SHOWN = 8
ANCHOR_CLIP = 80
ANCHOR_TERM_MIN_CHARS = 4
SIMILARITY_LIMIT = 0.6
MAX_STRATEGIES = 5
SUBJECT_MAX_TERMS = 2
SUBJECT_RARE_MIN_CHARS = 5
# Written queries get deliberate diversity: identical temp-0 completions
# from a near-identical prompt are what produce carbon-copy requeries.
QUERY_TEMPERATURE = 0.7

STOPWORDS = frozenset(
    "the and for with what who where when how why which does did done "
    "has have had was were are is been being not don dont doesn isn "
    "about into from this that these those there their they them then "
    "search find look tell now some more most very much many also "
    "site com www org net http https".split())

_TOKEN_RE = re.compile(r"[A-Za-z][\w'-]*")
_CONTENT_WORD_RE = re.compile(r"[a-z0-9]{3,}")

_LABEL_TEMPLATES = {
    TAXONOMY[0]: ("site:github.com {s}", "{s} github repositories"),
    TAXONOMY[1]: ("site:github.com {s}", "{s} documentation"),
    TAXONOMY[2]: ("{s} company about", "{s} founder"),
    TAXONOMY[3]: ("{s} profile OR interview", "{s} twitter OR reddit"),
    TAXONOMY[4]: ("{s} wikipedia", "{s} explained"),
}
_GENERIC_TEMPLATES = ("{s} github", "{s} twitter OR reddit", "{s} wikipedia")


@dataclass(frozen=True)
class TargetBelief:
    """What the target appears to be; settles after repeat confirmation."""

    label: str | None = None
    anchor: str | None = None
    confirmations: int = 0
    settled: bool = False

    def line(self) -> str | None:
        """One prompt line describing the target, or None when unknown."""
        if not self.label:
            return None
        detail = f" — best note: {self.anchor}" if self.anchor else ""
        return f"TARGET SO FAR: {self.label}{detail}"


def normalize_query(query: str) -> str:
    """Canonical form for duplicate detection: case, quotes, spacing."""
    return " ".join(query.strip().strip('"\'').lower().split())


def _content_words(text: str) -> set[str]:
    """Lowercased informative words: 3+ chars, stopwords removed."""
    return {word for word in _CONTENT_WORD_RE.findall(text.lower())
            if word not in STOPWORDS}


def too_similar(query: str, tried: list[str]) -> bool:
    """True when most of the query's content words sit in one tried query.

    Coverage is measured against the NEW query (how much of it is
    already-tried words), per tried query — a paraphrase like "X project
    overview" after "X project details" is 2/3 covered and rejected,
    while a query that adds real new terms passes.
    """
    words = _content_words(query)
    if not words:
        return True
    return any(
        len(words & _content_words(old)) / len(words) >= SIMILARITY_LIMIT
        for old in tried)


def _dedupe_casefold(words: list[str]) -> list[str]:
    """Order-preserving case-insensitive dedup."""
    seen: set[str] = set()
    kept: list[str] = []
    for word in words:
        if word.lower() not in seen:
            seen.add(word.lower())
            kept.append(word)
    return kept


def subject_terms(task: str) -> list[str]:
    """Salient subject words of a task, deterministically.

    Mid-sentence capitalized tokens (proper nouns like "Fosowl") win;
    otherwise capitalized-first rare words — a leading proper noun as in
    "Musk builds…" must survive even though position 0 is ambiguous.
    Used to compose strategy queries, so precision matters more than
    recall — an empty list is fine.
    """
    words = _TOKEN_RE.findall(task)
    capitalized = [word for position, word in enumerate(words)
                   if position > 0 and word[0].isupper()
                   and word.lower() not in STOPWORDS]
    if capitalized:
        return _dedupe_casefold(capitalized)[:SUBJECT_MAX_TERMS]
    rare = [word for word in words
            if (len(word) >= SUBJECT_RARE_MIN_CHARS or word[0].isupper())
            and word.lower() not in STOPWORDS]
    rare.sort(key=lambda word: (not word[0].isupper(), -len(word)))
    return _dedupe_casefold(rare)[:SUBJECT_MAX_TERMS]


def _anchor_term(anchor: str | None, subject: str) -> str | None:
    """Anchor word that adds information beyond the subject words.

    Capitalized candidates (a name like "Armadillo") beat position;
    otherwise the first informative word wins.
    """
    if not anchor:
        return None
    subject_words = {word.lower() for word in _TOKEN_RE.findall(subject)}
    candidates = [word for word in _TOKEN_RE.findall(anchor)
                  if len(word) >= ANCHOR_TERM_MIN_CHARS
                  and word.lower() not in STOPWORDS
                  and word.lower() not in subject_words]
    for word in candidates:
        if word[0].isupper():
            return word
    return candidates[0] if candidates else None


def _adds_new_angle(query: str, subject_words: set[str],
                    prior: list[str]) -> bool:
    """True when the candidate's non-subject words offer an untried angle.

    Subject words are free — every candidate contains them — so only the
    words a template *adds* count toward the overlap check. Without this
    exemption a two-word subject leaves every one-word template ≥2/3
    covered by the task itself and the whole menu silently vanishes.
    """
    added = _content_words(query) - subject_words
    if not added:
        return False
    return all(
        len(added & _content_words(old)) / len(added) < SIMILARITY_LIMIT
        for old in prior)


def strategy_queries(task: str, belief: TargetBelief,
                     tried: list[str]) -> list[str]:
    """Harness-composed candidate queries for the requery menu.

    Composition beats free text here: templates add operators the tiny
    model never writes (site:, OR), the anchor keyword ties the query to
    found evidence, and tried or same-angle strings are struck out — so
    the requery loop can never degenerate into task rewordings. Generic
    angles are offered only while the target is unlabelled; a labelled
    belief gets its own templates, and an exhausted list simply means
    the caller falls back to a written query.
    """
    subject = " ".join(subject_terms(task)) or task
    subject_words = {word.lower() for word in _TOKEN_RE.findall(subject)}
    candidates: list[str] = []
    anchor_term = _anchor_term(belief.anchor, subject)
    if anchor_term:
        candidates.append(f"{subject} {anchor_term}")
    templates = _LABEL_TEMPLATES[belief.label] if belief.label \
        else _GENERIC_TEMPLATES
    candidates.extend(template.format(s=subject) for template in templates)
    tried_keys = {normalize_query(query) for query in tried}
    fresh: list[str] = []
    for query in candidates:
        if normalize_query(query) not in tried_keys \
                and _adds_new_angle(query, subject_words, tried + fresh):
            fresh.append(query)
    return fresh[:MAX_STRATEGIES]


def update_belief(task: str, belief: TargetBelief, notes,
                  policy: Policy) -> TargetBelief:
    """One 1-token menu refreshes the belief; settled beliefs are free.

    The initial pick counts as the first confirmation; one matching
    re-pick reaches ``CONFIRMATIONS_TO_SETTLE`` and stops all future
    update calls. Escape or an invalid answer changes nothing. A
    labelled belief without an anchor also gets one pick over the top
    notes for the note that best names the subject (verbatim, clipped).
    """
    if belief.settled or not notes.count():
        return belief
    episode = Episode(BELIEF_PREFIX, task)
    episode.open_observation(
        "NOTES:\n" + notes.render(max_notes=BELIEF_NOTES_SHOWN))
    node = MenuNode("Based on the notes, the subject of this task "
                    "appears to be:", list(TAXONOMY))
    node.tag = "belief"
    decision = policy.decide(episode, node)
    if not decision.valid or decision.value == ESCAPE:
        return belief
    if decision.value == belief.label:
        confirmations = belief.confirmations + 1
        return replace(
            belief, confirmations=confirmations,
            anchor=belief.anchor or _pick_anchor(task, notes, policy),
            settled=confirmations >= CONFIRMATIONS_TO_SETTLE)
    fresh = TargetBelief(label=decision.value, confirmations=1)
    return replace(fresh, anchor=_pick_anchor(task, notes, policy))


def _pick_anchor(task: str, notes, policy: Policy) -> str | None:
    """Pick the ONE note that best names the subject; verbatim, clipped."""
    texts = notes.texts()[:BELIEF_NOTES_SHOWN]
    episode = Episode(ANCHOR_PREFIX, task)
    episode.open_observation(
        "NOTES:\n" + notes.render(max_notes=BELIEF_NOTES_SHOWN))
    node = PickManyNode(
        "Which ONE note best names or identifies the subject?",
        len(texts), max_picks=1, items=texts)
    node.tag = "anchor"
    decision = policy.decide(episode, node)
    if decision.valid and decision.value:
        return texts[decision.value[0] - 1][:ANCHOR_CLIP]
    return None


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import PolicyConfig
    from threetoks.web.notes import NoteStore

    assert subject_terms("don't code, search what Fosowl has build") \
        == ["Fosowl"]
    assert too_similar("Fosowl project overview", ["Fosowl project details"])
    assert not too_similar("site:github.com Fosowl", ["what Fosowl built"])

    belief = TargetBelief(label=TAXONOMY[0], anchor="Fosowl /agenticSeek repo")
    queries = strategy_queries("who is Fosowl?", belief, ["who is Fosowl?"])
    assert queries and any("github" in q for q in queries), queries
    assert queries[0] == "Fosowl agenticSeek", queries
    assert "TARGET SO FAR" in belief.line()
    carmack = TargetBelief(label=TAXONOMY[3],
                           anchor="John Carmack founded Armadillo Aerospace")
    two_word = strategy_queries("who is John Carmack", carmack,
                                ["who is John Carmack"])
    assert two_word[0] == "John Carmack Armadillo", two_word
    assert len(two_word) >= 3, two_word

    class _ScriptedBackend:
        def __init__(self, texts):
            self.texts = list(texts)

        def complete(self, model, raw_prompt, opts):
            return GenResult(self.texts.pop(0), 5, 2, 0.0, "stop")

    store = NoteStore()
    store.add("Fosowl wrote agenticSeek, a local AI agent.", "http://g", 1)
    policy = Policy(_ScriptedBackend([" 2", " 1"]),
                    PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    updated = update_belief("who is Fosowl?", TargetBelief(), store, policy)
    assert updated.label in TAXONOMY and updated.confirmations == 1
    assert updated.anchor and "agenticSeek" in updated.anchor
    settled = TargetBelief(label=TAXONOMY[0], settled=True)
    assert update_belief("q", settled, store,
                         Policy(_ScriptedBackend([]),
                                PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
                         ) is settled
    print("smoke OK")
