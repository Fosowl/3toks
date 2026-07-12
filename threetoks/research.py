"""Deep research: episodes repeat until a judge accepts the answer.

Loop: run a research episode -> unanimous 3-lens judge panel -> if
rejected and budget remains, refresh the target belief (what the target
appears to be, judged from the notes) and pick the next search angle —
a menu of harness-composed strategy queries first, free text as an
option and fallback, both checked against every query already tried.
The next round adopts the episode's note store as pruned by curation,
so evidence accumulates across rounds without junk piling up forever.
"""
import re

from threetoks.engine import run_episode
from threetoks.memory import render_recall
from threetoks.nodes import MenuNode, PickManyNode, ShortTextNode
from threetoks.policy import Policy
from threetoks.render import Episode
from threetoks.services import Services
from threetoks.web.notes import NoteStore
from threetoks.web.target import (QUERY_TEMPERATURE, TargetBelief,
                                 normalize_query, strategy_queries,
                                 too_similar, update_belief)
from threetoks.web.vertical import WebResearchVertical

JUDGE_PREFIX = ("You strictly judge research answers. Vague answers, "
                "meta-text about pages, or answers that dodge the "
                "question are bad.")
QUERY_PREFIX = "You write web search queries."
STRATEGY_PREFIX = "You choose the next web search to run."
OPT_WRITE_QUERY = "write a different query yourself"
SEARCH_OPT_PREFIX = "search: "
NEW_QUERY_MAX_TOKENS = 16
# Rotating direction nudges: the written-requery prompt differs every
# round, so a deterministic sampler cannot reproduce the same query.
# Enough angles that no hint repeats within any realistic round budget.
ANGLE_HINTS = (
    "name a specific person, organisation, or place",
    "aim at official or primary sources",
    "aim at recent news coverage",
    "aim at numbers, statistics, or reports",
    "aim at an encyclopedia or reference entry",
    "aim at a forum, community, or Q&A discussion",
    "aim at an interview, talk, or profile piece",
    "aim at technical documentation or a project page",
    "add a year or date to narrow the time range",
    "ask about the history or origin of the subject",
    "ask about criticism, problems, or controversy",
    "use completely different words than before",
)
MAX_STEPS_PER_ROUND = 25
JUDGE_TASK_CLIP = 48

# Curation: keep at most this many notes; skip the model call entirely at
# or below it (small stores are already tight enough to answer from).
CURATE_KEEP = 8
CURATE_POOL = 30
CURATE_PREFIX = "You keep only the best, non-duplicate evidence notes."

# Three lenses; ALL must approve (unanimous panel — one skeptic vetoes).
JUDGE_LENSES = (
    ("You strictly judge whether answers are direct and specific.",
     "good — it directly answers: {task}",
     "bad — vague, off-topic, or does not answer it"),
    ("You detect junk: garbled fragments, menus, titles, UI text.",
     "clean — readable sentences with real information",
     "junk — fragments, glued words, titles, dates, or UI text"),
    ("You judge usefulness for the person who asked.",
     "useful — the asker gets what they wanted: {task}",
     "useless — the asker would still have to search themselves"),
)
_INDEX_LIST_RE = re.compile(r"\d+(\s*,\s*\d+)+\s*,?")


def _obviously_bad(result: dict) -> bool:
    """Deterministic pre-checks: never ask the model about clear junk."""
    answer = (result.get("answer") or "").strip()
    if not answer or answer == "(no answer)":
        return True
    if _INDEX_LIST_RE.fullmatch(answer):
        return True  # "1,2,3" is menu bleed; a bare number may be a real answer
    return not (result.get("notes") or "").strip()


def _one_judge(task: str, result: dict, policy: Policy,
               lens: tuple[str, str, str]) -> bool:
    """One judging lens; True only on an explicit positive verdict."""
    system, good_template, bad = lens
    good = good_template.format(task=task[:JUDGE_TASK_CLIP])
    episode = Episode(system, task)
    episode.open_observation(
        f"PROPOSED ANSWER:\n{result['answer']}\n\n"
        f"SUPPORTING NOTES:\n{result['notes']}")
    node = MenuNode(f"Judge the answer to: {task}", [good, bad],
                    escape=False)
    node.tag = "judge"
    decision = policy.decide(episode, node)
    return decision.valid and decision.value == good


def judge_answer(task: str, result: dict, policy: Policy) -> bool:
    """Unanimous three-lens panel; any skeptic (or junk pre-check) vetoes.

    Code rejects obvious junk before any model call; then three
    differently-framed judges (directness, junk-detection, usefulness)
    must ALL approve — ~9 output tokens total.
    """
    if _obviously_bad(result):
        return False
    return all(_one_judge(task, result, policy, lens)
               for lens in JUDGE_LENSES)


def propose_query(task: str, tried: list[str], policy: Policy,
                  belief: TargetBelief | None = None) -> str:
    """Pick the next search angle, different from every prior try.

    With a belief, harness-composed strategy queries (site: filters,
    anchor keywords) go on a 1-digit menu, with writing a query kept as
    an ordinary option. Without a belief — or when the menu declines —
    the model writes one, which must survive the paraphrase check or a
    deterministic fallback replaces it.

    ``belief=None`` opts out of the strategy menu entirely (external
    callers, tests); a blank ``TargetBelief()`` still gets generic
    angles on the menu — pass one to participate in belief mechanics.
    """
    if belief is not None:
        picked = _pick_strategy(task, tried, belief, policy)
        if picked:
            return picked
    return _written_query(task, tried, belief, policy)


def _query_history(belief: TargetBelief | None, tried: list[str]) -> str:
    """Observation for requery nodes: belief line plus tried queries."""
    previous = "\n".join(f"- {q}" for q in tried)
    body = ("QUERIES ALREADY TRIED (all failed to find the answer):\n"
            f"{previous}")
    hint = belief.line() if belief else None
    return f"{hint}\n\n{body}" if hint else body


def _pick_strategy(task: str, tried: list[str], belief: TargetBelief,
                   policy: Policy) -> str | None:
    """Menu over composed strategy queries; None means write one instead."""
    strategies = strategy_queries(task, belief, tried)
    if not strategies:
        return None
    episode = Episode(STRATEGY_PREFIX, task)
    episode.open_observation(_query_history(belief, tried))
    options = [f"{SEARCH_OPT_PREFIX}{query}" for query in strategies]
    node = MenuNode("Which search should run next?",
                    options + [OPT_WRITE_QUERY],
                    escape=False)  # the write option IS the escape hatch
    node.tag = "requery"
    decision = policy.decide(episode, node)
    if decision.valid and decision.value in options:
        return decision.value[len(SEARCH_OPT_PREFIX):]
    return None


def _written_query(task: str, tried: list[str],
                   belief: TargetBelief | None, policy: Policy) -> str:
    """Free-text query at temperature; tried paraphrases fall back.

    Two diversity levers keep repeat proposals from ever recurring:
    the node runs at ``QUERY_TEMPERATURE`` instead of the temp-0
    ladder, and a rotating ``ANGLE_HINTS`` nudge makes the prompt
    itself differ from every earlier requery.
    """
    episode = Episode(QUERY_PREFIX, task)
    episode.open_observation(_query_history(belief, tried))
    hint = ANGLE_HINTS[len(tried) % len(ANGLE_HINTS)]
    node = ShortTextNode(
        f"Write ONE new web search query taking a different angle "
        f"— {hint}.",
        "QUERY:", max_tokens=NEW_QUERY_MAX_TOKENS)
    node.tag = "requery"
    node.temperature = QUERY_TEMPERATURE
    decision = policy.decide(episode, node)
    proposed = decision.value if decision.valid else ""
    tried_keys = {normalize_query(query) for query in tried}
    if proposed and normalize_query(proposed) not in tried_keys \
            and not too_similar(proposed, tried):
        return proposed
    return _fallback_query(task, tried, belief)


def _fallback_query(task: str, tried: list[str],
                    belief: TargetBelief | None) -> str:
    """Deterministic last resort: an unstruck strategy, then task+angle.

    The angle words are exactly the paraphrases this feature kills, so
    they only run when even the composed candidates are exhausted.
    """
    strategies = strategy_queries(task, belief or TargetBelief(), tried)
    if strategies:
        return strategies[0]
    tried_keys = {normalize_query(query) for query in tried}
    angles = ("details", "overview", "guide", "review")
    for offset in range(len(angles)):
        candidate = f"{task} {angles[(len(tried) + offset) % len(angles)]}"
        if normalize_query(candidate) not in tried_keys:
            return candidate
    return f"{task} {len(tried) + 1}"


def build_curate_node(task: str, note_texts: list[str]) -> PickManyNode:
    """One PICK_MANY that ranks the best notes; pick order is the ranking.

    Items are the numbered note texts; the model picks the most useful
    first, capped at ``CURATE_KEEP``. NoteStore.select preserves the pick
    order, so the picks double as a relevance ranking.
    """
    pool = note_texts[:CURATE_POOL]
    question = (f"Which notes are the most useful, non-duplicate evidence "
                f"for: {task}? Pick the best, most important first.")
    node = PickManyNode(question, min(len(note_texts), CURATE_POOL),
                        max_picks=CURATE_KEEP, items=pool)
    node.tag = "curate"
    return node


def curation_pool(notes: NoteStore) -> NoteStore:
    """Trim an overflowing store to the window curation can actually see.

    The curate menu shows at most ``CURATE_POOL`` notes; before this
    window existed the overflow silently kept the FIRST notes, so
    evidence found in later rounds never reached curation. The window
    keeps the head (the previous round's ranked keepers; in round 1
    simply the earliest notes) plus the newest notes. The dropped
    middle is gone for good — still strictly better than hiding
    everything after the cap.
    """
    count = notes.count()
    if count <= CURATE_POOL:
        return notes
    tail_length = CURATE_POOL - CURATE_KEEP
    head = list(range(1, CURATE_KEEP + 1))
    tail = list(range(count - tail_length + 1, count + 1))
    return notes.select(head + tail)


def curate_notes(task: str, notes: NoteStore, policy: Policy) -> NoteStore:
    """Rank-and-trim a note store down to the best ``CURATE_KEEP`` notes.

    Small stores pass through untouched. Otherwise one PICK_MANY ranks the
    pool; the model's pick order becomes the new store order. An invalid
    or empty pick falls back to keeping the first ``CURATE_KEEP`` notes.
    """
    if notes.count() <= CURATE_KEEP:
        return notes
    notes = curation_pool(notes)
    node = build_curate_node(task, notes.texts())
    episode = Episode(CURATE_PREFIX, task)
    episode.open_observation("NOTES:\n" + notes.render(max_notes=CURATE_POOL))
    decision = policy.decide(episode, node)
    if decision.valid and decision.value:
        return notes.select(decision.value)
    return notes.select(list(range(1, CURATE_KEEP + 1)))


def _target_summary(belief: TargetBelief) -> dict | None:
    """Plain-data belief for result consumers; None while unlabelled."""
    if not belief.label:
        return None
    return {"label": belief.label, "anchor": belief.anchor}


def deep_research(task: str, services: Services, policy: Policy) -> dict:
    """Run judged research rounds; evidence and target belief accumulate.

    Each rejected round refreshes the target belief from the notes and
    feeds it to the requery step; the next round's episode opens with
    the belief line, so page/answer decisions know what the target
    turned out to be. The final round skips both (nothing would use
    them). Memories recalled for this task (``services.recalled``) are
    logged into every round's episode as an EARLIER ANSWERS block.
    """
    notes = NoteStore()
    belief = TargetBelief()
    recall_line = render_recall(services.recalled)
    tried: list[str] = []
    seen_urls: set[str] = set()      # shared: no page reopens across rounds
    executed: set[str] = set()       # shared: no search reruns across rounds
    query = task
    result: dict = {"answer": None}
    total_rounds = max(1, services.max_research_rounds)
    for round_number in range(1, total_rounds + 1):
        tried.append(query)
        context = tuple(line for line in (recall_line, belief.line()) if line)
        vertical = WebResearchVertical(task, services.provider,
                                       services.fetch_page, notes,
                                       initial_query=query,
                                       context_lines=context,
                                       seen_urls=seen_urls,
                                       tried_queries=executed)
        result = run_episode(vertical, policy, max_steps=MAX_STEPS_PER_ROUND)
        notes = vertical.notes  # adopt curation pruning across rounds
        result["rounds"] = round_number
        result["queries"] = list(tried)
        if judge_answer(task, result, policy):
            result["judged_good"] = True
            result["target"] = _target_summary(belief)
            return result
        if round_number < total_rounds:
            belief = update_belief(task, belief, notes, policy)
            query = propose_query(task, tried, policy, belief)
    result["judged_good"] = False
    result["target"] = _target_summary(belief)
    return result


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import PolicyConfig

    class _YesJudge:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" 1", 5, 2, 0.0, "stop")

    policy = Policy(_YesJudge(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    verdict = judge_answer("q?", {"answer": "42", "notes": "[1] 42"}, policy)
    assert isinstance(verdict, bool)
    new_query = propose_query("q?", ["q?"], policy)
    assert isinstance(new_query, str) and new_query
    belief = TargetBelief(label="a github user or open-source developer")
    strategy = propose_query("who is Fosowl?", ["who is Fosowl?"],
                             Policy(_YesJudge(),
                                    PolicyConfig(ModelSpec("m",
                                                           FAMILY_CHATML))),
                             belief)
    assert isinstance(strategy, str) and strategy

    class _PickBackend:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" 2,1", 5, 2, 0.0, "stop")

    small = NoteStore()
    small.add("only note", "http://x", 1)
    assert curate_notes("q?", small, policy).count() == 1  # <= CURATE_KEEP
    big = NoteStore()
    for i in range(12):
        big.add(f"note number {i}", "http://x", i)
    pick_policy = Policy(_PickBackend(),
                         PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    ranked = curate_notes("q?", big, pick_policy)
    assert ranked.count() == 2 and ranked.entries()[0].text == "note number 1"
    overflow = NoteStore()
    for i in range(40):
        overflow.add(f"pool note {i}", "http://x", i)
    window = curation_pool(overflow)
    assert window.count() == CURATE_POOL
    assert window.entries()[CURATE_KEEP].text == "pool note 18"
    print("smoke OK")
