"""Web-research vertical: the decision tree for answering with the web.

State machine: RESULTS -> PAGE -> (notes) -> ... -> ANSWER. The harness
does everything deterministic (searching, fetching, chunking, note
storage, loop guards); the model only answers menus, picks sentence
numbers, writes search queries, and phrases the final answer.

Dependencies are injected and duck-typed so the vertical is testable
without the IO layer:
- provider: .search(query, max_results) -> [obj with .title/.url/.snippet]
- fetch_page: callable(url) -> obj with .title/.sentences (list[str])
- notes: .add(text, source_url, sentence_idx), .count(), .render()
"""
import re
from dataclasses import dataclass

from threetoks.nodes import ESCAPE, MenuNode, PickManyNode, ShortTextNode
from threetoks.render import Episode
from threetoks.web.notes import rank_by_overlap
from threetoks.web.target import QUERY_TEMPERATURE, normalize_query

MAX_RESULTS_SHOWN = 5
MAX_SEARCHES = 3
MAX_PAGES = 4
CHUNK_SENTENCES = 25
SNIPPET_CHARS = 160
TITLE_CHARS = 60
ANSWER_MAX_TOKENS = 80
QUERY_MAX_TOKENS = 16
FORCE_ANSWER_AT_STEPS_LEFT = 3
MIN_KEPT_SENTENCES = 20
MAX_KEPT_SENTENCES = 100
INITIAL_STEPS_LEFT = 40

# Harvest wide: note passes pick up to NOTE_MAX_PICKS sentences, and the
# harness auto-walks AUTO_CHUNKS_PER_PAGE chunks per page (queuing a note
# pass on each) before it hands control back to the page menu.
NOTE_MAX_PICKS = 8
AUTO_CHUNKS_PER_PAGE = 3
QUOTE_FALLBACK_NOTES = 2

_ERROR_PAGE_RE = re.compile(
    r"error while loading|please reload|perform that action|"
    r"enable javascript|checking your browser|verify you are human|"
    r"rate limit", re.IGNORECASE)

OPT_NEXT_CHUNK = "read more of this page"
OPT_LINKS = "follow a link on this page"
LINKS_SHOWN = 6

# Menu-format bleed: an "answer" that is a bare menu digit or a comma
# list of indices ("1" / "1,2,3") violates the words-contract; a real
# numeric answer like "1989" is fine.
_MENU_BLEED_RE = re.compile(
    r"^\d{1,3}$|^\d{5,}$|^\d+(\s*,\s*\d+)+\s*,?$")
OPT_BACK = "back to the search results"
OPT_NEW_SEARCH = "search with different words"
OPT_ANSWER = "answer the task now"
FINAL_PREFILL = "FINAL ANSWER:"

PREFIX = """You control a research agent by answering menus.
Rules: reply with exactly ONE digit for ACTIONS menus; reply with numbers
separated by commas when asked which sentences to note.
Example:
ACTIONS:
1 = open: Weather in Oslo
2 = search with different words
Reply with exactly ONE digit.
ANSWER: 1
Example:
Which sentences contain information needed to answer? Reply with their
numbers separated by commas (e.g. 3,7).
RELEVANT: 2,5"""


def _looks_like_error_page(sentences: list[str]) -> bool:
    """Empty pages or mostly error/challenge text = a failed fetch."""
    if not sentences:
        return True
    hits = sum(bool(_ERROR_PAGE_RE.search(s)) for s in sentences)
    return hits > 0 and hits * 2 >= len(sentences)


@dataclass(frozen=True)
class _LinkTarget:
    """Adapter so a followed link opens like a search result."""
    title: str
    url: str


class WebResearchVertical:
    """Drives one research episode over injected search/fetch/notes IO."""

    def __init__(self, task: str, provider, fetch_page, notes,
                 initial_query: str | None = None,
                 context_lines: tuple[str, ...] = (),
                 seen_urls: set[str] | None = None,
                 tried_queries: set[str] | None = None):
        self.task = task
        self.provider = provider
        self.fetch_page = fetch_page
        self.notes = notes
        self.episode = Episode(PREFIX, task)
        self.steps_left = INITIAL_STEPS_LEFT
        self.searches_done = 0
        self.pages_opened = 0
        # Shared MUTABLE sets when the caller passes them (deep_research
        # does, across rounds): a page or query exhausted in one round
        # must stay exhausted in every later round.
        self.seen_urls: set[str] = seen_urls if seen_urls is not None \
            else set()
        self._tried_queries: set[str] = tried_queries \
            if tried_queries is not None else set()
        self.results = []
        self.page = None
        self.chunk_start = 0
        self.pending = None
        self.answer = None
        self.answer_retried = False
        self._first_bleed = ""
        self._awaiting_curation = False
        self._auto_chunks_shown = 0
        self._option_map: dict[str, object] = {}
        for line in context_lines:  # folded into history at first observation
            self.episode.log_session(line)
        self._run_search(initial_query or task)

    # ------------------------------------------------------------ engine API

    def next_node(self):
        """Next decision node, or None when the episode is finished."""
        if self.answer is not None:
            return None
        if self.steps_left <= FORCE_ANSWER_AT_STEPS_LEFT:
            if not self._pending_is_answer():
                self.pending = self._answer_flow()
            return self.pending
        if self.pending is not None:
            return self.pending
        return self._page_menu() if self.page else self._results_menu()

    def _pending_is_answer(self) -> bool:
        """True when the queued node already belongs to the answer flow."""
        final_text = isinstance(self.pending, ShortTextNode) \
            and self.pending.prefill == FINAL_PREFILL
        pending_pick = self._awaiting_curation \
            and isinstance(self.pending, PickManyNode)
        return final_text or pending_pick

    def apply(self, node, decision):
        """Execute one decision; invalid decisions map to the escape path."""
        value = decision.value if decision.valid else ESCAPE
        if isinstance(node, PickManyNode) and self._awaiting_curation:
            self._apply_curation(value if decision.valid else [])
        elif isinstance(node, PickManyNode):
            self._apply_notes(value if decision.valid else [])
        elif isinstance(node, ShortTextNode) and node.prefill == FINAL_PREFILL:
            self._accept_answer(value if decision.valid else "")
        elif isinstance(node, ShortTextNode):
            self._run_search(value if decision.valid else self.task)
        else:
            self._apply_menu(value)

    def result(self) -> dict:
        """Episode outcome with note provenance."""
        return {"task": self.task, "answer": self.answer,
                "notes": self.notes.render(), "searches": self.searches_done,
                "pages": self.pages_opened}

    # ------------------------------------------------------------ menus

    def _results_menu(self) -> MenuNode:
        options = list(self._build_open_options())
        if self.searches_done < MAX_SEARCHES:
            options.append(OPT_NEW_SEARCH)
        if self.notes.count():
            options.append(OPT_ANSWER)
        return MenuNode("What next?", options or [OPT_ANSWER])

    def _build_open_options(self):
        """Openable results only: never re-offer seen or failed URLs."""
        self._option_map = {}
        if self.pages_opened >= MAX_PAGES:
            return
        openable = [r for r in self.results[:MAX_RESULTS_SHOWN]
                    if r.url not in self.seen_urls]
        for index, result in enumerate(openable, 1):
            text = f"open result {index}: {result.title[:TITLE_CHARS]}"
            self._option_map[text] = result
            yield text

    def _page_menu(self) -> MenuNode:
        options = []
        if self.chunk_start + CHUNK_SENTENCES < len(self.page.sentences):
            options.append(OPT_NEXT_CHUNK)
        if getattr(self.page, "links", None) and self.pages_opened < MAX_PAGES:
            options.append(OPT_LINKS)
        options.append(OPT_BACK)
        if self.searches_done < MAX_SEARCHES:
            options.append(OPT_NEW_SEARCH)
        if self.notes.count():
            options.append(OPT_ANSWER)
        return MenuNode("What next?", options)

    def _answer_flow(self):
        """Curate the notes (once), then synthesise the answer from them.

        Harvest→curate→synthesise: when the store overflows CURATE_KEEP a
        PICK_MANY first ranks the notes down to the best; that ranked store
        then grounds one short GENERATION step. Curation is skipped for
        small stores and for note stores that cannot ``select`` (test
        fakes), and the no-notes case goes straight to synthesis.
        """
        if self._needs_curation():
            return self._curate_node()
        return self._synthesis_node()

    def _needs_curation(self) -> bool:
        """True when the note store is big enough and supports select()."""
        from threetoks.research import CURATE_KEEP  # deferred: breaks a cycle
        return (self.notes.count() > CURATE_KEEP
                and hasattr(self.notes, "select"))

    def _curate_node(self) -> PickManyNode:
        """PICK_MANY that ranks notes to the best CURATE_KEEP, most first."""
        from threetoks.research import (  # deferred: breaks a cycle
            build_curate_node, curation_pool)
        self.notes = curation_pool(self.notes)
        self.episode.open_observation(
            f"NOTES COLLECTED:\n{self.notes.render()}",
            "> curating the strongest evidence")
        self.page = None
        self._awaiting_curation = True
        return build_curate_node(self.task, self.notes.texts())

    def _apply_curation(self, picked: list[int]) -> None:
        """Replace the store with the ranked selection, then synthesise."""
        from threetoks.research import CURATE_KEEP  # deferred: breaks a cycle
        self._awaiting_curation = False
        ranking = picked or list(range(1, CURATE_KEEP + 1))
        self.notes = self.notes.select(ranking)
        self.pending = self._synthesis_node()

    def _synthesis_node(self) -> ShortTextNode:
        """Generate the answer from the (curated) notes, or "(no answer)".

        The notes are now high-quality, so a short generation grounded in
        them beats quoting a fragment; the bleed guard and the top-2
        verbatim fallback in ``_accept_answer`` catch a garbled generation.
        Notes are shown WITHOUT their "(source: ...)" tags here — a live
        run showed the 1.5b copying the tags into the answer text.
        """
        body = "\n".join(f"[{i}] {t}"
                         for i, t in enumerate(self.notes.texts(), 1)) \
            or "(none)"
        self.episode.open_observation(f"NOTES COLLECTED:\n{body}",
                                      "> moving to final answer")
        self.page = None
        return ShortTextNode(
            f"Using ONLY the notes above, write one or two short sentences "
            f"that answer: {self.task}. Reply with words.",
            FINAL_PREFILL, max_tokens=ANSWER_MAX_TOKENS)

    def _links_menu(self):
        """Menu over relevant page links; picks open a new page."""
        self._option_map = {}
        for link in self._ranked_links()[:LINKS_SHOWN]:
            text = f"go: {link.label[:TITLE_CHARS]}"
            self._option_map[text] = _LinkTarget(link.label, link.url)
        if not self._option_map:
            return self._page_menu()
        return MenuNode("Which link?", list(self._option_map))

    def _accept_answer(self, text: str) -> None:
        """Accept a synthesised answer, retrying once on menu-format bleed.

        On a second bleed the generation is abandoned: if notes exist the
        answer falls back to quoting the top curated notes verbatim (still
        useful evidence), otherwise "(no answer)".
        """
        bleed = bool(text) and bool(_MENU_BLEED_RE.fullmatch(text.strip()))
        if bleed and self.answer_retried:
            repeated_digit = (text.strip() == self._first_bleed
                              and len(text.strip()) == 1)
            if repeated_digit:  # a persistent single digit is a real answer
                self.answer = text.strip()
            else:
                self.answer = self._quote_fallback()
            return
        if bleed:
            self.answer_retried = True
            self._first_bleed = text.strip()
            self.pending = ShortTextNode(
                f"That was a number, not an answer. In plain words, "
                f"state the answer to: {self.task}",
                FINAL_PREFILL, max_tokens=ANSWER_MAX_TOKENS)
            return
        self.answer = text or "(no answer)"

    def _quote_fallback(self) -> str:
        """Quote the notes closest to the task when synthesis keeps bleeding.

        Ranked by task-keyword overlap, not store order: the fallback must
        stay aimed at the goal even though the generation step gave up.
        """
        top = rank_by_overlap(self.notes.texts(), self.task,
                              QUOTE_FALLBACK_NOTES)
        return "\n".join(top) if top else "(no answer)"

    # ------------------------------------------------------------ transitions

    def _apply_menu(self, value: str) -> None:
        if value == OPT_ANSWER:
            self.pending = self._answer_flow()
        elif value == OPT_NEW_SEARCH or (value == ESCAPE and not self.page):
            if self.searches_done >= MAX_SEARCHES:
                self.pending = self._answer_flow()
            else:
                node = ShortTextNode(
                    "Suggest a better web search query for the task.",
                    "QUERY:", max_tokens=QUERY_MAX_TOKENS)
                node.temperature = QUERY_TEMPERATURE
                self.pending = node
        elif value == OPT_LINKS:
            self.pending = self._links_menu()
        elif value == OPT_NEXT_CHUNK:
            self.chunk_start += CHUNK_SENTENCES
            self._show_page_chunk("> read next section")
            self._queue_note_pass()
        elif value == OPT_BACK or value == ESCAPE:
            self._return_to_results(f"> left page '{self._page_title()}'")
        else:
            self._open_result(value)

    def _apply_notes(self, picked: list[int]) -> None:
        chunk = self._chunk()
        for idx in picked:
            self.notes.add(chunk[idx - 1], self.page.url,
                           self.chunk_start + idx)
        self.episode.log_session(f"> noted {len(picked)} sentences")
        self.pending = None
        self._auto_walk()

    def _auto_walk(self) -> None:
        """Advance to the next chunk automatically, up to the walk budget.

        Rich pages hid dozens of notes behind a "read more" the model
        rarely chose, so the harness now walks AUTO_CHUNKS_PER_PAGE chunks
        deterministically — each with its own note pass — before the page
        menu reappears for manual continuation.
        """
        more_sentences = (self.chunk_start + CHUNK_SENTENCES
                          < len(self.page.sentences))
        if more_sentences and self._auto_chunks_shown < AUTO_CHUNKS_PER_PAGE:
            self.chunk_start += CHUNK_SENTENCES
            self._auto_chunks_shown += 1
            self._show_page_chunk("> auto-read next section")
            self._queue_note_pass()

    def _open_result(self, value: str) -> None:
        match = self._option_map.get(value)
        if match is None:
            self._return_to_results("> could not open that result")
            return
        self._load_page(match)

    def _load_page(self, result) -> None:
        try:
            page = self.fetch_page(result.url)
        except Exception as error:  # fetch failures become a log line
            self.episode.log_session(f"> failed to open '{result.title[:40]}'")
            self.seen_urls.add(result.url)
            self.pending = None
            return
        page.url = page.url or result.url  # fakes may omit it
        if _looks_like_error_page(page.sentences):
            self.episode.log_session(
                f"> page '{result.title[:40]}' failed to load properly")
            self.seen_urls.add(result.url)
            self.pending = None
            return
        page.sentences = self._relevant_sentences(page.sentences)
        self.page = page
        self.chunk_start = self._best_chunk_start(page.sentences)
        self.pages_opened += 1
        self._auto_chunks_shown = 1  # the entry chunk is the first auto chunk
        self.seen_urls.add(result.url)
        self._show_page_chunk(f"> opened '{result.title[:TITLE_CHARS]}'")
        self._queue_note_pass()

    def _queue_note_pass(self) -> None:
        """Every freshly shown chunk gets an immediate note-picking pass.

        The page was opened deliberately, so asking "take notes?" first is
        a wasted (and fallible) decision — E3/E1 data. Harvest wide:
        NOTE_MAX_PICKS raises the per-pass cap so rich chunks yield many
        notes, not the old four.
        """
        chunk = self._chunk()
        if chunk:
            self.pending = PickManyNode(
                "Which sentences contain information needed to answer?",
                len(chunk), max_picks=NOTE_MAX_PICKS, items=chunk)

    def _run_search(self, query: str) -> None:
        query = query.strip().strip('"\'').strip()  # models love exact-match quotes; they shrink results
        self.searches_done += 1
        key = normalize_query(query)
        if key in self._tried_queries:
            # Strict: a query never runs twice (within or across rounds).
            self.episode.log_session(
                f"> already searched '{query[:40]}' — use different words")
            self.pending = None
            return
        self._tried_queries.add(key)
        try:
            self.results = self.provider.search(query)
        except Exception:  # a dead provider must not kill the episode
            self.results = []
        self.page = None
        self.pending = None
        lines = [f"[{i}] {r.title[:TITLE_CHARS]} — {r.snippet[:SNIPPET_CHARS]}"
                 for i, r in enumerate(self.results[:MAX_RESULTS_SHOWN], 1)]
        body = "\n".join(lines) or "(no results)"
        self.episode.open_observation(
            f"SEARCH RESULTS for '{query}':\n{body}",
            f"> searched '{query}' ({len(self.results)} results)")

    def _return_to_results(self, closing: str) -> None:
        self.page = None
        self.pending = None
        self._render_results(closing)

    def _render_results(self, closing: str) -> None:
        lines = [f"[{i}] {r.title[:TITLE_CHARS]} — {r.snippet[:SNIPPET_CHARS]}"
                 for i, r in enumerate(self.results[:MAX_RESULTS_SHOWN], 1)]
        self.episode.open_observation(
            "SEARCH RESULTS:\n" + ("\n".join(lines) or "(no results)"), closing)

    def _show_page_chunk(self, closing: str) -> None:
        chunk = self._chunk()
        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(chunk, 1))
        section = self.chunk_start // CHUNK_SENTENCES + 1
        self.episode.open_observation(
            f"PAGE '{self._page_title()}' (section {section}):\n{numbered}",
            closing)
        self.pending = None

    def _relevant_sentences(self, sentences: list[str]) -> list[str]:
        """Keep task-relevant sentences (plus neighbors for context).

        Zero-keyword banner/citation junk never even gets displayed, so
        it can never be picked as a note. Short pages pass through.
        """
        keywords = set(re.findall(r"\w{4,}", self.task.lower()))
        if len(sentences) <= MIN_KEPT_SENTENCES or not keywords:
            return sentences[:MAX_KEPT_SENTENCES]
        keep: set[int] = set()
        for index, sentence in enumerate(sentences):
            lowered = sentence.lower()
            if any(word in lowered for word in keywords):
                keep.update((index - 1, index, index + 1))
        kept = [s for i, s in enumerate(sentences) if i in keep]
        if len(kept) < MIN_KEPT_SENTENCES:
            return sentences[:MAX_KEPT_SENTENCES]
        return kept[:MAX_KEPT_SENTENCES]

    def _best_chunk_start(self, sentences: list[str]) -> int:
        """Open the page at the chunk most relevant to the task.

        Long pages bury the payload (e.g. a pricing table at sentence
        100) — always showing chunk 1 wastes the model's view on
        preamble. Deterministic keyword overlap picks the entry chunk;
        "read more" still walks forward from there.
        """
        keywords = set(re.findall(r"\w{4,}", self.task.lower()))
        if not keywords or len(sentences) <= CHUNK_SENTENCES:
            return 0
        best_start, best_score = 0, -1
        for start in range(0, len(sentences), CHUNK_SENTENCES):
            window = " ".join(
                sentences[start:start + CHUNK_SENTENCES]).lower()
            score = sum(window.count(word) for word in keywords)
            if score > best_score:
                best_start, best_score = start, score
        return best_start

    def _ranked_links(self):
        """Unseen links with wordy labels, task-keyword matches first."""
        keywords = set(re.findall(r"\w{4,}", self.task.lower()))

        def wordy(link):
            return len(re.findall(r"[A-Za-z]{2,}", link.label)) >= 2 \
                and not link.label.startswith("http")

        candidates = [l for l in self.page.links
                      if l.url not in self.seen_urls and wordy(l)]
        return sorted(candidates, key=lambda l: -sum(
            w in l.label.lower() for w in keywords))

    def _chunk(self) -> list[str]:
        return self.page.sentences[self.chunk_start:
                                   self.chunk_start + CHUNK_SENTENCES]

    def _page_title(self) -> str:
        return (self.page.title if self.page else "")[:TITLE_CHARS]
