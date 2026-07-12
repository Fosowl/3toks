"""EditVertical: the state machine driving one edit episode.

Implements the engine's Vertical protocol so it runs unmodified under
run_episode + Policy. Phase order, every phase skipped for free whenever
the harness already knows the answer:

    LOCATE     free index lookup when the request names a unique symbol;
               keyword-ranked free-take when the ranking has a clear
               winner; one disambiguation menu otherwise; else NAVIGATE.
    NAVIGATE   folder -> file -> (def | class -> method) menus, paged,
               each level skipped when it has one option.
    OPERATION  inferred for free when the target provably calls an
               undefined helper (insert it); otherwise one menu:
               replace / insert-after / delete.
    DELETE     one menu over the target's statements (already-tried spans
               struck out); zero generation.
    GENERATE   a stateless micro-prompt for the delta only, gated by
               placeholder/oscillation/whole-file-parse checks inside the
               Policy retry ladder.
    CHECK      the caller's trusted test (must fail before, pass after);
               without a test, the edited module must still import.

Repair escalates instead of looping on one wrong choice (E8 scenarios
2/4/5): first failure regenerates the body, the next re-asks the
operation, the next re-locates to the runner-up candidate — all bounded
by MAX_REPAIRS_TOTAL and the episode step budget.

Two episodes, one boundary: MENU decisions read the growing session log
(located target, chosen operation, what failed) — the state-machine
memory that lets a re-asked menu choose differently. GENERATE decisions
get a fresh episode every time and never see failure text: repair is
blind resampling, because E5b measured error-feedback repair as WORSE
than blind (docs/DESIGN-coding-agent.md §2).
"""
import re
from dataclasses import dataclass
from pathlib import Path

from threetoks.code import gates
from threetoks.code.edit import nav, ops, oracle
from threetoks.code.edit.index import Symbol, SymbolIndex
from threetoks.nodes import MenuNode
from threetoks.render import Episode

MAX_REPAIRS_TOTAL = 5     # hard cap on failed checks across the episode
NAV_BACKTRACKS_MAX = 2    # escapes tolerated while navigating
FORCE_DONE_AT_STEPS_LEFT = 2
REPAIR_TEMPERATURE = 0.4
LABEL_CLIP = 72
_IDENT = re.compile(r"[a-z_][a-z0-9_]*")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

OP_REPLACE, OP_INSERT_AFTER, OP_DELETE = "replace", "insert_after", "delete"
OPERATION_LABELS = {
    OP_REPLACE: "replace the target function's body with corrected code",
    OP_INSERT_AFTER: "insert new helper code right after the target function",
    OP_DELETE: "delete the offending line(s) inside the target function",
}
SYSTEM_MENU = ("You are a menu selector controlling a code-editing agent. "
               "Reply with exactly one digit: the number of the best action.")


@dataclass
class EditOutcome:
    """What happened in one episode."""
    success: bool = False
    verified: bool = False        # a real caller test passed (not just gates)
    reason: str = ""
    path_taken: str = ""          # "direct" | "disambiguate" | "navigate"
    operation: str = ""
    target: str = ""              # qualname of the resolved symbol
    target_file: str = ""
    repairs: int = 0


class EditVertical:
    """Drives one edit episode against a corpus of .py files.

    ``test_code`` is the trusted oracle (must fail before the edit, pass
    after). When None, the vertical still edits but only gates on the
    edited module importing cleanly, and reports ``verified=False``.
    """

    def __init__(self, request: str, corpus_root: Path,
                 test_code: str | None = None):
        self.request = request
        self.root = Path(corpus_root)
        self.test_code = test_code
        self.index = SymbolIndex(self.root)
        # Menus see the growing session log (what was located, what
        # failed) — that is state-machine memory, and it is what lets a
        # re-asked operation menu choose differently. Generations must
        # NOT: repair is blind resample (E5b — error feedback measurably
        # underperforms), so next_node() swaps in a fresh stateless
        # episode for every generate step.
        self._menu_episode = Episode(SYSTEM_MENU, request)
        self.episode = self._menu_episode
        self.steps_left = 0
        self.phase = "start"
        self.outcome = EditOutcome()

        self._req_tokens = set(_IDENT.findall(request.lower()))
        self._locate_pool: list[Symbol] = []   # ranked runners-up
        self._locate_labels: list[str] = []
        self._locate_choice: nav.LevelChoice | None = None
        self._nav: nav.LevelChoice | None = None
        self._nav_values: dict[str, object] = {}   # menu label -> value
        self._nav_folder = ""
        self._nav_file = ""
        self._nav_class = ""
        self._excluded: dict[str, set] = {"folder": set(), "file": set(),
                                          "def": set()}
        self._backtracks = 0
        self.target: Symbol | None = None
        self._original_source = ""
        self._target_indent = ""
        self._seen_hashes: set[int] = set()
        self._tried_deletes: set[tuple[int, int]] = set()
        self._delete_choice: nav.LevelChoice | None = None
        self._delete_spans: list[tuple[int, int]] = []
        self._generate_node: ops.GenerateSpanNode | None = None
        self._repairs = 0       # total failed checks, capped for the episode
        self._stage = 0         # escalation stage for the CURRENT target
        self._skip_inference = False   # escalation forces the real menu

    # ------------------------------------------------------------ engine API

    def next_node(self):
        """Next decision node; loops through free phases without a node."""
        while True:
            if self.phase == "done":
                return None
            if self.phase == "start":
                self._run_precheck()
                continue
            if self.steps_left <= FORCE_DONE_AT_STEPS_LEFT:
                self._finish(False, "step budget exhausted")
                continue
            node = getattr(self, f"_node_{self.phase}")()
            if node is not None:
                self.episode = Episode(ops.GENERATE_SYSTEM, self.request) \
                    if self.phase == "generate" else self._menu_episode
                return node
            # a phase that resolved for free loops back with no node

    def apply(self, node, decision) -> None:
        """Route one decision to the current phase's handler."""
        getattr(self, f"_apply_{self.phase}")(node, decision)

    def result(self) -> dict:
        """The episode outcome as a plain dict."""
        out = self.outcome
        return {"success": out.success, "verified": out.verified,
                "reason": out.reason, "path_taken": out.path_taken,
                "operation": out.operation, "target": out.target,
                "target_file": out.target_file, "repairs": out.repairs}

    # ------------------------------------------------------------ precheck

    def _run_precheck(self) -> None:
        """Free gates before any model call: the bug must reproduce."""
        if self.test_code is not None:
            ok, _ = oracle.run_test(self.root, self.test_code)
            if ok:
                self._finish(False, "test passes before any edit — "
                             "nothing to fix")
                return
        self._locate()

    # ------------------------------------------------------------ locate

    def _locate(self) -> None:
        """Free lookup by name; ranked free-take; menu; else navigate."""
        hits = self._name_hits()
        if not hits:
            self.outcome.path_taken = "navigate"
            self._enter_folder_level()
            return
        ranked = self._rank(hits)
        if len(hits) == 1 or self._clear_winner(ranked):
            self.outcome.path_taken = "direct" if len(hits) == 1 \
                else "disambiguate"
            self._locate_pool = [s for _, s in ranked[1:]]
            self._set_target(ranked[0][1])
            self.phase = "operation"
            return
        self.outcome.path_taken = "disambiguate"
        self._locate_pool = [s for _, s in ranked]
        self._locate_labels = _uniquify(
            [self._symbol_label(s) for _, s in ranked])
        self._locate_choice = nav.LevelChoice(
            "Several definitions match. Which one is the request about?",
            list(self._locate_labels))
        self.phase = "locate_menu"

    def _name_hits(self) -> list[Symbol]:
        """Symbols whose bare name appears verbatim in the request."""
        tokens = set(_IDENT.findall(self.request.lower()))
        hits: list[Symbol] = []
        for symbol in self.index.symbols:
            if symbol.name.lower() in tokens and symbol not in hits:
                hits.append(symbol)
        return hits

    def _rank(self, hits: list[Symbol]) -> list[tuple[int, Symbol]]:
        """Candidates scored by request-keyword overlap, best first.

        The shared bare name is excluded from scoring (every candidate has
        it); path segments, qualname parts, and the docstring first line
        all count — the free pre-ranking E8's scenario 5 was missing.
        """
        scored = []
        for symbol in hits:
            own = _symbol_tokens(symbol) - {symbol.name.lower()}
            scored.append((_overlap(self._req_tokens, own), symbol))
        scored.sort(key=lambda pair: -pair[0])
        return scored

    @staticmethod
    def _clear_winner(ranked: list[tuple[int, Symbol]]) -> bool:
        """True when the top candidate strictly out-scores the runner-up."""
        return ranked[0][0] > 0 and ranked[0][0] > ranked[1][0]

    def _symbol_label(self, symbol: Symbol) -> str:
        """Menu label: qualname, file, and a docstring hint when present."""
        label = f"{symbol.qualname} — {symbol.file}"
        if symbol.doc:
            label += f" ({symbol.doc})"
        return _clip(label)

    def _set_target(self, symbol: Symbol) -> None:
        """Fix the edit target and seed the oscillation guard."""
        self.target = symbol
        self._original_source = self.index.source_of(symbol.file)
        def_line = self._original_source.splitlines()[symbol.lineno - 1]
        self._target_indent = def_line[:len(def_line) - len(def_line.lstrip())]
        span = ops.span_text(self._original_source, symbol.lineno,
                             symbol.end_lineno)
        self._seen_hashes = {hash(_dedent(span, self._target_indent))}
        self._tried_deletes = set()
        self._stage = 0        # each target gets its own escalation ladder
        self.outcome.target = symbol.qualname
        self.outcome.target_file = symbol.file
        self._menu_episode.log_session(f"> target: {symbol.qualname} in {symbol.file}")

    def _node_locate_menu(self):
        return self._menu_or_free(self._locate_choice, self._picked_candidate)

    def _apply_locate_menu(self, node, decision) -> None:
        picked = self._locate_choice.apply(decision)
        if picked is None:
            self._finish(False, "could not disambiguate the target symbol")
        elif picked != nav.MORE_LABEL:
            self._picked_candidate(picked)

    def _picked_candidate(self, label: str) -> None:
        position = self._locate_labels.index(label)
        self._locate_labels.pop(position)
        self._set_target(self._locate_pool.pop(position))
        self.phase = "operation"

    def _menu_or_free(self, choice: nav.LevelChoice, on_pick):
        """A LevelChoice's menu, or its single remaining item for free."""
        node = choice.node()
        if node is not None:
            return node
        on_pick(choice.free_take())
        return None

    # ------------------------------------------------------------ navigate

    def _enter_level(self, phase: str, question: str,
                     entries: list[tuple[object, str, set]], on_pick) -> None:
        """Rank a level's (value, label, tokens) entries by request-keyword
        overlap: a lone entry or a strictly best-scoring one is taken for
        FREE; otherwise a paged menu is shown, best candidates first."""
        if not entries:
            self._backtrack_from(phase)
            return
        scored = sorted(entries,
                        key=lambda e: -_overlap(self._req_tokens, e[2]))
        top = _overlap(self._req_tokens, scored[0][2])
        second = _overlap(self._req_tokens, scored[1][2]) \
            if len(scored) > 1 else -1
        if len(scored) == 1 or (top > 0 and top > second):
            on_pick(scored[0][0])
            return
        labels = _uniquify([_clip(label) for _, label, _ in scored])
        self._nav_values = {label: value
                            for label, (value, _, _) in zip(labels, scored)}
        self._nav = nav.LevelChoice(question, labels)
        self._nav_on_pick = on_pick
        self.phase = phase

    def _apply_nav(self, decision, fail_reason: str) -> None:
        """Shared decision handling: pick, page, or backtrack on escape."""
        picked = self._nav.apply(decision)
        if picked is None:
            self._backtrack_from(self.phase, fail_reason)
        elif picked != nav.MORE_LABEL:
            self._nav_on_pick(self._nav_values[picked])

    def _backtrack_from(self, phase: str, why: str = "nothing fits") -> None:
        """An escape mid-navigation goes UP one level (the escaped choice
        struck out), instead of abandoning the episode (E8 live runs died
        exactly here). Bounded by NAV_BACKTRACKS_MAX."""
        self._backtracks += 1
        if phase == "nav_folder" or self._backtracks > NAV_BACKTRACKS_MAX:
            self._finish(False, f"navigation abandoned: {why}")
        elif phase == "nav_file":
            self._excluded["folder"].add(self._nav_folder)
            self._enter_folder_level()
        else:
            self._excluded["file"].add(self._rel_path())
            self._enter_file_level()

    def _enter_folder_level(self) -> None:
        entries = []
        for folder in self._candidate_folders():
            files = [f"{folder}/{name}" if folder != "." else name
                     for name in nav.list_files(self._folder_path(folder))]
            tokens = set().union(*(self._file_tokens(rel) for rel in files),
                                 _words(folder))
            label = f"{folder} — {', '.join(Path(f).name for f in files)}"
            entries.append((folder, label, tokens))
        self._enter_level("nav_folder", "Which folder holds the code to fix?",
                          entries, self._picked_folder)

    def _candidate_folders(self) -> list[str]:
        folders = [f for f in nav.list_folders(self.root)
                   if f not in self._excluded["folder"]]
        if nav.list_files(self.root) and "." not in self._excluded["folder"]:
            folders.append(".")            # corpus files at the root itself
        return folders

    def _folder_path(self, folder: str) -> Path:
        return self.root if folder == "." else self.root / folder

    def _picked_folder(self, folder: str) -> None:
        self._nav_folder = folder
        self._enter_file_level()

    def _enter_file_level(self) -> None:
        folder = self._nav_folder
        entries = []
        for name in nav.list_files(self._folder_path(folder)):
            rel = f"{folder}/{name}" if folder != "." else name
            if rel in self._excluded["file"]:
                continue
            defs = [d for d, _ in self.index.top_level_names(rel)]
            label = f"{name} — {', '.join(defs)}" if defs else name
            entries.append((name, label, self._file_tokens(rel)))
        self._enter_level("nav_file",
                          f"Which file in '{folder}' holds the code to fix?",
                          entries, self._picked_file)

    def _file_tokens(self, rel: str) -> set:
        """Rankable words of one file: stem, symbols, and body text (body
        identifiers are free signal — a request often names a variable the
        docstring never mentions)."""
        return _words(Path(rel).stem) | _words(self.index.source_of(rel))

    def _picked_file(self, filename: str) -> None:
        self._nav_file = filename
        rel = self._rel_path()
        source = self.index.source_of(rel)
        entries = []
        for name, kind in self.index.top_level_names(rel):
            symbol = self.index.symbol_at(rel, name)
            hint = symbol.doc or (", ".join(self.index.methods_of(rel, name))
                                  if kind == "class" else "")
            label = f"{name} ({kind}) — {hint}" if hint else f"{name} ({kind})"
            entries.append(((name, kind), label,
                            _def_tokens(symbol) | self._body_words(source,
                                                                   symbol)))
        self._enter_level("nav_def",
                          f"Which definition in '{filename}' is it?",
                          entries, self._picked_def)

    @staticmethod
    def _body_words(source: str, symbol: Symbol) -> set:
        """The def's own body text as ranking signal."""
        return _words(ops.span_text(source, symbol.lineno, symbol.end_lineno))

    def _picked_def(self, value: tuple[str, str]) -> None:
        name, kind = value
        rel = self._rel_path()
        if kind == "class":
            self._nav_class = name
            source = self.index.source_of(rel)
            entries = []
            for method in self.index.methods_of(rel, name):
                symbol = self.index.symbol_at(rel, f"{name}.{method}")
                label = f"{method} — {symbol.doc}" if symbol.doc else method
                entries.append((method, label,
                                _def_tokens(symbol)
                                | self._body_words(source, symbol)))
            self._enter_level("nav_method",
                              f"Which method of '{name}' is it?",
                              entries, self._picked_method)
            return
        self._set_target(self.index.symbol_at(rel, name))
        self.phase = "operation"

    def _picked_method(self, method: str) -> None:
        qual = f"{self._nav_class}.{method}"
        self._set_target(self.index.symbol_at(self._rel_path(), qual))
        self.phase = "operation"

    def _node_nav_folder(self):
        return self._nav.node()

    def _apply_nav_folder(self, node, decision) -> None:
        self._apply_nav(decision, "no folder chosen")

    def _node_nav_file(self):
        return self._nav.node()

    def _apply_nav_file(self, node, decision) -> None:
        self._apply_nav(decision, "no file chosen")

    def _node_nav_def(self):
        return self._nav.node()

    def _apply_nav_def(self, node, decision) -> None:
        self._apply_nav(decision, "no definition chosen")

    def _node_nav_method(self):
        return self._nav.node()

    def _apply_nav_method(self, node, decision) -> None:
        self._apply_nav(decision, "no method chosen")

    def _rel_path(self) -> str:
        return f"{self._nav_folder}/{self._nav_file}" \
            if self._nav_folder and self._nav_folder != "." else self._nav_file

    # ------------------------------------------------------------ operation

    def _node_operation(self):
        """Infer the operation for free when possible, else one menu.

        The re-ask escalation stage sets ``_skip_inference`` — otherwise a
        deterministic inference would just re-pick the operation that
        already failed, and the "widen the scope" stage would be a no-op.
        """
        if not self._skip_inference:
            inferred = self._infer_operation()
            if inferred is not None:
                self._begin_operation(inferred)
                return None
        self._skip_inference = False
        options = [OPERATION_LABELS[op]
                   for op in (OP_REPLACE, OP_INSERT_AFTER, OP_DELETE)]
        return MenuNode("How should this be fixed?", options, escape=True)

    def _infer_operation(self) -> str | None:
        """insert-after, for free, when the target calls an undefined name.

        The evidence is unambiguous (E8 scenario 2 failed exactly here by
        asking a menu instead): a callee that resolves nowhere must be
        inserted before any other fix can matter.
        """
        helper = self._missing_helper()
        if helper is None:
            return None
        self._menu_episode.log_session(
            f"> inferred: insert missing helper {helper[0]}()")
        return OP_INSERT_AFTER

    def _missing_helper(self) -> tuple[str, str] | None:
        """The (name, args) of an undefined callee in the target, if any.

        "Known" covers everything the file binds at top level — imports,
        constants, defs — so a call to an imported name is never mistaken
        for a missing helper (that inference would shadow the import).
        """
        source = self.index.source_of(self.target.file)
        span = _dedent(ops.span_text(source, self.target.lineno,
                                     self.target.end_lineno),
                       self._target_indent)
        known = ops.module_level_names(source) \
            | {s.name for s in self.index.symbols
               if s.file == self.target.file}
        return ops.find_missing_helper(span, self.target.name, known)

    def _apply_operation(self, node, decision) -> None:
        if not decision.valid or decision.value not in OPERATION_LABELS.values():
            self._finish(False, "no operation chosen")
            return
        chosen = next(op for op, label in OPERATION_LABELS.items()
                      if label == decision.value)
        self._begin_operation(chosen)

    def _begin_operation(self, operation: str) -> None:
        self.outcome.operation = operation
        self._menu_episode.log_session(f"> op: {operation}")
        if operation == OP_DELETE:
            self._start_delete()
        else:
            self._start_generate(operation)

    # ------------------------------------------------------------ delete

    def _start_delete(self) -> None:
        """Offer the target's statements, minus spans already tried and
        minus any whose deletion would break the file (checked free).
        A statement whose text strictly out-matches every sibling on
        request keywords is deleted for free, no menu."""
        source = self.index.source_of(self.target.file)
        spans = [s for s in ops.statement_spans(source, self.target.qualname)
                 if s not in self._tried_deletes
                 and gates.parses(ops.delete_span(source, *s))]
        if not spans:
            self._escalate("every deletable line was already tried")
            return
        scores = [_overlap(self._req_tokens,
                           _words(ops.span_text(source, start, end)))
                  for start, end in spans]
        ranked = sorted(zip(scores, spans), key=lambda pair: -pair[0])
        if len(ranked) == 1 or (ranked[0][0] > 0
                                and ranked[0][0] > ranked[1][0]):
            self._tried_deletes.add(ranked[0][1])
            self._check_candidate(ops.delete_span(source, *ranked[0][1]))
            return
        self._delete_spans = [span for _, span in ranked]
        self._delete_choice = nav.LevelChoice(
            f"Which line inside {self.target.name}() is wrong?",
            _uniquify([_clip(ops.span_text(source, s, e))
                       for _, (s, e) in ranked]))
        self.phase = "delete_menu"

    def _node_delete_menu(self):
        return self._menu_or_free(self._delete_choice, self._picked_delete)

    def _apply_delete_menu(self, node, decision) -> None:
        picked = self._delete_choice.apply(decision)
        if picked is None:
            self._finish(False, "no line chosen to delete")
        elif picked != nav.MORE_LABEL:
            self._picked_delete(picked)

    def _picked_delete(self, label: str) -> None:
        span = self._delete_spans[self._delete_choice.items.index(label)]
        self._tried_deletes.add(span)
        source = self.index.source_of(self.target.file)
        self._check_candidate(ops.delete_span(source, *span))

    # ------------------------------------------------------------ generate

    def _start_generate(self, operation: str) -> None:
        """Build the delta micro-prompt for replace or insert-after."""
        source = self.index.source_of(self.target.file)
        self._original_source = source
        build = self._replace_node if operation == OP_REPLACE \
            else self._insert_node
        node = build(source)
        if node is None:
            return
        node.temperature = REPAIR_TEMPERATURE if self._repairs else None
        self._generate_node = node
        self.phase = "generate"

    def _replace_node(self, source: str) -> ops.GenerateSpanNode | None:
        target = self.target
        args = ops.def_args(source, target.qualname)
        splice = ops.replace_splice_fn(source, target.lineno,
                                       target.end_lineno, target.name, args,
                                       self._seen_hashes, self._target_indent)
        span = ops.span_text(source, target.lineno, target.end_lineno)
        before, after = ops.context_lines(source, target.lineno,
                                          target.end_lineno)
        instruction = _replace_instruction(
            _dedent(span, self._target_indent), before, after)
        prefill = f"def {target.name}({args}):\n    "
        return ops.GenerateSpanNode(instruction, prefill, splice)

    def _insert_node(self, source: str) -> ops.GenerateSpanNode | None:
        helper = self._missing_helper()
        if helper is None:
            self._escalate("insert-after: no undefined helper to add")
            return None
        name, args = helper
        end = self._insert_line(source)
        splice = ops.insert_after_splice_fn(source, end, name, args,
                                            self._seen_hashes)
        span = ops.span_text(source, self.target.lineno, self.target.end_lineno)
        instruction = (
            f"Existing function (verbatim, do not repeat it):\n{span}\n"
            f"Write the missing helper function {name}({args}) that this "
            "code calls but that is never defined. Do not use pass or "
            "raise NotImplementedError — write a real body.")
        return ops.GenerateSpanNode(instruction, f"def {name}({args}):\n    ",
                                    splice)

    def _insert_line(self, source: str) -> int:
        """Helpers land at top level: after the def, or the whole class."""
        if self.target.kind != "method":
            return self.target.end_lineno
        class_name = self.target.qualname.split(".", 1)[0]
        owner = self.index.symbol_at(self.target.file, class_name)
        return owner.end_lineno if owner else self.target.end_lineno

    def _node_generate(self):
        return self._generate_node

    def _apply_generate(self, node, decision) -> None:
        if not decision.valid:
            self._escalate("no candidate survived the generation gates")
            return
        candidate, new_source = decision.value
        self._seen_hashes.add(hash(new_source))
        self._check_candidate(candidate)

    # ------------------------------------------------------------ check + repair

    def _check_candidate(self, candidate: str) -> None:
        """Write, test, and either finish or escalate (file restored)."""
        path = self.root / self.target.file
        path.write_text(candidate)
        ok, err = self._run_oracle()
        if ok:
            verified = self.test_code is not None
            self.outcome.verified = verified
            self._finish(True, "test passes after edit" if verified
                         else "edit applied; module imports (no test given)")
            return
        path.write_text(self._original_source)
        self._escalate(f"{self.outcome.operation} edit failed: {err[:80]}")

    def _run_oracle(self) -> tuple[bool, str]:
        """The caller's test, or an import check when none was given."""
        if self.test_code is not None:
            return oracle.run_test(self.root, self.test_code)
        module = self.target.file.removesuffix(".py").replace("/", ".")
        module = module.removesuffix(".__init__")
        return oracle.run_test(self.root, f"import {module}\n")

    def _escalate(self, why: str) -> None:
        """Widen the repair scope instead of looping on one wrong choice.

        Per target: regenerate once, then re-ask the operation, then move
        to the next locate candidate (which resets the ladder for the new
        target). MAX_REPAIRS_TOTAL bounds the whole episode.
        """
        self._menu_episode.log_session(f"> {why}")
        self._repairs += 1
        self.outcome.repairs = self._repairs
        if self._repairs > MAX_REPAIRS_TOTAL \
                or self.steps_left <= FORCE_DONE_AT_STEPS_LEFT:
            self._finish(False, why)
            return
        stage, self._stage = self._stage, self._stage + 1
        if stage == 0 and self.outcome.operation:
            self._retry_same_operation()
        elif stage <= 1:
            self._skip_inference = True
            self.phase = "operation"
        elif self._locate_pool:
            self._relocate()
        else:
            self._finish(False, why)

    def _retry_same_operation(self) -> None:
        """Stage 1: same operation again (delete re-menus, others resample)."""
        if self.outcome.operation == OP_DELETE:
            self._start_delete()
        else:
            self._start_generate(self.outcome.operation)

    def _relocate(self) -> None:
        """Stage 3: give up on this target; take the ranked runner-up."""
        runner_up = self._locate_pool.pop(0)
        self._menu_episode.log_session(f"> retargeting: {runner_up.qualname}")
        self._set_target(runner_up)
        self.phase = "operation"

    def _finish(self, success: bool, reason: str) -> None:
        self.outcome.success = success
        self.outcome.reason = reason
        self.phase = "done"


def _replace_instruction(span: str, before: str, after: str) -> str:
    """The replace micro-prompt: verbatim span + context (the request is
    already the fresh generate episode's TASK line)."""
    lines = ["Current function (verbatim):", span]
    if before:
        lines += ["Lines just before it:", before]
    if after:
        lines += ["Lines just after it:", after]
    lines.append("Write this function with a corrected, complete body. Do "
                 "not use pass or raise NotImplementedError — write a real "
                 "fix.")
    return "\n".join(lines)


# Words too generic to carry ranking signal in Python source.
_STOPWORDS = frozenset(
    "return self def class import from if elif else for while in not and or "
    "none true false pass raise with as try except finally print len int str "
    "float list dict set range the a an it its is are was to of on this that "
    "py".split())
_PREFIX_MATCH_MIN = 5


def _words(text: str) -> set[str]:
    """Lowercased signal tokens of any text (identifier parts split too)."""
    return {token for token in _TOKEN_SPLIT.split(text.lower())
            if token and token not in _STOPWORDS}


def _overlap(request_tokens: set[str], tokens: set[str]) -> int:
    """Keyword overlap, counting long-prefix pairs too.

    Exact matches count; so does a pair like "uppercase"/"upper" where one
    token is a >=5-char prefix of the other — request prose rarely uses
    the exact identifier morphology (E8 live: "UPPERCASE" vs .upper()).
    """
    score = 0
    for token in tokens:
        if token in request_tokens:
            score += 1
        elif len(token) >= _PREFIX_MATCH_MIN and any(
                len(req) >= _PREFIX_MATCH_MIN
                and (req.startswith(token) or token.startswith(req))
                for req in request_tokens):
            score += 1
    return score


def _symbol_tokens(symbol: Symbol) -> set[str]:
    """Rankable words of one candidate: path parts, qualname parts, doc."""
    return _words(f"{symbol.file} {symbol.qualname} {symbol.doc}") - {"py"}


def _def_tokens(symbol: Symbol) -> set[str]:
    """Rankable words of one def inside an already-chosen file."""
    return _words(f"{symbol.qualname} {symbol.doc}")


def _dedent(text: str, indent: str) -> str:
    """Strip one known indent level from every line (method spans)."""
    if not indent:
        return text
    return "\n".join(line[len(indent):] if line.startswith(indent) else line
                     for line in text.splitlines())


def _clip(text: str, limit: int = LABEL_CLIP) -> str:
    """One-line, length-capped menu label text."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _uniquify(labels: list[str]) -> list[str]:
    """Deduplicate menu labels that clipping made identical.

    Labels double as lookup keys back to the picked candidate, so two
    entries sharing a long common prefix must not collapse into one.
    """
    counts: dict[str, int] = {}
    out = []
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
        out.append(label if counts[label] == 1
                   else f"{label} ({counts[label]})")
    return out


if __name__ == "__main__":
    import tempfile

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.engine import run_episode
    from threetoks.policy import Policy, PolicyConfig

    class _Scripted:
        """Menu picks by keyword grep; one canned completion otherwise."""

        def __init__(self, keywords, completion):
            self.keywords = list(keywords)
            self.completion = completion

        def complete(self, model, raw_prompt, opts):
            if "ACTIONS:" in raw_prompt:
                for keyword in self.keywords:
                    for line in raw_prompt.splitlines():
                        if re.match(r"^\d+ = ", line) and keyword in line:
                            digit = line.split(" = ", 1)[0]
                            return GenResult(f" {digit}", 10, 1, 0.0, "stop")
                return GenResult(" 1", 10, 1, 0.0, "stop")
            return GenResult(self.completion, 10, 20, 0.0, "stop")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pkg").mkdir()
        (root / "pkg" / "mod.py").write_text(
            "def add(a, b):\n    return a - b\n")
        test = "from pkg.mod import add\nassert add(2, 2) == 4\n"
        vertical = EditVertical("fix the bug in add", root, test)
        backend = _Scripted(["replace the target"], "return a + b")
        policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = run_episode(vertical, policy, max_steps=20)
        assert outcome["success"] and outcome["verified"], outcome
        assert outcome["path_taken"] == "direct"
        assert (root / "pkg" / "mod.py").read_text().count("a + b") == 1
    print("smoke OK")
