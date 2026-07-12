"""EditVertical: the state machine that drives one bug-fix episode.

Implements threetoks/engine.py's Vertical protocol (episode / next_node /
apply / result / steps_left) so it runs unmodified under
threetoks.engine.run_episode and threetoks.policy.Policy -- the real
harness pieces, not a re-implementation.

Phase order (a phase is skipped for free whenever the harness already
knows the answer -- no menu is ever shown for a 0- or 1-option decision):

    LOCATE     free lookup by name (0 model calls) if the request names a
               symbol unique in the corpus; one disambiguation menu if the
               name is ambiguous; else NAVIGATE.
    NAVIGATE   folder -> file -> (def | class -> method) menus, each
               skipped when there is only one option at that level.
    OPERATION  one menu: replace-span / insert-after-span / delete-span.
    DELETE     (delete only) one menu over the target function's own
               statements -- no model generation at all.
    GENERATE   (replace / insert-after only) a stateless micro-prompt for
               the replacement/insertion body; the splice is gated by a
               whole-file ast.parse and, on failure, resampled blind by
               Policy's own retry ladder (temps 0.0/0.0/0.4).
    ORACLE     run the scenario's trusted test; on failure, bounded repair
               (regenerate) up to MAX_REPAIR_ROUNDS, else report failure.

Every model decision this vertical ever asks is a MenuNode or a
GenerateSpanNode from the real threetoks package -- nothing here
re-implements retries, permutation, or prefill steering.
"""
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_DIR.parent.parent
for _p in (str(SPIKE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import edit_ops  # noqa: E402
import nav  # noqa: E402
import oracle  # noqa: E402
from symbol_index import SymbolIndex, Symbol  # noqa: E402

from threetoks.nodes import ESCAPE, MenuNode  # noqa: E402
from threetoks.render import Episode  # noqa: E402

MAX_REPAIR_ROUNDS = 2
FORCE_DONE_AT_STEPS_LEFT = 2
REPAIR_TEMPERATURE = 0.4
_IDENT = re.compile(r"[a-z_][a-z0-9_]*")

# edit_ops only defines the splice primitives; the operation vocabulary
# (and its menu labels) is the vertical's own concern.
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
    """What happened in one episode -- the report's raw material."""
    success: bool = False
    reason: str = ""
    path_taken: str = ""          # "direct" | "disambiguate" | "navigate"
    operation: str = ""
    target: str = ""               # qualname of the resolved symbol
    target_file: str = ""
    repair_rounds: int = 0


class EditVertical:
    """Drives one edit episode against a materialized fixture corpus."""

    def __init__(self, request: str, corpus_root: Path, test_code: str):
        self.request = request
        self.root = Path(corpus_root)
        self.test_code = test_code
        self.index = SymbolIndex(self.root)

        self.episode = Episode(SYSTEM_MENU, request)
        self.steps_left = 0
        self.phase = "start"
        self.outcome = EditOutcome()

        self._candidates: list[Symbol] = []
        self._nav: nav.LevelChoice | None = None
        self._nav_folder: str | None = None
        self._nav_file: str | None = None
        self._nav_class: str | None = None
        self.target: Symbol | None = None
        self._original_source: str | None = None
        self._delete_choice: nav.LevelChoice | None = None
        self._delete_spans: list[tuple[int, int]] = []
        self._repair_round = 0
        self._done_result: tuple[bool, str] | None = None

    # ------------------------------------------------------------ engine API

    def next_node(self):
        while True:
            if self.phase == "start":
                self._run_precheck()
                continue
            if self.phase == "done":
                return None
            if self.steps_left <= FORCE_DONE_AT_STEPS_LEFT and not self._done_result:
                self._finish(False, "step budget exhausted")
                continue
            node = self._dispatch()
            if node is not None:
                return node
            # phases that resolve for free loop back with no node emitted

    def apply(self, node, decision) -> None:
        handler = getattr(self, f"_apply_{self.phase}", None)
        if handler is None:
            raise RuntimeError(f"no handler for phase {self.phase!r}")
        handler(node, decision)

    def result(self) -> dict:
        return {
            "success": self.outcome.success,
            "reason": self.outcome.reason,
            "path_taken": self.outcome.path_taken,
            "operation": self.outcome.operation,
            "target": self.outcome.target,
            "target_file": self.outcome.target_file,
            "repair_rounds": self.outcome.repair_rounds,
        }

    # ------------------------------------------------------------ dispatch

    def _dispatch(self):
        return getattr(self, f"_node_{self.phase}")()

    # ------------------------------------------------------------ 0. precheck

    def _run_precheck(self) -> None:
        """Deterministic, zero-token: the bug must actually reproduce."""
        ok, _ = oracle.run_test(self.root, self.test_code)
        if ok:
            self._finish(False, "fixture's test passed before any edit "
                        "(scenario setup is broken)")
            return
        self._locate()

    def _locate(self) -> None:
        """Free lookup by name; else disambiguate; else navigate."""
        tokens = set(_IDENT.findall(self.request.lower()))
        names = {s.name for s in self.index.symbols}
        matched_names = tokens & names
        hits: list[Symbol] = []
        for name in matched_names:
            for symbol in self.index.find_by_name(name):
                if symbol not in hits:
                    hits.append(symbol)
        if len(hits) == 1:
            self.outcome.path_taken = "direct"
            self._set_target(hits[0])
            self.phase = "operation"
        elif len(hits) > 1:
            self.outcome.path_taken = "disambiguate"
            self._candidates = hits
            self.phase = "locate_menu"
        else:
            self.outcome.path_taken = "navigate"
            self._nav = nav.LevelChoice(
                "Which folder holds the code to fix?",
                nav.list_folders(self.root))
            self.phase = "nav_folder"

    def _set_target(self, symbol: Symbol) -> None:
        self.target = symbol
        self._original_source = self.index.source_of(symbol.file)
        self.outcome.target = symbol.qualname
        self.outcome.target_file = symbol.file

    # ------------------------------------------------------------ 1. locate menu

    def _node_locate_menu(self):
        labels = [f"{s.qualname} — {s.file}" for s in self._candidates]
        return MenuNode("Several functions share this name. Which one is "
                        "the request about?", labels, escape=True)

    def _apply_locate_menu(self, node, decision) -> None:
        if not decision.valid or decision.value == ESCAPE:
            self._finish(False, "could not disambiguate the target symbol")
            return
        index = node.options.index(decision.value)
        self._set_target(self._candidates[index])
        self.phase = "operation"

    # ------------------------------------------------------------ 2. navigate

    def _node_nav_folder(self):
        return self._level_node(self._nav)

    def _apply_nav_folder(self, node, decision) -> None:
        picked = self._nav.apply(decision)
        if picked is None:
            self._finish(False, "no folder chosen")
        elif picked != nav.MORE_LABEL:
            self._nav_folder = picked
            files = nav.list_files(self.root / picked)
            self._nav = nav.LevelChoice(
                f"Which file in '{picked}' holds the code to fix?", files)
            self.phase = "nav_file"

    def _node_nav_file(self):
        return self._level_node(self._nav)

    def _apply_nav_file(self, node, decision) -> None:
        picked = self._nav.apply(decision)
        if picked is None:
            self._finish(False, "no file chosen")
        elif picked != nav.MORE_LABEL:
            self._nav_file = picked
            rel = f"{self._nav_folder}/{picked}"
            defs = self.index.top_level_names(rel)
            self._nav = nav.LevelChoice(
                f"Which definition in '{picked}' holds the code to fix?",
                [name for name, _ in defs])
            self._nav_defs = defs
            self.phase = "nav_def"

    def _node_nav_def(self):
        return self._level_node(self._nav)

    def _apply_nav_def(self, node, decision) -> None:
        picked = self._nav.apply(decision)
        if picked is None:
            self._finish(False, "no definition chosen")
            return
        if picked == nav.MORE_LABEL:
            return
        rel = f"{self._nav_folder}/{self._nav_file}"
        kind = dict(self._nav_defs)[picked]
        if kind == "class":
            self._nav_class = picked
            methods = self.index.methods_of(rel, picked)
            self._nav = nav.LevelChoice(
                f"Which method of '{picked}' holds the code to fix?", methods)
            self.phase = "nav_method"
            return
        symbol = next(s for s in self.index.symbols
                      if s.file == rel and s.qualname == picked)
        self._set_target(symbol)
        self.phase = "operation"

    def _node_nav_method(self):
        return self._level_node(self._nav)

    def _apply_nav_method(self, node, decision) -> None:
        picked = self._nav.apply(decision)
        if picked is None:
            self._finish(False, "no method chosen")
            return
        if picked == nav.MORE_LABEL:
            return
        rel = f"{self._nav_folder}/{self._nav_file}"
        qual = f"{self._nav_class}.{picked}"
        symbol = next(s for s in self.index.symbols
                      if s.file == rel and s.qualname == qual)
        self._set_target(symbol)
        self.phase = "operation"

    def _level_node(self, choice: "nav.LevelChoice"):
        """A menu for the current nav level, or auto-resolve when trivial."""
        node = choice.node()
        if node is not None:
            return node
        if len(choice.items) == 1:
            # free-take: fabricate a decision with no model call at all
            from threetoks.nodes import Decision
            fake = Decision("menu", choice.items[0], "", valid=True)
            self.apply(None, fake)
            return None
        self._finish(False, "nothing to navigate to at this level")
        return None

    # ------------------------------------------------------------ 3. operation

    def _node_operation(self):
        options = [OPERATION_LABELS[op] for op in
                  (OP_REPLACE, OP_INSERT_AFTER, OP_DELETE)]
        return MenuNode(
            f"Fix request: {self.request}\nHow should this be fixed?",
            options, escape=True)

    def _apply_operation(self, node, decision) -> None:
        if not decision.valid or decision.value == ESCAPE:
            self._finish(False, "no operation chosen")
            return
        chosen = next(op for op, label in OPERATION_LABELS.items()
                     if label == decision.value)
        self.outcome.operation = chosen
        if chosen == OP_DELETE:
            self._start_delete()
        else:
            self._start_generate(chosen)

    # ------------------------------------------------------------ 4a. delete

    def _start_delete(self) -> None:
        source = self.index.source_of(self.target.file)
        spans = edit_ops.statement_spans(source, self.target.name)
        labels = [_clip(edit_ops.span_text(source, s, e)) for s, e in spans]
        self._delete_spans = spans
        self._delete_choice = nav.LevelChoice(
            f"Which line inside {self.target.name}() is wrong?", labels)
        self.phase = "delete_menu"

    def _node_delete_menu(self):
        return self._level_node(self._delete_choice)

    def _apply_delete_menu(self, node, decision) -> None:
        picked = self._delete_choice.apply(decision)
        if picked is None:
            self._finish(False, "no line chosen to delete")
            return
        if picked == nav.MORE_LABEL:
            return
        labels = [_clip(edit_ops.span_text(
            self.index.source_of(self.target.file), s, e))
            for s, e in self._delete_spans]
        span = self._delete_spans[labels.index(picked)]
        source = self.index.source_of(self.target.file)
        candidate = edit_ops.delete_span(source, *span)
        self._apply_and_check(candidate, allow_repair=self._repair_round == 0)

    # ------------------------------------------------------------ 4b. generate

    def _start_generate(self, operation: str) -> None:
        source = self.index.source_of(self.target.file)
        self._original_source = source
        if operation == OP_REPLACE:
            args = _function_args(source, self.target.name)
            splice_fn = edit_ops.replace_splice_fn(
                source, self.target.lineno, self.target.end_lineno,
                self.target.name, args)
            prefill = f"def {self.target.name}({args}):\n    "
            span = edit_ops.span_text(source, self.target.lineno,
                                      self.target.end_lineno)
            before, after = edit_ops.context_lines(
                source, self.target.lineno, self.target.end_lineno)
            instruction = self._generate_instruction(span, before, after)
        else:
            known = {s.name for s in self.index.symbols
                    if s.file == self.target.file}
            func_source = edit_ops.span_text(
                source, self.target.lineno, self.target.end_lineno)
            helper = edit_ops.find_missing_helper(
                func_source, self.target.name, known)
            if helper is None:
                self._finish(False, "insert-after: no undefined helper "
                            "found to add")
                return
            name, args = helper
            splice_fn = edit_ops.insert_after_splice_fn(
                source, self.target.end_lineno, name, args)
            prefill = f"def {name}({args}):\n    "
            instruction = (
                f"Fix request: {self.request}\n"
                f"Existing function (verbatim, do not repeat it):\n{func_source}\n"
                f"Write the missing helper function {name}({args}) that "
                "this code calls but that is never defined. Do not use "
                "pass or raise NotImplementedError -- write a real body.")
        temperature = REPAIR_TEMPERATURE if self._repair_round else None
        self._current_node = edit_ops.GenerateSpanNode(
            instruction, prefill, splice_fn, temperature)
        self.phase = "generate"

    def _generate_instruction(self, span: str, before: str, after: str) -> str:
        lines = [f"Fix request: {self.request}",
                 "Current function (verbatim):", span]
        if before:
            lines += ["Lines just before it:", before]
        if after:
            lines += ["Lines just after it:", after]
        lines.append("Write this function with a corrected, complete body. "
                    "Do not use pass or raise NotImplementedError -- write "
                    "a real fix.")
        return "\n".join(lines)

    def _node_generate(self):
        return self._current_node

    def _apply_generate(self, node, decision) -> None:
        if not decision.valid:
            self._finish(False, "generated edit never parsed after retries")
            return
        self._apply_and_check(decision.value, allow_repair=True)

    # ------------------------------------------------------------ 5. oracle + repair

    def _apply_and_check(self, candidate: str, allow_repair: bool) -> None:
        path = self.root / self.target.file
        path.write_text(candidate)
        ok, err = oracle.run_test(self.root, self.test_code)
        if ok:
            self._finish(True, "oracle test passes after edit")
            return
        path.write_text(self._original_source)      # restore before retrying
        if allow_repair and self._repair_round < MAX_REPAIR_ROUNDS \
                and self.steps_left > FORCE_DONE_AT_STEPS_LEFT:
            self._repair_round += 1
            self.outcome.repair_rounds = self._repair_round
            operation = self.outcome.operation
            if operation == OP_DELETE:
                self.phase = "operation"     # try a different operation entirely
            else:
                self._start_generate(operation)
        else:
            self._finish(False, f"oracle still fails after edit: {err[:120]}")

    # ------------------------------------------------------------ done

    def _finish(self, success: bool, reason: str) -> None:
        self.outcome.success = success
        self.outcome.reason = reason
        self._done_result = (success, reason)
        self.phase = "done"


def _function_args(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == name)
    return ", ".join(a.arg for a in node.args.args)


def _clip(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


if __name__ == "__main__":
    import tempfile

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.engine import run_episode
    from threetoks.policy import Policy, PolicyConfig

    class _Scripted:
        """Answers by grepping the rendered menu/prompt for a keyword."""

        def __init__(self, keyword_order, completion):
            self.keywords = list(keyword_order)
            self.completion = completion

        def complete(self, model, raw_prompt, opts):
            if "ACTIONS:" in raw_prompt:
                for kw in self.keywords:
                    for line in raw_prompt.splitlines():
                        if re.match(r"^\d+ = ", line) and kw in line:
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
        vertical = EditVertical(
            "fix the bug in add", root, test)
        backend = _Scripted(
            ["replace the target function's body"], "return a + b")
        policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = run_episode(vertical, policy, max_steps=20)
        assert outcome["success"], outcome
        assert outcome["path_taken"] == "direct"
    print("smoke OK")
