"""EditVertical: the state machine driving one edit episode.

Implements the engine's Vertical protocol (run_episode + Policy). It
owns the phases and the repair ladder and delegates the work to three
model-free collaborators: search.TargetSearch (exact name or shortlist
pick, then confirm by pointing), operations.TargetOperations (free
insert inference, viable operations, delete menu, delta-only generate
nodes prefilled with the verbatim def header), and check.CandidateChecker
(caller test, else repo tests, else import; original bytes always
restored). UNCHANGED output strikes the target and resumes the search.
Success returns the candidate in result(); the agent decides to apply.

Repair escalates instead of looping: regenerate once at
REPAIR_TEMPERATURE, re-ask the operation when more than one is viable,
then strike the target and resume the search. MAX_REPAIRS_TOTAL and the
step budget bound the episode. MENU decisions read the growing session
log; GENERATE decisions get a fresh episode and never see failure text
(E5b: blind resampling beats error feedback, DESIGN-coding-agent.md §2).
"""
from dataclasses import dataclass, field
from pathlib import Path

from threetoks.code.edit import nav
from threetoks.code.edit.check import ERROR_CLIP, CandidateChecker
from threetoks.code.edit.index import SymbolIndex
from threetoks.code.edit.operations import (OP_DELETE, OP_INSERT_AFTER,
                                            TargetOperations, operation_menu,
                                            operation_of)
from threetoks.code.edit.search import ACCEPTED, FAILED, TargetSearch
from threetoks.code.edit.splice import UNCHANGED, canonical_key
from threetoks.nodes import ESCAPE
from threetoks.render import Episode

# Failed checks tolerated per episode: the 6th failure ends it.
MAX_REPAIRS_TOTAL = 6
FORCE_DONE_AT_STEPS_LEFT = 2
REPAIR_TEMPERATURE = 0.4
STAGE_RESAMPLE, STAGE_REASK = 0, 1      # per-target escalation ladder

SYSTEM_MENU = ("You are a menu selector controlling a code-editing agent. "
               "Reply with exactly one digit: the number of the best action.")


@dataclass
class EditOutcome:
    """What happened in one episode (see result() for the meaning)."""
    success: bool = False
    verified: bool = False
    reason: str = ""
    operation: str = ""
    edited_symbol: str = ""
    repairs: int = 0
    candidate: str = ""
    tests: dict = field(default_factory=dict)


@dataclass
class _Repair:
    """Repair state: escalation stage and what already failed."""
    stage: int = 0                      # per target
    total: int = 0                      # per episode
    skip_inference: bool = False
    tried: set = field(default_factory=set)   # canonical keys, per target
    tried_deletes: set = field(default_factory=set)  # (start, end) spans


class EditVertical:
    """Drives one edit episode against a corpus of .py files.

    ``test_code`` is the trusted oracle (must fail before the edit, pass
    after); without it the checker falls back to repo tests or an import
    check and ``verified`` stays honest.
    """

    def __init__(self, request: str, corpus_root: Path,
                 test_code: str | None = None):
        """Prepare one episode; no model call and no disk write.

        The code agent and tests create one vertical per edit request,
        with the user's request, the corpus root, and an optional test.
        It builds the symbol index, the search, and the checker.

        Args:
            request: str, the user's request text.
            corpus_root: Path, root of the .py corpus (Services.files_root).
            test_code: str | None, trusted caller test source, or None.
        """
        self.request = request
        self.root = Path(corpus_root)
        self.test_code = test_code
        self.index = SymbolIndex(self.root)
        self._menu_episode = Episode(SYSTEM_MENU, request)
        self.episode = self._menu_episode
        self.steps_left = 0
        self.phase = "start"
        self.outcome = EditOutcome()
        self._search = TargetSearch(self.index, request, self._menu_episode)
        self._checker = CandidateChecker(self.index, test_code,
                                         self._menu_episode.log_session)
        self._repair = _Repair()
        self._ops: TargetOperations | None = None   # per accepted target
        self._edited = ""          # symbol the pending candidate changes
        self._delete_menu = None   # operations.DeleteMenu when deleting
        self._generate_node = None   # GenerateSpanNode when generating

    @property
    def target(self):
        """The current target Symbol (shown or accepted), or None."""
        return self._search.target

    # ------------------------------------------------------------ engine API

    def next_node(self):
        """Give the next decision node, running free phases in between.

        run_episode calls this before each model call. Phases without a
        node resolve here for free and the loop continues.

        Returns:
            MenuNode | GenerateSpanNode | None: None when the episode ends.
        """
        while self.phase != "done":
            if self.phase == "start":
                self._run_precheck()
                continue
            if self.steps_left <= FORCE_DONE_AT_STEPS_LEFT:
                self._finish(False, "step budget exhausted")
                continue
            node = getattr(self, f"_node_{self.phase}")()
            if node is not None:
                self.episode = Episode("", self.request) \
                    if self.phase == "generate" else self._menu_episode
                return node
        return None

    def apply(self, node, decision) -> None:
        """Route one decision to the current phase's handler.

        Args:
            node: the node next_node returned.
            decision: Decision, the Policy's parse of the model answer.
        """
        getattr(self, f"_apply_{self.phase}")(node, decision)

    def result(self) -> dict:
        """Give the episode outcome as a plain dict (contract C7).

        run_episode returns this to the code agent, which shows a diff of
        original -> candidate and applies it only after confirmation.

        Returns:
            dict: success, verified, reason, path_taken, operation, target,
            target_file, edited_symbol, repairs, original, candidate,
            tests, index_truncated.
        """
        out, target = self.outcome, self.target
        target_file = target.file if target else ""
        return {"success": out.success, "verified": out.verified,
                "reason": out.reason, "path_taken": self._search.path_taken,
                "operation": out.operation,
                "target": target.qualname if target else "",
                "target_file": target_file,
                "edited_symbol": out.edited_symbol, "repairs": out.repairs,
                "original": self.index.source_of(target_file),
                "candidate": out.candidate if out.success else "",
                "tests": dict(out.tests),
                "index_truncated": self.index.truncated}

    # ------------------------------------------------------------ search

    def _run_precheck(self) -> None:
        """Run the free gate before any model call, then start the search.

        Only next_node calls this, once. A caller test that already
        passes means there is nothing to fix.
        """
        if self._checker.caller_test_passes():
            self._finish(False, "test passes before any edit — "
                         "nothing to fix")
            return
        self._search.start()
        self._resume_search()

    def _resume_search(self) -> None:
        """Run the search phase; accept or finish when the search ended.

        The precheck, the search menus, and _strike call this.
        """
        self.phase = "search"
        if self._search.state == ACCEPTED:
            self._accept_target()
        elif self._search.state == FAILED:
            self._finish(False, self._search.failure)

    def _node_search(self):
        """Give the search's pick or confirm menu.

        Returns:
            MenuNode | None: None when the search ended (handled here).
        """
        node = self._search.node()
        if node is None:
            self._resume_search()
        return node

    def _apply_search(self, node, decision) -> None:
        """Pass a pick or confirm answer to the search.

        Args:
            node: MenuNode from _node_search.
            decision: Decision from the Policy.
        """
        self._search.apply(decision)
        self._resume_search()

    def _accept_target(self) -> None:
        """Fix the confirmed target and prepare its check.

        It resets the per-target repair state (a body that failed on one
        target may be right for another); a module that does not import
        ends the episode here, before any generation.
        """
        target = self.target
        self._ops = TargetOperations(self.index, target)
        self._repair = _Repair(total=self._repair.total)
        self.outcome.operation = ""
        self.outcome.tests = {}
        failure = self._checker.prepare(target)
        if failure:
            self._finish(False, failure)
        else:
            self.phase = "operation"

    # ------------------------------------------------------------ operation

    def _node_operation(self):
        """Infer the operation for free when possible, else one menu.

        The re-ask escalation stage sets ``skip_inference`` so that the
        real menu shows instead of the inference that already failed.

        Returns:
            MenuNode | None: the operation menu, or None when free.
        """
        if not self._repair.skip_inference and self._infer_insert():
            return None
        self._repair.skip_inference = False
        viable = self._ops.viable(self._repair.tried_deletes)
        if len(viable) == 1:
            self._begin_operation(viable[0])
            return None
        return operation_menu(viable)

    def _infer_insert(self) -> bool:
        """Start insert-after for free when the target calls a missing name.

        Returns:
            bool: True when the operation was inferred and started.
        """
        helper = self._ops.missing_helper()
        if helper is None:
            return False
        self._menu_episode.log_session(
            f"> inferred: insert missing helper {helper[0]}()")
        self._begin_operation(OP_INSERT_AFTER)
        return True

    def _apply_operation(self, node, decision) -> None:
        """Start the chosen operation; escape gives up on this target.

        Args:
            node: MenuNode from _node_operation.
            decision: Decision from the Policy.
        """
        if not decision.valid:
            self._finish(False, "no operation chosen")
            return
        if decision.value == ESCAPE:
            self._strike("no operation fits the target",
                         f"giving up on {self.target.qualname}")
            return
        self._begin_operation(operation_of(decision.value))

    def _begin_operation(self, operation: str) -> None:
        """Record the operation and start it.

        Args:
            operation: str, one of OP_REPLACE, OP_INSERT_AFTER, OP_DELETE.
        """
        self.outcome.operation = operation
        self._menu_episode.log_session(f"> op: {operation}")
        if operation == OP_DELETE:
            self._start_delete()
        else:
            self._start_generate(operation)

    # ------------------------------------------------------------ delete

    def _start_delete(self) -> None:
        """Offer the deletable statements as one paged menu."""
        menu = self._ops.delete_menu(self._repair.tried_deletes)
        if menu is None:
            self._escalate("every deletable line was already tried")
            return
        self._delete_menu = menu
        self.phase = "delete_menu"

    def _node_delete_menu(self):
        """Give the delete menu; a lone statement is taken for free.

        Returns:
            MenuNode | None: the current page, or None when free.
        """
        node = self._delete_menu.node()
        if node is None:
            self._picked_delete(self._delete_menu.free_take())
        return node

    def _apply_delete_menu(self, node, decision) -> None:
        """Delete the picked line; escape escalates.

        Args:
            node: MenuNode from _node_delete_menu.
            decision: Decision from the Policy.
        """
        picked = self._delete_menu.apply(decision)
        if picked is None:
            self._escalate("no line chosen to delete")
        elif picked != nav.MORE_LABEL:
            self._picked_delete(picked)

    def _picked_delete(self, label: str) -> None:
        """Check the file without the picked statement.

        Args:
            label: str, the delete menu label of the statement.
        """
        span = self._delete_menu.spans[label]
        self._repair.tried_deletes.add(span)
        self._edited = self.target.qualname
        self._check_candidate(self._ops.without(span), "")

    # ------------------------------------------------------------ generate

    def _start_generate(self, operation: str) -> None:
        """Build the delta micro-prompt for replace or insert-after.

        Args:
            operation: str, OP_REPLACE or OP_INSERT_AFTER.
        """
        node, edited = self._ops.generate_node(operation, self._repair.tried)
        if node is None:
            self._escalate(edited)
            return
        node.temperature = REPAIR_TEMPERATURE if self._repair.stage else None
        self._edited = edited
        self._generate_node = node
        self.phase = "generate"

    def _node_generate(self):
        """Give the pending generate node.

        Returns:
            GenerateSpanNode: the node built by _start_generate.
        """
        return self._generate_node

    def _apply_generate(self, node, decision) -> None:
        """Check a candidate; UNCHANGED moves on; invalid escalates.

        Args:
            node: GenerateSpanNode from _start_generate.
            decision: Decision; value is (file, function) or UNCHANGED.
        """
        if not decision.valid:
            self._escalate("no candidate survived the generation gates")
            return
        if decision.value == UNCHANGED:
            why = ("the model found nothing to change in "
                   f"{self.target.qualname}")
            self._strike(why, why)
            return
        candidate, new_source = decision.value
        self._check_candidate(candidate, new_source)

    # ----------------------------------------------------------- check, repair

    def _check_candidate(self, candidate: str, new_source: str) -> None:
        """Check one candidate; keep it in the outcome or escalate.

        Args:
            candidate: str, the full new file text (LF).
            new_source: str, the new function source, "" for a delete.
        """
        result = self._checker.check(self.target, candidate)
        if result.tests:
            self.outcome.tests = result.tests
        if not result.passed:
            if new_source:
                self._repair.tried.add(canonical_key(new_source))
            self._escalate(f"{self.outcome.operation} edit failed: "
                           f"{result.error[:ERROR_CLIP]}")
            return
        self.outcome.verified = result.verified
        self.outcome.candidate = candidate
        self.outcome.edited_symbol = self._edited
        self._finish(True, self._checker.success_reason(result))

    def _escalate(self, why: str) -> None:
        """Widen the repair scope instead of looping on one wrong choice.

        Per target: regenerate once, then re-ask the operation when more
        than one is viable, then go back to the shortlist without this
        target. MAX_REPAIRS_TOTAL bounds the whole episode.

        Args:
            why: str, the failure line (menus see it, generations do not).
        """
        self._menu_episode.log_session(f"> {why}")
        self._repair.total += 1
        self.outcome.repairs = self._repair.total
        if self._repair.total >= MAX_REPAIRS_TOTAL \
                or self.steps_left <= FORCE_DONE_AT_STEPS_LEFT:
            self._finish(False, why)
            return
        stage = self._repair.stage
        self._repair.stage += 1
        if stage == STAGE_RESAMPLE and self.outcome.operation:
            self._retry_same_operation()
        elif stage <= STAGE_REASK \
                and len(self._ops.viable(self._repair.tried_deletes)) > 1:
            self._repair.skip_inference = True
            self.phase = "operation"
        else:
            self._strike(why, f"giving up on {self.target.qualname}")

    def _retry_same_operation(self) -> None:
        """Run the same operation again (delete re-menus, others resample)."""
        if self.outcome.operation == OP_DELETE:
            self._start_delete()
        else:
            self._start_generate(self.outcome.operation)

    def _strike(self, why: str, note: str) -> None:
        """Drop the target and resume the search without it.

        UNCHANGED output, a failed repair ladder, and an operation-menu
        escape call this.

        Args:
            why: str, the finish reason when no candidate remains.
            note: str, the session log line for menus.
        """
        self._menu_episode.log_session(f"> {note}")
        self._search.strike(why)
        self._resume_search()

    def _finish(self, success: bool, reason: str) -> None:
        """End the episode with an outcome.

        Args:
            success: bool, True when a candidate passed its check.
            reason: str, one line for the user.
        """
        self.outcome.success = success
        self.outcome.reason = reason
        self.phase = "done"


if __name__ == "__main__":
    import tempfile

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.engine import run_episode
    from threetoks.policy import Policy, PolicyConfig

    class _Scripted:
        """Menus: the "yes" option; generation: the fix."""

        def complete(self, model, raw_prompt, opts):
            if "ACTIONS:" not in raw_prompt:
                return GenResult("return a + b", 10, 1, 0.0, "stop")
            yes = next(line for line in raw_prompt.split("\n")
                       if " = yes" in line)
            return GenResult(" " + yes.split(" = ")[0], 10, 1, 0.0, "stop")

    with tempfile.TemporaryDirectory() as tmp:
        target_file = Path(tmp) / "mod.py"
        target_file.write_text("def add(a, b):\n    return a - b\n")
        test = "from mod import add\nassert add(2, 2) == 4\n"
        config = PolicyConfig(ModelSpec("m", FAMILY_CHATML))
        outcome = run_episode(EditVertical("fix the bug in add", Path(tmp),
                                           test),
                              Policy(_Scripted(), config), max_steps=20)
        assert outcome["success"] and outcome["verified"], outcome
        assert outcome["path_taken"] == "direct"
        assert "a + b" in outcome["candidate"]
        assert target_file.read_text() == "def add(a, b):\n    return a - b\n"
    print("smoke OK")
