"""CodeVertical: plan a few functions, then write and check them one by one.

The forced pipeline (docs/DESIGN-coding-agent.md §3): PLAN once, then per
method in declaration order IMPLEMENT -> deterministic gates -> generate
tests once -> run asserts -> repair (resample) or finalize. A method that
never passes its gates is left as a NotImplementedError stub; a method that
passes gates but fails only advisory asserts keeps its runnable body. The
file on disk always parses because a body is stored only after ast checks.

Once every method is terminal, the runner phase (§7b) executes the whole
module for free — trusted examples plus one smoke call of the script
entry — and each failure's traceback is mapped to the method it happened
in. That blamed method is unlocked and re-enters the same implement flow
(blind resample at repair temperature — no error feedback, per E5b),
bounded by ``MAX_REPAIR_ROUNDS`` overall and ``MAX_METHOD_REPAIRS`` per
method. Runner checks run in ``run_cwd`` (a temp dir when None).
"""
import builtins
import re
from pathlib import Path

from threetoks.code import gates, runner, sandbox
from threetoks.code.nodes import (MAX_CAP_DOUBLINGS, ImplementNode, PlanNode,
                                 TestNode)
from threetoks.code.store import (STATUS_PLANNED, STATUS_STUBBED,
                                 STATUS_TESTED, MethodStore)
from threetoks.render import Episode

MAX_METHOD_ATTEMPTS = 3
REPAIR_TEMPERATURE = 0.4
FORCE_STUB_AT_STEPS_LEFT = 2
INITIAL_STEPS_LEFT = 40
MAX_REPAIR_ROUNDS = 2       # whole-module run -> unlock cycles
MAX_METHOD_REPAIRS = 2      # times one method may be unlocked
MIN_STEPS_TO_REPAIR = FORCE_STUB_AT_STEPS_LEFT + 2
FALLBACK_METHOD = ("main", "", "solve the whole task in one function")
# Compute mode needs a runnable entry whose return value is the answer.
COMPUTE_ENTRY = ("main", "", "compute and return the final answer the task "
                 "asks for, using the functions above")
# No space before "(": a real signature is name(args); prose like
# "the data (fast)" has a space and must not be mistaken for one.
_ENTRY = re.compile(r"([a-z_][a-z0-9_]*)\(([^)]*)\)")
_PARAM_LIST = re.compile(r"^([a-z_][a-z0-9_]*(\s*,\s*[a-z_][a-z0-9_]*)*)?$")


class CodeVertical:
    """Drives one module-writing episode from a natural-language goal.

    ``retriever`` is the optional retrieval-as-repair hook
    (threetoks/code/retrieve.py). The vertical consults it only when an
    anchor-bearing method used all its generation attempts and would
    otherwise stub; None disables it (the config default).

    ``trusted`` maps a method name to a list of trusted anchor
    expressions (a single str per entry is accepted). Two anchors, normal
    plus boundary, close the single-anchor blind spot E9 measured.

    With ``run_cwd`` set, the runner checks run in that folder under the
    project interpreter (``sandbox.project_python``); else in a temp dir
    under the harness interpreter.
    """

    def __init__(self, goal: str, gen_tests: bool = True,
                 trusted_examples: dict | None = None, retriever=None,
                 run_cwd: Path | None = None, echo_result: bool = False):
        """Set up one episode.

        The code agent creates one vertical per author or compute request.

        Args:
            goal: the user's request text.
            gen_tests: True asks the model for advisory asserts.
            trusted_examples: method name -> anchor expression(s).
            retriever: optional retrieval-as-repair hook, or None.
            run_cwd: working directory of the runner checks; None = temp.
            echo_result: True prints the entry's non-None return value
                in the smoke call and in the ``__main__`` guard.
        """
        self.goal = goal
        self.gen_tests = gen_tests
        self.run_cwd = run_cwd
        self.echo_result = echo_result
        self.retriever = retriever
        self.trusted = {name: value if isinstance(value, list) else [value]
                        for name, value in (trusted_examples or {}).items()}
        self.store = MethodStore(goal)
        self.episode = Episode("", goal)
        self.steps_left = INITIAL_STEPS_LEFT
        self.planned = False
        self.index = 0
        self.await_test = False
        self._repeated = False
        self.repair_round = 0
        self.run_results: list = []
        self._run_clean = False
        self._run_attempted = False

    # ------------------------------------------------------------ engine API

    def next_node(self):
        """Next generation node, or None when every method is terminal.

        When no method is pending, the free runner phase executes the
        module; a failing check may unlock its blamed method, in which
        case the episode continues with that method's implement node.
        """
        if not self.planned:
            self._set_episode()
            return PlanNode(self.goal)
        self._seek_pending()
        if self.index >= len(self.store.methods):
            self._rescue_stubs()
        if self.index >= len(self.store.methods):
            self._run_module_checks()
            if self.index >= len(self.store.methods):
                return None
        if self.steps_left <= FORCE_STUB_AT_STEPS_LEFT:
            self._stub_remaining()
            return None
        return self._node_for_current()

    def apply(self, node, decision) -> None:
        """Route one decision to the matching phase handler."""
        if isinstance(node, PlanNode):
            self._apply_plan(decision)
        elif isinstance(node, ImplementNode):
            self._apply_implement(decision)
        elif isinstance(node, TestNode):
            self._apply_test(decision)

    def result(self) -> dict:
        """The rendered script, per-method status, and the final run report.

        The engine calls this when the episode ends. The answer is the
        module plus a ``__main__`` guard when a runnable entry exists, so
        the delivered file executes as ``python3 module.py``, doing
        exactly what the runner's smoke call already exercised.

        Returns:
            dict: goal, answer, methods, output, run.
        """
        return {"goal": self.goal,
                "answer": self._render_script(self._script_entry()),
                "methods": [self._method_row(m) for m in self.store.methods],
                "stubbed": self.store.stubbed_names(),
                "output": self._entry_output(),
                "run": self._run_report()}

    def _render_script(self, entry: str | None) -> str:
        """The module plus a guard that runs the entry like the smoke call.

        result() calls this. The guard body comes from
        ``runner.entry_call``, so ``echo_result`` behaves the same in the
        smoke call and in the delivered script.

        Args:
            entry: the script entry from _script_entry, or None.

        Returns:
            str: the module, with a ``__main__`` guard when entry is set.
        """
        module = self.store.render_module()
        if entry is None:
            return module
        call = runner.entry_call(entry, self.echo_result, gates.BODY_INDENT)
        return f'{module}\n\nif __name__ == "__main__":\n{call}\n'

    def _entry_output(self) -> str:
        """What the delivered script prints when run, from the runner.

        result() calls this. Only a PASSED entry check counts: a crashed
        or interactive run gives no output (its error stays in the run
        report).

        Returns:
            str: stdout of the passed entry call, else "".
        """
        entry = self._script_entry()
        return next((r.stdout for r in self.run_results
                     if r.passed and r.check.name == entry
                     and r.check.kind == runner.KIND_CALL), "")

    def _script_entry(self) -> str | None:
        """The function the ``__main__`` guard calls, or None for no guard.

        Deterministic preference: a bodied zero-arg-callable ``main``,
        else the goal-named entry when bodied and zero-arg-callable, else
        the module's sole bodied zero-arg-callable function (all-default
        params count — ``say_hello(name="World")`` runs as a script).
        Never a stub and never an arbitrary helper — a script that
        crashes or picks randomly is worse than no guard at all. The
        "sole function" guess is skipped when a method is stubbed: the
        module is unfinished, and that one helper is not its entry.
        """
        zero_arg = [m.name for m in self.store.methods
                    if m.body is not None
                    and gates.zero_arg_callable(m.args)]
        if "main" in zero_arg:
            return "main"
        match = _ENTRY.search(self.goal)
        if match and match.group(1) in zero_arg:
            return match.group(1)
        sole = len(zero_arg) == 1 and not self.store.stubbed_names()
        return zero_arg[0] if sole else None

    # ------------------------------------------------------------ planning

    def _apply_plan(self, decision) -> None:
        """Populate the store from the plan; guarantee the goal's entry."""
        methods = decision.value if decision.valid else []
        for parsed in methods:
            if not self._plannable(parsed.name, parsed.args):
                continue
            record = self.store.add(parsed.name, parsed.args, parsed.contract,
                                    self.trusted.get(parsed.name))
            if parsed.advisory_assert:
                record.asserts.append(parsed.advisory_assert)
        self._ensure_entry()
        if not self.store.methods:
            self.store.add(*FALLBACK_METHOD)
        self._ensure_compute_entry()
        self.planned = True

    def _plannable(self, name: str, args: str) -> bool:
        """True when ``def name(args):`` renders a legal, non-shadowing stub.

        The lenient plan parser can emit garbage like ``print(i * i)`` —
        a builtin-shadowing name with an expression for a parameter list.
        Admitting it breaks the render invariant (the stub itself is a
        SyntaxError) and its candidate-import gate then stubs every other
        method too, so one junk plan line must never reach the store.
        """
        if hasattr(builtins, name):
            return False
        return gates.parses(f"def {name}({args}):\n    pass")

    def _ensure_entry(self) -> None:
        """Add the function the goal names by signature, if the plan missed it.

        E6 showed the model can decompose into helpers yet omit the entry
        the task asked for. When the goal contains a clean ``name(params)``,
        the harness guarantees that method exists so the interface is met.
        A signature that _plannable rejects (keyword params such as
        ``from``, builtin names) is skipped, so the module always parses.
        """
        match = _ENTRY.search(self.goal)
        if match is None:
            return
        name, args = match.group(1), match.group(2).strip()
        if not _PARAM_LIST.match(args) or not self._plannable(name, args):
            return
        if name not in self.store.names():
            self.store.add(name, args, f"the {name} function the task asks for")

    def _ensure_compute_entry(self) -> None:
        """Add a zero-arg ``main`` when compute mode has no runnable entry.

        Compute mode answers with the printed return value of the script
        entry. ``_apply_plan`` calls this after the plan is stored. A live
        plan gave only ``sum_primes(limit)``, so no entry could run and the
        answer was empty. This adds ``main()`` last, so it is written after
        the helpers it calls.

        Args:
            none (reads ``self.echo_result`` and the planned methods).

        Returns:
            None. The store gains one method when the guard is needed.
        """
        if not self.echo_result:
            return
        if any(gates.zero_arg_callable(m.args) for m in self.store.methods):
            return
        self.store.add(*COMPUTE_ENTRY)

    # ------------------------------------------------------------ implement

    def _node_for_current(self):
        """A test node when a body awaits tests, else an implement node."""
        method = self.store.methods[self.index]
        if self.await_test:
            self._set_episode(f"{method.name}({method.args}):"
                              f" {method.contract}")
            return TestNode(method.name, method.args)
        self._set_episode(self._sibling_context())
        heated = method.attempts or method.repairs   # repairs resample varied
        temperature = REPAIR_TEMPERATURE if heated else None
        return ImplementNode(method.name, method.args, method.contract,
                             method.examples, temperature, method.retry)

    def _apply_implement(self, decision) -> None:
        """Gate a fresh body; store it and move to tests, or repair/stub."""
        method = self.store.methods[self.index]
        method.attempts += 1
        method.retry.undefined = []
        source = decision.value if decision.valid else None
        self._repeated = source is not None and self._is_repeat(method, source)
        if source is None or not self._gates_pass(source, method):
            self._close_or_repair(method, discard=True)
            return
        method.retry.cut_off = False
        method.body = gates.ensure_docstring(source, method.contract)
        if self.gen_tests and not method.asserts and not method.examples:
            self.await_test = True
            return
        self._run_asserts(method)

    def _apply_test(self, decision) -> None:
        """Cache the model's asserts, then run all asserts for this method."""
        method = self.store.methods[self.index]
        self.await_test = False
        method.asserts.extend(a for a in decision.value if a not in method.asserts)
        self._run_asserts(method)

    def _run_asserts(self, method) -> None:
        """Trusted example gates; advisory asserts keep the body on failure.

        ``verified`` means real asserts actually passed — a body with no
        asserts is gated but not verified, so the flag never reads True
        vacuously (E6: model self-tests are optional, don't fake the signal).
        A failing example whose traceback lands in a *different* method is
        that method's fault, not this one's: the body is kept (unverified)
        and the culprit is queued for repair instead of burning attempts
        resampling an innocent function (§7b). An EOF/interrupt death is
        inconclusive — interactive code, same rule as the runner phase —
        so the body is kept unverified rather than discarded.
        """
        module = self.store.render_module()
        if method.examples:
            ok, stderr = gates.execute_asserts(
                module, [f"assert {e}" for e in method.examples])
            if not ok and runner.is_inconclusive(stderr):
                return self._finalize(method, verified=False)
            if not ok and self._blame_elsewhere(method, module, stderr):
                return self._finalize(method, verified=False)
            if not ok:
                return self._close_or_repair(method, discard=True)
        passed, _ = gates.run_asserts(module, method.asserts)
        if not passed:
            return self._close_or_repair(method, discard=False)
        checked = bool(method.examples) or bool(method.asserts)
        self._finalize(method, verified=checked)

    def _blame_elsewhere(self, method, module: str, stderr: str) -> bool:
        """True when the example failure is another method's to fix.

        The culprit is either unlocked for repair or still pending (an
        unwritten sibling this method legitimately calls); both mean the
        current body gets the benefit of the doubt until the runner phase
        re-checks it against the finished module.
        """
        culprit = runner.blame(stderr, runner.method_spans(module),
                               method.name)
        if culprit == method.name:
            return False
        return self._unlock(culprit) or self._is_pending(culprit)

    def _close_or_repair(self, method, discard: bool) -> None:
        """Resample while attempts and novelty remain, else finalize/stub.

        ``discard`` means the body is known bad — it failed a gate or the
        trusted example — so on exhaustion it must be stubbed, never kept.
        Only an advisory-only failure (``discard=False``) keeps a runnable
        body, since the failing assert may itself be wrong.
        """
        self.await_test = False
        if method.attempts < MAX_METHOD_ATTEMPTS and not self._repeated:
            if discard:
                method.body = None
            return
        keep_body = method.body is not None and not discard
        if not keep_body and self._try_retrieval(method):
            return
        self._finalize(method, verified=False, stub=not keep_body)

    def _try_retrieval(self, method) -> bool:
        """Last resort before stubbing: fetch a classic implementation.

        Fires only for anchor-bearing methods (no anchor, no free judge
        for foreign code) and only after generation exhausted its
        attempts — E9's evidence: the retrieval tail is exactly where
        blind generation loses (roman_to_int, caesar_encode), and the
        subprocess anchor check replaces model relevance judgment.
        """
        if self.retriever is None or not method.examples:
            return False
        try:
            source = self.retriever(method.name, method.args,
                                    method.contract, method.examples)
        except Exception:
            return False       # network trouble never breaks the episode
        if source is None:
            return False
        module = self._candidate_module(method, source)
        if not gates.import_ok(module)[0]:
            return False
        passed, _ = gates.execute_asserts(
            module, [f"assert {e}" for e in method.examples])
        if not passed:
            return False
        method.body = source
        method.retrieved = True
        self._finalize(method, verified=True)
        return True

    def _finalize(self, method, verified: bool, stub: bool = False) -> None:
        """Lock a method as tested (has body) or stubbed, and advance.

        A repair that ends in a stub gets the body from before the repair
        back: finished code never turns into a NotImplementedError stub.
        """
        if stub and method.fallback is not None:
            method.body = method.fallback
            stub, verified = False, False
        method.verified = verified
        method.status = STATUS_STUBBED if stub else STATUS_TESTED
        if stub:
            method.body = None
        self.index += 1
        self.await_test = False

    # ------------------------------------------------------------ gates

    def _gates_pass(self, source: str, method) -> bool:
        """Signature, no-placeholder, undefined-name, import — deterministic.

        Undefined names are kept on the method, so the next attempt's
        prompt can name them (a mechanical fact, not a guess).
        """
        if not gates.signature_matches(source, method.name, method.args):
            return False
        if gates.is_placeholder(source, method.name):
            return False                       # pass / NotImplementedError cheat
        method.retry.undefined = gates.undefined_names(
            source, set(self.store.names()))
        if method.retry.undefined:
            return False
        candidate = self._candidate_module(method, source)
        return gates.import_ok(candidate)[0]

    def _candidate_module(self, method, source: str) -> str:
        """Render the module with this method's body swapped in."""
        saved = method.body
        method.body = source
        rendered = self.store.render_module()
        method.body = saved
        return rendered

    # ------------------------------------------------------------ runner phase

    def _run_module_checks(self) -> None:
        """Free: execute the finished module; unlock what the blame names.

        Runs trusted examples and the script entry's smoke call via the
        runner, in ``run_cwd``.
        Every real failure unlocks its blamed method (round- and
        per-method-capped); with no budget left the run is report-only.
        A clean run latches ``_run_clean`` so the phase never re-executes.
        """
        if self._run_clean:
            return
        if not any(m.body for m in self.store.methods):
            self.run_results = []          # nothing runnable to report on
            return
        module = self.store.render_module()
        checks = runner.collect_checks(self.store.methods,
                                       self._script_entry(), self.echo_result)
        self.run_results = runner.run_checks(module, checks, self.run_cwd,
                                             interpreter=self._interpreter())
        self._run_attempted = True
        failed = runner.failures(self.run_results)
        if not failed:
            self._run_clean = True
            self._mark_run_verified()
            return
        if self.repair_round >= MAX_REPAIR_ROUNDS or \
                self.steps_left <= MIN_STEPS_TO_REPAIR:
            return
        self.repair_round += 1
        for failure in failed:
            self._unlock(failure.blamed)

    def _interpreter(self) -> str | None:
        """The interpreter for the runner checks.

        _run_module_checks calls this. With ``run_cwd`` the module is the
        user's compute script, so it runs under the project interpreter
        (contract C1); else the harness interpreter is used.

        Returns:
            str | None: project python path, or None for sys.executable.
        """
        if self.run_cwd is None:
            return None
        return sandbox.project_python(Path(self.run_cwd))

    def _unlock(self, name: str) -> bool:
        """Reopen a finished method for one more implement cycle.

        Keeps ``seen_hashes`` so a resample that reproduces the same body
        still trips the oscillation guard, and keeps the old body so the
        module stays runnable for sibling checks until it is replaced.
        Re-seeks the cursor on success — an unlock can reopen a method
        earlier than ``index``, and the caller may be mid-``next_node``.
        """
        method = next((m for m in self.store.methods if m.name == name), None)
        if method is None or method.status == STATUS_PLANNED \
                or method.repairs >= MAX_METHOD_REPAIRS:
            return False
        method.repairs += 1
        method.status = STATUS_PLANNED
        method.attempts = 0
        if method.body is not None:
            method.fallback = method.body
        method.retry.cut_off = False
        method.retry.undefined = []
        method.verified = False
        self._seek_pending()
        return True

    def _is_pending(self, name: str) -> bool:
        """True when the named method has not been finalized yet."""
        method = next((m for m in self.store.methods if m.name == name), None)
        return method is not None and method.status == STATUS_PLANNED

    def _mark_run_verified(self) -> None:
        """A passed example check is a real verification — record it."""
        passed_examples = {r.check.name for r in self.run_results
                           if r.passed and r.check.kind == runner.KIND_EXAMPLE}
        for method in self.store.methods:
            if method.name in passed_examples:
                method.verified = True

    def _run_report(self) -> dict:
        """The final run outcome for the result payload.

        ``ran`` distinguishes "executed and found nothing wrong" from "the
        runner phase never got to execute" (budget spent, all stubs) — the
        two must not read the same in an honest report.
        """
        stubbed = bool(self.store.stubbed_names())
        return {"ok": self._run_clean and not stubbed,
                "ran": self._run_attempted,
                "checks": len(self.run_results), "rounds": self.repair_round,
                "failures": [{"check": r.check.name, "kind": r.check.kind,
                              "error": r.error, "blamed": r.blamed}
                             for r in runner.failures(self.run_results)]}

    # ------------------------------------------------------------ helpers

    def _seek_pending(self) -> None:
        """Point ``index`` at the first method still awaiting work.

        A repair can reopen a method *earlier* than the cursor, so this
        scans from the top instead of only skipping forward.
        """
        finished = {STATUS_TESTED, STATUS_STUBBED}
        self.index = next(
            (i for i, m in enumerate(self.store.methods)
             if m.status not in finished), len(self.store.methods))

    def _stub_remaining(self) -> None:
        """Out of budget: stub every method still lacking a body."""
        for method in self.store.methods:
            if method.status not in {STATUS_TESTED, STATUS_STUBBED}:
                self._finalize(method, verified=False, stub=method.body is None)

    def _rescue_stubs(self) -> None:
        """Give every stubbed method one more try before the module ends.

        next_node calls this when all methods are terminal. A stub means
        unfinished code, so each stubbed method is reopened once while
        the step budget allows. By now its siblings have bodies, so the
        prompt shows more context, and its token cap has already grown.
        The method may stub again; it is never reopened twice here. A
        method cut off even at the highest cap is not reopened: the same
        prompt would be cut off again.
        """
        if self.steps_left <= MIN_STEPS_TO_REPAIR:
            return
        for method in self.store.methods:
            if method.status == STATUS_STUBBED and method.repairs == 0 \
                    and method.retry.truncations <= MAX_CAP_DOUBLINGS:
                self._unlock(method.name)

    def _is_repeat(self, method, source: str) -> bool:
        """Track body hashes; True if this body was already generated."""
        key = hash(source.strip())
        repeat = key in method.seen_hashes
        method.seen_hashes.add(key)
        return repeat

    def _sibling_context(self) -> str:
        """Signature lines of the other methods, written or not yet.

        Methods that have no body are listed as "written separately": the
        model then sees the scope of the other functions and keeps its
        own function small (a whole-program answer was cut off by the
        token cap, again and again).
        """
        done = self.store.bodied_except(self.index)
        text = ("ALREADY IMPLEMENTED (you may call these):\n"
                + self.store.signature_lines(done))
        later = self.store.unwritten_except(self.index)
        if not later:
            return text
        return (text + "\nWRITTEN SEPARATELY (you may call these, do not "
                "write them):\n" + self.store.signature_lines(later))

    def _set_episode(self, observation: str = "") -> None:
        """Rebuild a fresh stateless micro-prompt for the next generation.

        next_node calls this before each generation node. The episode has
        no prefix: the node's own ``system`` text reaches the model once,
        in the system block that Policy sends.

        Args:
            observation: context text for the node, or "" for none.
        """
        self.episode = Episode("", self.goal)
        if observation:
            self.episode.open_observation(observation)

    def _method_row(self, method) -> dict:
        """One method's outcome for the result payload."""
        return {"name": method.name, "status": method.status,
                "verified": method.verified, "attempts": method.attempts,
                "repairs": method.repairs, "retrieved": method.retrieved}


def _demo_backend(bodies: list[str]):
    """A scripted backend that returns queued texts (offline smoke only)."""
    from threetoks.backend.base import GenResult

    class _Scripted:
        def __init__(self):
            self.queue = list(bodies)

        def complete(self, model, raw_prompt, opts):
            text = self.queue.pop(0) if self.queue else "return None"
            return GenResult(text, 10, 5, 0.0, "stop")

    return _Scripted()


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, ModelSpec
    from threetoks.engine import run_episode
    from threetoks.policy import Policy, PolicyConfig

    scripted = _demo_backend([
        "double(n): returns n times two",                 # plan
        "return n * 2",                                    # implement double
        "assert double(3) == 6\nassert double(0) == 0"])   # tests
    policy = Policy(scripted, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    vertical = CodeVertical("double a number", gen_tests=True)
    outcome = run_episode(vertical, policy, max_steps=20)
    assert "def double(n):" in outcome["answer"], outcome["answer"]
    assert "return n * 2" in outcome["answer"]
    row = outcome["methods"][0]
    assert row["status"] == "tested" and row["verified"], row
    import ast
    ast.parse(outcome["answer"])

    # two trusted anchors: a body passing only the normal case is rejected
    two_anchor = CodeVertical(
        "check primality with is_prime(n)", gen_tests=False,
        trusted_examples={"is_prime": ["is_prime(17) == True",
                                       "is_prime(1) == False"]})
    naive = _demo_backend([
        "is_prime(n): true when n is prime",
        "for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n            return False\n    return True",
        "if n < 2:\n        return False\n"
        "    for i in range(2, int(n ** 0.5) + 1):\n"
        "        if n % i == 0:\n            return False\n    return True"])
    two_policy = Policy(naive, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    two_outcome = run_episode(two_anchor, two_policy, max_steps=20)
    assert "n < 2" in two_outcome["answer"], two_outcome["answer"]
    assert two_outcome["methods"][0]["verified"], two_outcome["methods"]

    echo = CodeVertical("compute the answer", gen_tests=False,
                        echo_result=True)
    echo_policy = Policy(_demo_backend(["main(): the answer", "return 6 * 7"]),
                         PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    echo_outcome = run_episode(echo, echo_policy, max_steps=20)
    assert echo_outcome["output"] == "42\n", echo_outcome
    assert "print(_entry_result)" in echo_outcome["answer"]

    from threetoks.nodes import Decision
    entry = CodeVertical("Write triple(n) that multiplies n by three")
    entry._apply_plan(Decision("plan", [], "", valid=False))  # empty plan
    assert "triple" in entry.store.names(), entry.store.names()
    prose = CodeVertical("process the data (fast) please")     # parenthetical
    prose._apply_plan(Decision("plan", [], "", valid=False))
    assert prose.store.names() == ["main"], prose.store.names()  # no 'data'
    keyword = CodeVertical("write convert(amount, from, to)")   # `from`
    keyword._apply_plan(Decision("plan", [], "", valid=False))
    assert gates.parses(keyword.store.render_module()), keyword.store.names()
    print("smoke OK")
