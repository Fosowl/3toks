"""Module runner: execute the assembled module and blame failures.

Zero model calls. After the vertical finishes its methods, the runner
executes the rendered module in a sandboxed subprocess — once per check —
and turns each failure into a *blamed method* by mapping the traceback's
``File "<stdin>", line N`` frames onto the module's function spans. The
deepest frame that lands inside a def wins: a stub's
``raise NotImplementedError`` blames the stub, a crashing helper called by
``main()`` blames the helper, a plain assert failure blames the checked
method itself.

Checks are deterministic only: trusted example asserts, plus a smoke call
for zero-argument functions (``main()``-style entries). Model-written
advisory asserts are never run here — they stay advisory (E6). A check
that dies on ``EOFError``/``KeyboardInterrupt`` is *skipped*, not failed:
an ``input()``-driven function is interactive, not broken.
"""
import ast
import re
from dataclasses import dataclass

from threetoks.code import gates

KIND_EXAMPLE = "example"
KIND_CALL = "call"
_STDIN_LINE = re.compile(r'File "<stdin>", line (\d+)')
_INCONCLUSIVE_TYPES = ("EOFError", "KeyboardInterrupt")


@dataclass(frozen=True)
class Check:
    """One executable statement tied to the method it exercises."""
    name: str
    statement: str
    kind: str


@dataclass(frozen=True)
class CheckResult:
    """A check's outcome: pass, skip (inconclusive), or fail with blame."""
    check: Check
    passed: bool
    skipped: bool = False
    error: str = ""
    blamed: str = ""


def collect_checks(methods) -> list[Check]:
    """Trusted examples plus zero-arg smoke calls, for bodied methods only.

    Stubs are excluded as check *subjects* (nothing to verify) but stay in
    the blame map — a stub that a check trips over gets blamed via its
    ``raise NotImplementedError`` frame.
    """
    checks = []
    for method in methods:
        if method.body is None:
            continue
        if method.examples:
            checks.extend(Check(method.name, f"assert {example}",
                                KIND_EXAMPLE) for example in method.examples)
        elif not method.args.strip():
            checks.append(Check(method.name, f"{method.name}()", KIND_CALL))
    return checks


def method_spans(module_source: str) -> dict[str, tuple[int, int]]:
    """Each top-level function's name -> (first line, last line), 1-based."""
    try:
        tree = ast.parse(module_source)
    except (SyntaxError, ValueError):
        return {}
    defs = (ast.FunctionDef, ast.AsyncFunctionDef)
    return {node.name: (node.lineno, node.end_lineno)
            for node in tree.body if isinstance(node, defs)}


def blame(stderr: str, spans: dict[str, tuple[int, int]],
          default: str) -> str:
    """The method whose code the failure's deepest in-module frame is in.

    Traceback frames print outermost-first, so scanning the ``<stdin>``
    line numbers in reverse finds the deepest frame; the first one inside
    a function span names the culprit. A failure with no in-span frame
    (the appended check statement itself) blames ``default``.
    """
    for line_number in reversed(_STDIN_LINE.findall(stderr)):
        line = int(line_number)
        for name, (start, end) in spans.items():
            if start <= line <= end:
                return name
    return default


def is_inconclusive(stderr: str) -> bool:
    """True when the process died on EOF/interrupt rather than a bug.

    Only the traceback's *final* line — the raised exception's own type —
    counts. A substring match over the whole stderr would let any error
    merely mentioning "EOFError" in its message masquerade as interactive
    code and ship a crashing module as a clean run.
    """
    lines = [line for line in stderr.splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    return any(last == kind or last.startswith(kind + ":")
               for kind in _INCONCLUSIVE_TYPES)


def run_checks(module_source: str, checks: list[Check]) -> list[CheckResult]:
    """Execute each check in its own subprocess against the module."""
    spans = method_spans(module_source)
    return [_run_one(module_source, check, spans) for check in checks]


def _run_one(module_source: str, check: Check,
             spans: dict[str, tuple[int, int]]) -> CheckResult:
    """One subprocess run; failures carry a first-error line and a blame."""
    ok, stderr = gates.execute(module_source + "\n" + check.statement + "\n")
    if ok:
        return CheckResult(check, passed=True)
    if is_inconclusive(stderr):
        return CheckResult(check, passed=False, skipped=True)
    return CheckResult(check, passed=False, error=gates.first_error(stderr),
                       blamed=blame(stderr, spans, check.name))


def failures(results: list[CheckResult]) -> list[CheckResult]:
    """The results that are real failures (skips are inconclusive)."""
    return [r for r in results if not r.passed and not r.skipped]


if __name__ == "__main__":
    module = ('"demo"\n\n\ndef helper(n):\n    return n // 0\n\n\n'
              "def main():\n    return helper(4)\n\n\n"
              "def add(a, b):\n    return a + b\n")
    spans = method_spans(module)
    assert set(spans) == {"helper", "main", "add"}, spans

    class _M:  # minimal MethodRecord stand-in
        def __init__(self, name, args, body, examples=()):
            self.name, self.args, self.body, self.examples = \
                name, args, body, list(examples)

    checks = collect_checks([_M("helper", "n", "x"), _M("main", "", "x"),
                             _M("add", "a, b", "x",
                                ["add(1, 2) == 3", "add(0, 0) == 0"]),
                             _M("ghost", "", None)])
    kinds = [(c.name, c.kind) for c in checks]
    assert kinds == [("main", KIND_CALL), ("add", KIND_EXAMPLE),
                     ("add", KIND_EXAMPLE)], kinds   # one check per anchor

    results = run_checks(module, checks)
    by_name = {r.check.name: r for r in results}
    assert not by_name["main"].passed and by_name["main"].blamed == "helper", \
        by_name["main"]                     # deepest frame is inside helper
    assert by_name["add"].passed, by_name["add"]
    assert failures(results) == [by_name["main"]]

    stub = module + "\n\ndef stubbed():\n    raise NotImplementedError\n"
    caller = stub + "\n\ndef entry():\n    return stubbed()\n"
    [hit] = run_checks(caller, [Check("entry", "entry()", KIND_CALL)])
    assert hit.blamed == "stubbed", hit     # runtime stub call is blamed

    wrong = 'def add(a, b):\n    return a - b\n'
    [bad] = run_checks(wrong, [Check("add", "assert add(1, 2) == 3",
                                     KIND_EXAMPLE)])
    assert not bad.passed and bad.blamed == "add", bad   # assert -> default

    interactive = 'def ask():\n    return input("? ")\n'
    [asked] = run_checks(interactive, [Check("ask", "ask()", KIND_CALL)])
    assert asked.skipped and not asked.passed, asked     # EOF is not a bug
    print("smoke OK")
