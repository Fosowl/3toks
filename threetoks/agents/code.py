"""Code agent: route a coding request to one of five modes and run it.

The front door is the E7-measured mode router (threetoks/code/route.py):
a free pre-check for read-only questions, else one one-token menu. The
modes map to machinery that already exists —

    navigate  symbol index + ranked shortlist; zero model calls for a
              unique name, one menu when several symbols fit
    edit      the edit vertical proposes a checked candidate; the user
              confirms the diff before the file is written
    compute   the greenfield vertical writes a script, the runner
              executes its entry, and the echoed result IS the answer
    author    the greenfield vertical writes a script with a ``main``
              entry; the script then runs once in the project folder and
              its output is shown under the code
    shell     the model writes one shell command; it runs in the project
              folder after the user says yes (threetoks/code/shell.py)
"""
import difflib
from pathlib import Path

from threetoks.agents.base import AgentSpec
from threetoks.code import execute, route, shell
from threetoks.code.edit import SymbolIndex, nav, rank
from threetoks.code.edit.encoding import decode_source, encode_like
from threetoks.code.edit.vertical import EditVertical
from threetoks.code.vertical import CodeVertical, RunSetup
from threetoks.engine import run_episode
from threetoks.nodes import ESCAPE, MENU_MAX_OPTIONS, MenuNode
from threetoks.render import Episode
from threetoks.services import Services
from threetoks.trace import span

AGENT_NAME = "code"
AGENT_DESCRIPTION = ("write, run or fix Python code, run shell commands, "
                     "or explain the code in this project")
MAX_STEPS = 40
MAX_ANSWER_LINES = 40         # navigate answers are clipped here
NAVIGATE_KINDS = ("function", "method", "class")
NAVIGATE_MENU_SIZE = MENU_MAX_OPTIONS - 1     # one slot for the escape
NAVIGATE_QUESTION = "Which code symbol is the request about?"
NAVIGATE_PREFIX = """You help a user find code in their own project.
The user asks a question about code that already exists. You see a list
of code symbols (functions, methods, classes) with their file and line.
Pick the symbol that the question is about. If no symbol fits, pick the
"none of these fit" option."""
MENU_FALLBACK_NOTE = ("(could not classify this request: the model gave no "
                      "valid mode answer, so it was answered as a "
                      "navigate question)")
TRUNCATED_NOTE = ("(the project is large: only part of it was indexed, so "
                  "some code was not searched)")
NO_CONFIRM_REASON = "no way to ask you for confirmation here"
DECLINED_REASON = "you declined it"
CHANGED_REASON = "the file changed since it was read"
UNDECODABLE_REASON = ("the file on disk can no longer be decoded as "
                      "Python source")
NO_OUTPUT_REASON = "the script printed nothing"
NO_ENTRY_NOTE = ("(not run: the script has no entry function that the "
                 "__main__ guard can call)")
UNWRITTEN_REASON = "these functions could not be written: {names}"
INCOMPLETE_NOTE = ("(INCOMPLETE: {count} of {total} functions could not be "
                   "written and still raise NotImplementedError: {names})")


def run(task: str, services: Services, policy) -> dict:
    """Run one coding request in the mode the router chooses.

    The top-level agent router uses this as the code agent's entry point.
    The REPL calls it with the user task, the shared Services and the
    live policy.
    It routes the task to a mode, runs that mode and tags the result. A
    menu fallback gets a note at the top of the answer.

    Args:
        task (str): the user's coding request, from the REPL.
        services (Services): shared services, from the REPL state.
        policy (Policy): the live decision policy, from the REPL state.

    Returns:
        dict: the mode result plus "agent", "mode" and "routed_by" keys.
    """
    mode, how = route.route_mode(task, policy)
    handler = {route.NAVIGATE: _navigate, route.EDIT: _edit,
               route.COMPUTE: _compute, route.AUTHOR: _author,
               route.SHELL: shell.run_shell}[mode]
    result = handler(task, services, policy)
    if how == "menu-fallback":
        answer = result.get("answer") or ""
        result["answer"] = f"{MENU_FALLBACK_NOTE}\n{answer}"
    result.update(agent=AGENT_NAME, mode=mode, routed_by=how)
    return result


def _author(task: str, services: Services, policy) -> dict:
    """Write a new script for the task, run it once, and show both.

    The author mode uses this when the user wants new code delivered.
    run calls it with the task, the services and the policy.
    It runs the greenfield vertical without self-tests: E6 showed that
    model-written asserts do not improve whole-module correctness on a
    1.5b model; the deterministic gates carry the quality. The checks run
    in a temp folder under the project's python, and the plan always gets
    a ``main`` entry. Then the delivered script runs once in the project
    folder, so its files land there, and the run report follows the code.
    Stubbed functions are named in a note at the top of the answer, so
    unfinished code is never delivered without a warning.

    Args:
        task (str): the user's coding request, from run.
        services (Services): shared services (retriever, files root),
            from run.
        policy (Policy): the live decision policy, from run.

    Returns:
        dict: the CodeVertical result; "answer" is the script plus the
        run report, "script" the script, "execution" the run or None.
        A failed run sets "judged_good" to False.
    """
    root = Path(services.files_root)
    vertical = CodeVertical(task, gen_tests=False,
                            retriever=services.retriever,
                            run=RunSetup(project=root, add_main=True))
    result = run_episode(vertical, policy, max_steps=MAX_STEPS)
    result["script"] = result["answer"]
    result["execution"] = _run_delivered(result, root, policy)
    result["answer"] = _author_answer(result)
    result["output"] = ""  # temp-folder check output; the report replaces it
    if result["execution"] is not None and not result["execution"].ok:
        result["judged_good"] = False
    return result


def _run_delivered(result: dict, root: Path, policy):
    """Run the delivered script once, inside a "run the script" span.

    _author calls this after the vertical. A script without an entry has
    no ``__main__`` guard, so running it would do nothing: it is skipped.

    Args:
        result (dict): the CodeVertical result with "script" and "entry".
        root (Path): the project folder, from services.files_root.
        policy (Policy): the live policy; its tracer gets the span.

    Returns:
        execute.Execution | None: the run, or None when skipped.
    """
    if result.get("entry") is None:
        return None
    with span(getattr(policy, "tracer", None), "run the script") as ran:
        execution = execute.run_script(result["script"], root)
        ran["result"] = execute.status(execution)
    return execution


def _author_answer(result: dict) -> str:
    """Join the stub note, the script and the run report into the answer.

    Args:
        result (dict): the author result with "script", "stubbed",
            "methods" and "execution", from _author.

    Returns:
        str: the answer text for the panel.
    """
    parts = [result["script"]]
    stubbed = result.get("stubbed") or []
    if stubbed:
        parts.insert(0, INCOMPLETE_NOTE.format(
            count=len(stubbed), total=len(result["methods"]),
            names=", ".join(stubbed)))
    execution = result["execution"]
    parts.append(NO_ENTRY_NOTE if execution is None
                 else execute.report(execution))
    return "\n".join(parts)


def _compute(task: str, services: Services, policy) -> dict:
    """Write a script, run its entry once, and answer with its output.

    The compute mode uses this when the user wants a fact, not code.
    run calls it with the task, the services and the policy.
    The vertical runs the entry in the files root and prints its return
    value, so the passed entry check's stdout is the answer. There is no
    second run of the script.

    Args:
        task (str): the user's request, from run.
        services (Services): shared services (files root), from run.
        policy (Policy): the live decision policy, from run.

    Returns:
        dict: the CodeVertical result; "answer" is the output, "script"
        the delivered script.
    """
    root = Path(services.files_root)
    vertical = CodeVertical(task, gen_tests=False,
                            retriever=services.retriever,
                            run=RunSetup(cwd=root, project=root,
                                         echo_result=True))
    result = run_episode(vertical, policy, max_steps=MAX_STEPS)
    result["script"] = result.get("answer") or ""
    output = (result.get("output") or "").strip()
    result["answer"] = output or _compute_fallback(result)
    return result


def _compute_fallback(result: dict) -> str:
    """Explain a computation that gave no output, and show the script.

    The compute mode uses this so the user sees why there is no answer.
    _compute calls it with the vertical result after the script was set.
    It takes the first run failure error; else the names of the functions
    that could not be written; else a "printed nothing" reason.

    Args:
        result (dict): the CodeVertical result with "script" and
            "stubbed", from _compute.

    Returns:
        str: the fallback answer text.
    """
    failures = (result.get("run") or {}).get("failures") or []
    stubbed = result.get("stubbed") or []
    error = NO_OUTPUT_REASON
    if stubbed:
        error = UNWRITTEN_REASON.format(names=", ".join(stubbed))
    if failures:
        error = failures[0]["error"]
    return (f"(the computation did not produce output: {error}; "
            f"here is the script)\n{result['script']}")


def _edit(task: str, services: Services, policy) -> dict:
    """Propose an edit, show its diff, and apply it only when confirmed.

    The edit mode uses this to change existing code safely.
    run calls it with the task, the services and the policy.
    The vertical checks a candidate without leaving it on disk; this
    function builds the diff, asks services.confirm_edit, and writes the
    file only after a yes and an unchanged-file check.

    Args:
        task (str): the user's edit request, from run.
        services (Services): files root and confirm_edit, from run.
        policy (Policy): the live decision policy, from run.

    Returns:
        dict: the EditVertical result plus "applied", "diff" and "answer".
    """
    vertical = EditVertical(task, services.files_root, test_code=None)
    result = run_episode(vertical, policy, max_steps=MAX_STEPS)
    result["applied"] = False
    if not result["success"]:
        result["answer"] = f"(no edit made: {result['reason']})"
        if result.get("index_truncated"):
            result["answer"] += f"\n{TRUNCATED_NOTE}"
        return result
    result["diff"] = _unified_diff(result)
    result["applied"], why = _apply_edit(services, result)
    result["answer"] = _edit_answer(result, why)
    return result


def _unified_diff(result: dict) -> str:
    """Make the unified diff from the original file to the candidate.

    The user reads this diff before an edit is applied.
    _edit calls it with a successful EditVertical result.
    It compares the LF texts line by line with a/<rel> and b/<rel> labels.

    Args:
        result (dict): EditVertical result with "target_file",
            "original" and "candidate", from _edit.

    Returns:
        str: the diff text, lines joined with "\\n".
    """
    rel = result["target_file"]
    lines = difflib.unified_diff(result["original"].splitlines(),
                                 result["candidate"].splitlines(),
                                 f"a/{rel}", f"b/{rel}", lineterm="")
    return "\n".join(lines)


def _apply_edit(services: Services, result: dict) -> tuple[bool, str]:
    """Ask the user about the diff and write the file after a yes.

    The edit mode uses this so that no file changes without consent.
    _edit calls it with the services and the result that has "diff".
    It returns "not applied" when there is no confirm callback or the user
    declines; else it writes through _write_if_unchanged.

    Args:
        services (Services): files root and confirm_edit, from _edit.
        result (dict): EditVertical result with "diff", from _edit.

    Returns:
        tuple[bool, str]: (applied, why not; "" when applied).
    """
    rel = result["target_file"]
    if services.confirm_edit is None:
        return False, NO_CONFIRM_REASON
    if not services.confirm_edit(rel, result["diff"]):
        return False, DECLINED_REASON
    return _write_if_unchanged(Path(services.files_root) / rel, result)


def _write_if_unchanged(path: Path, result: dict) -> tuple[bool, str]:
    """Write the candidate only when the file is still the indexed text.

    The edit mode uses this to never overwrite a change made meanwhile.
    _apply_edit calls it with the target path and the vertical result.
    It re-reads the bytes, decodes them with decode_source, compares
    the LF text to result["original"], and writes the candidate with the
    original line endings and encoding.

    Args:
        path (Path): absolute path of the target file, from _apply_edit.
        result (dict): EditVertical result with "original" and
            "candidate", from _apply_edit.

    Returns:
        tuple[bool, str]: (written, why not; "" when written).
    """
    try:
        original_bytes = path.read_bytes()
    except OSError as error:
        return False, f"could not read the file: {error}"
    current = decode_source(original_bytes)
    if current is None:
        return False, UNDECODABLE_REASON
    if current != result["original"]:
        return False, CHANGED_REASON
    try:
        path.write_bytes(encode_like(original_bytes, result["candidate"]))
    except (OSError, UnicodeEncodeError) as error:
        return False, f"could not write the file: {error}"
    return True, ""


def _edit_answer(result: dict, why: str) -> str:
    """Write the edit answer: applied or not, verification, and the diff.

    The user reads this answer in the result panel.
    _edit calls it after the apply step.
    It gives one status line, one verification line, the edited symbol
    and the diff.

    Args:
        result (dict): EditVertical result with "applied" and "diff",
            from _edit.
        why (str): why the edit was not applied ("" when applied), from
            _apply_edit.

    Returns:
        str: the answer text.
    """
    rel = result["target_file"]
    symbol = result.get("edited_symbol") or result.get("target", "")
    status = (f"applied: {rel} was changed" if result["applied"]
              else f"not applied ({why}): {rel} is unchanged")
    return "\n".join([status, _verified_line(result),
                      f"edited symbol: {symbol}", result["diff"]])


def _verified_line(result: dict) -> str:
    """Say if the edit is verified and which repo tests changed state.

    The edit answer uses this line so the user knows how much to trust
    the change. _edit_answer calls it with the EditVertical result.
    It joins the verified flag, the vertical reason and the test lists.

    Args:
        result (dict): EditVertical result with "verified", "reason" and
            "tests", from _edit_answer.

    Returns:
        str: one verification line.
    """
    check = "verified" if result.get("verified") else "not verified"
    line = f"{check}: {result.get('reason', '')}"
    tests = result.get("tests") or {}
    if not tests:
        return f"{line} (no repo tests ran)"
    fixed = ", ".join(tests.get("fixed") or []) or "none"
    broken = ", ".join(tests.get("broken") or []) or "none"
    return f"{line}; tests fixed: {fixed}; tests broken: {broken}"


def _navigate(task: str, services: Services, policy) -> dict:
    """Answer a where-is/what-does question from the project source.

    The navigate mode uses this for read-only questions about code.
    run calls it with the task, the services and the policy.
    It finds one symbol (free, or with one menu) and answers with its
    location and clipped source, or with an honest miss.

    Args:
        task (str): the user's question, from run.
        services (Services): shared services (files root), from run.
        policy (Policy): the live decision policy, from run.

    Returns:
        dict: "answer", plus "symbol" and "file" ("" on a miss).
    """
    index = SymbolIndex(services.files_root)
    symbol = _pick_symbol(index, task, policy)
    if symbol is None:
        answer = ("(no matching function, class, or method found under "
                  f"{services.files_root})")
    else:
        answer = _describe(index, symbol)
    if index.truncated:
        answer = f"{answer}\n{TRUNCATED_NOTE}"
    return {"answer": answer, "symbol": symbol.qualname if symbol else "",
            "file": symbol.file if symbol else ""}


def _pick_symbol(index: SymbolIndex, task: str, policy):
    """Find the symbol that a navigate question is about.

    The navigate mode uses this to spend a model call only on doubt.
    _navigate calls it with the built index, the task and the policy.
    A unique exact name wins free; else the ranked shortlist gives no
    symbol, one symbol, or one menu over several.

    Args:
        index (SymbolIndex): the project index, from _navigate.
        task (str): the user's question, from _navigate.
        policy (Policy): the live decision policy, from _navigate.

    Returns:
        Symbol | None: the chosen symbol, or None for a miss.
    """
    exact = rank.exact_hits(index, task, kinds=NAVIGATE_KINDS)
    if len(exact) == 1:
        return exact[0]
    candidates = rank.shortlist(index, task, limit=NAVIGATE_MENU_SIZE,
                                kinds=NAVIGATE_KINDS)
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    return _ask_symbol(task, candidates, policy)


def _ask_symbol(task: str, candidates: list, policy):
    """Ask the model once which candidate symbol the question means.

    The navigate mode uses this when several symbols fit the question.
    _pick_symbol calls it with two or more shortlisted symbols.
    It shows one menu with a "none of these fit" escape.

    Args:
        task (str): the user's question, from _pick_symbol.
        candidates (list[Symbol]): ranked symbols, from _pick_symbol.
        policy (Policy): the live decision policy, from _pick_symbol.

    Returns:
        Symbol | None: the chosen symbol, or None on escape or no answer.
    """
    labels = [nav.symbol_label(symbol, with_kind=True)
              for symbol in candidates]
    decision = policy.decide(Episode(NAVIGATE_PREFIX, task),
                             MenuNode(NAVIGATE_QUESTION, labels))
    if not decision.valid or decision.value == ESCAPE:
        return None
    return candidates[labels.index(decision.value)]


def _describe(index: SymbolIndex, symbol) -> str:
    """Write the navigate answer for one symbol.

    The user reads this answer in the result panel.
    _navigate calls it with the index and the chosen symbol.
    It gives "qualname — file:line" and then the clipped source; a class
    gets its class line, docstring and method names instead.

    Args:
        index (SymbolIndex): the project index, from _navigate.
        symbol (Symbol): the chosen symbol, from _navigate.

    Returns:
        str: the answer text.
    """
    source = index.source_of(symbol.file).split("\n")
    lines = source[symbol.lineno - 1:symbol.end_lineno]
    if symbol.kind == "class":
        lines = _class_outline(index, symbol, lines[:1])
    shown, hidden = nav.clip_lines(lines, MAX_ANSWER_LINES)
    if hidden:
        shown.append(f"… {hidden} more lines")
    header = f"{symbol.qualname} — {symbol.file}:{symbol.lineno}"
    return "\n".join([header] + shown)


def _class_outline(index: SymbolIndex, symbol, class_line: list) -> list:
    """List a class as its first line, its docstring and its methods.

    The navigate answer uses this so a big class stays short.
    _describe calls it with the index, the class symbol and its line.
    It takes the method names from index.methods_of in source order.

    Args:
        index (SymbolIndex): the project index, from _describe.
        symbol (Symbol): the class symbol, from _describe.
        class_line (list[str]): the class line (empty when unknown), from
            _describe.

    Returns:
        list[str]: the outline lines.
    """
    methods = index.methods_of(symbol.file, symbol.qualname)
    doc = [f'    """{symbol.doc}"""'] if symbol.doc else []
    return class_line + doc + ["methods: " + (", ".join(methods) or "none")]


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)


if __name__ == "__main__":
    import tempfile

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    class _MenuBackend:
        """Answers each menu with the option that names the keyword."""

        def __init__(self, keyword):
            self.keyword = keyword

        def complete(self, model, raw_prompt, opts):
            menu = raw_prompt[raw_prompt.rindex("ACTIONS:"):]
            for line in menu.splitlines():
                if line[:1].isdigit() and self.keyword in line:
                    return GenResult(f" {line.split(' = ')[0]}",
                                     5, 1, 0.0, "stop")
            return GenResult(" x", 5, 1, 0.0, "stop")

    def _policy(backend):
        return Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "mod.py").write_text("def greet(name):\n    return 'hi'\n\n"
                                     "def greeting():\n    return 'yo'\n")
        services = Services(files_root=root)
        # navigate: unique exact name, free route, zero model calls
        outcome = SPEC.run("where is the greet function defined?",
                           services, _policy(_MenuBackend("never")))
        assert outcome["mode"] == "navigate", outcome
        assert "mod.py:1" in outcome["answer"], outcome["answer"]
        # menu fallback: the note is the first answer line
        outcome = SPEC.run("hmm, greet stuff", services,
                           _policy(_MenuBackend("never")))
        assert outcome["answer"].startswith(MENU_FALLBACK_NOTE), outcome
        # navigate: long answers are clipped with a count
        body = "    x = 1\n" * (MAX_ANSWER_LINES + 1)
        (root / "long.py").write_text(f"def long_one():\n{body}")
        outcome = SPEC.run("where is the long_one function?",
                           Services(files_root=root),
                           _policy(_MenuBackend("never")))
        assert "… 2 more lines" in outcome["answer"], outcome["answer"]
    print("smoke OK")
