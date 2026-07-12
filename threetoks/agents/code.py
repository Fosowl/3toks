"""Code agent: route a coding request to one of four modes and run it.

The front door is the E7-measured mode router (threetoks/code/route.py):
free pre-checks, then one one-token menu. The modes map to machinery that
already exists —

    navigate  free symbol-index lookup, answer is verbatim source (zero
              model calls when the name is unique)
    edit      the edit vertical over ``services.files_root``
    compute   the greenfield vertical writes a script, the harness runs
              it, and the printed output IS the answer
    author    the greenfield vertical, module delivered as the answer
"""
import re
import subprocess

from threetoks.agents.base import AgentSpec
from threetoks.code import route
from threetoks.code.edit import SymbolIndex
from threetoks.code.edit.vertical import EditVertical
from threetoks.code.vertical import CodeVertical
from threetoks.engine import run_episode
from threetoks.services import Services

AGENT_NAME = "code"
AGENT_DESCRIPTION = "write or debug Python code: a function, script, or program"
MAX_STEPS = 40
COMPUTE_TIMEOUT_S = 10
MAX_NAVIGATE_HITS = 3
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def run(task: str, services: Services, policy) -> dict:
    """Route to a mode, run it, and tag the result."""
    mode, how = route.route_mode(task, policy, services.files_root)
    handler = {route.NAVIGATE: _navigate, route.EDIT: _edit,
               route.COMPUTE: _compute, route.AUTHOR: _author}[mode]
    result = handler(task, services, policy)
    result.update(agent=AGENT_NAME, mode=mode, routed_by=how)
    return result


def _author(task: str, services: Services, policy) -> dict:
    """Write a module for the task; the module is the answer.

    Self-tests are off: E6 showed model-written asserts don't improve
    whole-module correctness on a 1.5b and only add latency; the
    deterministic gates carry the quality.
    """
    vertical = CodeVertical(task, gen_tests=False)
    return run_episode(vertical, policy, max_steps=MAX_STEPS)


def _compute(task: str, services: Services, policy) -> dict:
    """Write a script, run it for free, and answer with its output."""
    result = _author(task, services, policy)
    script = result.get("answer") or ""
    output = _script_output(script)
    result["script"] = script
    if output is not None:
        result["answer"] = output
    else:
        result["answer"] = (f"(the computation did not produce output; "
                            f"here is the script)\n{script}")
    return result


def _script_output(script: str) -> str | None:
    """Run the script in a subprocess; its stdout, or None on failure."""
    if not script:
        return None
    try:
        done = subprocess.run(["python3", "-"], input=script, text=True,
                              capture_output=True,
                              timeout=COMPUTE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    output = done.stdout.strip()
    return output if done.returncode == 0 and output else None


def _edit(task: str, services: Services, policy) -> dict:
    """Run the edit vertical against the files root (import-gated: there
    is no caller-supplied test in the agent path, so ``verified`` stays
    False and the answer says so honestly)."""
    vertical = EditVertical(task, services.files_root, test_code=None)
    result = run_episode(vertical, policy, max_steps=MAX_STEPS)
    if result["success"]:
        source = _def_source(services, result["target_file"],
                             result["target"])
        result["answer"] = (f"edited {result['target']} in "
                            f"{result['target_file']} — {result['reason']}\n"
                            f"{source}")
    else:
        result["answer"] = f"(no edit made: {result['reason']})"
    return result


def _navigate(task: str, services: Services, policy) -> dict:
    """Answer a where-is/what-does question extractively, zero model calls.

    The request's own words are matched against the symbol index; hits are
    answered with verbatim source and location. No hit is an honest miss.
    """
    index = SymbolIndex(services.files_root)
    tokens = {t.lower() for t in _IDENT.findall(task)}
    hits = [s for s in index.symbols if s.name.lower() in tokens]
    if not hits:
        return {"answer": "(no matching function, class, or method found "
                          f"under {services.files_root})", "hits": 0}
    shown = []
    for symbol in hits[:MAX_NAVIGATE_HITS]:
        source = index.source_of(symbol.file)
        lines = source.splitlines()[symbol.lineno - 1:symbol.end_lineno]
        shown.append(f"{symbol.qualname} — {symbol.file}:{symbol.lineno}\n"
                     + "\n".join(lines))
    return {"answer": "\n\n".join(shown), "hits": len(hits)}


def _def_source(services: Services, rel_file: str, qualname: str) -> str:
    """The edited def's current text, re-read from disk."""
    index = SymbolIndex(services.files_root)
    symbol = index.symbol_at(rel_file, qualname)
    if symbol is None:
        return ""
    lines = index.source_of(rel_file).splitlines()
    return "\n".join(lines[symbol.lineno - 1:symbol.end_lineno])


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    class _Backend:
        def __init__(self, texts):
            self.texts = list(texts)

        def complete(self, model, raw_prompt, opts):
            return GenResult(self.texts.pop(0) if self.texts else "return None",
                             8, 4, 0.0, "stop")

    # author path (pre-routed free by "write ... function" phrasing)
    scripted = _Backend(["square(n): multiply n by itself",
                         "return n * n"])
    policy = Policy(scripted, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    outcome = SPEC.run("write a function that squares a number",
                       Services(), policy)
    assert outcome["agent"] == "code" and outcome["mode"] == "author"
    assert "return n * n" in outcome["answer"], outcome["answer"]

    # navigate path: extractive, zero model calls
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "mod.py").write_text("def greet(name):\n    return 'hi'\n")
        nav_policy = Policy(_Backend([]),
                            PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = SPEC.run("where is the greet function defined?",
                           Services(files_root=root), nav_policy)
        assert outcome["mode"] == "navigate", outcome
        assert "mod.py:1" in outcome["answer"], outcome["answer"]
    print("smoke OK")
