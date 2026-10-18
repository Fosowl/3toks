"""Shell mode of the code agent: one model-written command, run after a yes.

The code-mode router sends a request here when the user wants something
done in a terminal: install a package, run the tests, use git, list or
move files. The model writes ONE command from the request and a short
view of the project folder. The user sees the command and says yes or
no (``services.confirm_command``); only a yes runs it, in the project
folder, with the user's own shell and environment. Nothing is sandboxed.
"""
import os
import platform
import shlex
from pathlib import Path

from threetoks.code import execute, sandbox
from threetoks.nodes import ShortTextNode
from threetoks.render import Episode
from threetoks.trace import span

SHELL_PREFIX = """You write ONE shell command that does what the user asks.
The command runs in the project FOLDER shown below, on the user's computer.
Write PYTHON for the project's python: PYTHON main.py, PYTHON -m pytest,
PYTHON -m pip install NAME. The real path replaces PYTHON before it runs.
Write only the command, on one line: no explanation, no backticks."""
PYTHON_WORD = "PYTHON"                  # the model's name for the python
PYTHON_NAMES = ("python", "python3")    # a first word that means it too
PIP_NAMES = ("pip", "pip3")             # a first word that means its pip
COMMAND_QUESTION = "Which shell command does the request need?"
COMMAND_PREFILL = "COMMAND:"
COMMAND_MAX_TOKENS = 48
LISTED_ENTRIES = 40         # folder entries shown to the model
NO_COMMAND_ANSWER = "(no command: the model did not write a shell command)"
NO_CONFIRM_REASON = "no way to ask you for confirmation here"
DECLINED_REASON = "you declined it"


def run_shell(task: str, services, policy) -> dict:
    """Write one command for the request, ask the user, and run it.

    The code agent calls this for a request in shell mode, with the
    task, the shared services and the live policy. The run is one
    "run the command" span, so the progress tree shows its exit code.

    Args:
        task (str): the user's request, from the code agent.
        services (Services): files_root and confirm_command, from the REPL.
        policy (Policy): the live decision policy, from the REPL.

    Returns:
        dict: "answer" (the run report, or why nothing ran), "command"
        (str) and "ran" (bool); "judged_good" is False unless the
        command ran and exited with 0.
    """
    root = Path(services.files_root)
    command = propose_command(task, root, policy)
    if not command:
        return {"answer": NO_COMMAND_ANSWER, "command": "", "ran": False,
                "judged_good": False}
    refused = _refusal(services, command)
    if refused:
        return {"answer": f"(not run: {refused})\n$ {command}",
                "command": command, "ran": False, "judged_good": False}
    with span(getattr(policy, "tracer", None), "run the command") as ran:
        execution = execute.run_command(command, root)
        ran["result"] = execute.status(execution)
    result = {"answer": execute.report(execution), "command": command,
              "ran": True}
    if not execution.ok:
        result["judged_good"] = False
    return result


def propose_command(task: str, root: Path, policy) -> str:
    """Ask the model for one shell command that does the request.

    run_shell calls this with the request and the project folder. The
    model sees the system, the folder, the project python and the folder
    entries, and writes one line after "COMMAND:".

    Args:
        task (str): the user's request.
        root (Path): the project folder, from services.files_root.
        policy (Policy): the live decision policy.

    Returns:
        str: the command without backticks or a "$ " prompt, with the
        project python in place of PYTHON; "" when the model gave no
        valid line.
    """
    episode = Episode(SHELL_PREFIX, task)
    episode.open_observation(folder_context(root))
    node = ShortTextNode(COMMAND_QUESTION, COMMAND_PREFILL,
                         max_tokens=COMMAND_MAX_TOKENS)
    node.tag = "command"
    decision = policy.decide(episode, node)
    if not decision.valid:
        return ""
    command = str(decision.value).strip().strip("`").strip()
    return with_project_python(command.removeprefix("$ ").strip(), root)


def with_project_python(command: str, root: Path) -> str:
    """Put the project python where the command names a python or pip.

    propose_command calls this on the model's line, so the command the
    user confirms is the command that runs. Every PYTHON word becomes
    the project python; a first word python/python3 does too, and a
    first word pip/pip3 becomes "<python> -m pip".

    Args:
        command (str): the cleaned command line from the model.
        root (Path): the project folder, from services.files_root.

    Returns:
        str: the command with the interpreter path quoted for the shell.
    """
    python = shlex.quote(sandbox.project_python(root))
    words = command.split(" ")
    if words[0] in PYTHON_NAMES:
        words[0] = python
    elif words[0] in PIP_NAMES:
        words[0] = f"{python} -m pip"
    return " ".join(python if word == PYTHON_WORD else word for word in words)


def folder_context(root: Path) -> str:
    """Describe where the command will run, for the command prompt.

    propose_command shows this to the model, so a command like "run
    main.py" uses the project python and real file names.

    Args:
        root (Path): the project folder.

    Returns:
        str: SYSTEM, FOLDER, PYTHON and FILES lines.
    """
    shell_name = os.environ.get("SHELL", "sh")
    return "\n".join([
        f"SYSTEM: {platform.system()}, shell {shell_name}",
        f"FOLDER: {root.resolve()}",
        f"PYTHON: {sandbox.project_python(root)}",
        f"FILES: {_listing(root)}"])


def _listing(root: Path) -> str:
    """Name the visible entries of the folder; folders end with "/".

    Args:
        root (Path): the project folder.

    Returns:
        str: comma-separated names, with a "+N more" tail when cut.
    """
    try:
        entries = sorted(path for path in root.iterdir()
                         if not path.name.startswith("."))
    except OSError:
        return "(cannot list the folder)"
    names = [path.name + ("/" if path.is_dir() else "")
             for path in entries[:LISTED_ENTRIES]]
    more = len(entries) - LISTED_ENTRIES
    return ", ".join(names + ([f"+{more} more"] if more > 0 else [])) \
        or "(empty)"


def _refusal(services, command: str) -> str:
    """Ask the user about the command; give why it must not run, or "".

    Args:
        services (Services): its confirm_command callable, or None.
        command (str): the command the model wrote.

    Returns:
        str: "" when the user said yes, else the reason it does not run.
    """
    confirm = getattr(services, "confirm_command", None)
    if confirm is None:
        return NO_CONFIRM_REASON
    return "" if confirm(command) else DECLINED_REASON


if __name__ == "__main__":
    import tempfile
    from types import SimpleNamespace

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    class _Backend:
        def complete(self, model, raw_prompt, opts):
            return GenResult(" `ls`", 5, 2, 0.0, "stop")

    policy = Policy(_Backend(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "data.csv").write_text("a\n")
        asked: list[str] = []
        services = SimpleNamespace(
            files_root=Path(folder),
            confirm_command=lambda cmd: asked.append(cmd) or True)
        outcome = run_shell("list the files", services, policy)
        assert asked == ["ls"] and outcome["ran"], outcome
        assert "data.csv" in outcome["answer"], outcome
        denied = run_shell("list the files",
                           SimpleNamespace(files_root=Path(folder)), policy)
        assert not denied["ran"] and NO_CONFIRM_REASON in denied["answer"]
    print("smoke OK")
