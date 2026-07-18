"""Files agent: exploration of local files and folders.

A Vertical mirroring the web-research one (threetoks/web/vertical.py):
the harness owns everything deterministic (listing, sandboxing, reading,
chunking, note storage, loop guards); the model only answers menus, picks
line numbers to note, writes the odd shell command, and phrases the final
answer from the notes.

State machine: DIR -> FILE -> (auto note pass) -> ... -> ANSWER, rooted at
services.files_root, which the agent never escapes. Browsing is read-only;
the one write-capable surface is the shell option, which runs only after a
deterministic deny-list AND a fresh-context one-digit safety judge both
clear the command.
"""
import re
import subprocess
from pathlib import Path

from threetoks.agents.base import AgentSpec
from threetoks.engine import run_episode
from threetoks.nodes import ESCAPE, MenuNode, PickManyNode, ShortTextNode
from threetoks.render import Episode
from threetoks.web.notes import NoteStore, rank_by_overlap

AGENT_NAME = "files"
AGENT_DESCRIPTION = ("browse files and folders on this machine — only "
                     "when the request names them")

MAX_ENTRIES_SHOWN = 15
MAX_OPEN_OPTIONS = 6
MAX_FILES_OPENED = 4
CHUNK_LINES = 25
MAX_FILE_BYTES = 200_000
NAME_CHARS = 60
ANSWER_MAX_TOKENS = 80
QUOTE_FALLBACK_NOTES = 2
FORCE_ANSWER_AT_STEPS_LEFT = 3
INITIAL_STEPS_LEFT = 40
MAX_STEPS = 20

MAX_SHELL_RUNS = 3
SHELL_TIMEOUT_S = 10
SHELL_LINE_CHARS = 200
COMMAND_MAX_TOKENS = 24

OPT_GO_UP = "go up to the parent folder"
OPT_ANSWER = "answer the task now"
OPT_BACK = "back to the folder"
OPT_NEXT_CHUNK = "read more of this file"
OPT_SHELL = "run a shell command to inspect files"
FINAL_PREFILL = "FINAL ANSWER: It"
CMD_PREFILL = "COMMAND:"
# A bare menu digit or comma-index list ("1" / "1,2,3") is menu-format
# bleed, not an answer; a real numeric answer like "1989" is fine.
_MENU_BLEED = re.compile(
    r"^\d{1,3}$|^\d{5,}$|^\d+(\s*,\s*\d+)+\s*,?$")

# Shell safety layer 1 (deterministic, free): no chaining, piping,
# redirection, or substitution; no deny-listed write/execute/network
# words anywhere in the command; no path that could leave the sandbox.
_SHELL_META_RE = re.compile(r"[;&|<>`]|\$\(")
_SHELL_DENY_WORDS = frozenset((
    "rm", "rmdir", "mv", "cp", "dd", "ln", "touch", "mkdir", "chmod",
    "chown", "chgrp", "kill", "pkill", "sudo", "su", "sh", "bash", "zsh",
    "python", "python3", "perl", "ruby", "node", "xargs", "eval", "exec",
    "curl", "wget", "ssh", "scp", "nc", "git", "make", "open", "tee"))

# Shell safety layer 2: a fresh-context model judge. Two semantically
# described options (never bare yes/no — design rule), one digit out.
SHELL_JUDGE_PREFIX = ("You judge whether shell commands are safe to run. "
                      "Only commands that read are safe.")
OPT_CMD_SAFE = ("safe — only reads: lists, views, counts, or searches "
                "files (like ls, cat, head, grep, wc, find)")
OPT_CMD_UNSAFE = ("unsafe — writes, deletes, moves, or changes anything, "
                  "runs programs, or touches the network")

BINARY_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".pdf",
    ".zip", ".gz", ".tar", ".7z", ".rar", ".pyc", ".bin", ".exe", ".dll",
    ".so", ".dylib", ".o", ".a", ".gguf", ".safetensors", ".mp3", ".mp4",
    ".mov", ".wav", ".class", ".jar", ".woff", ".woff2", ".ttf"})

PREFIX = """You explore local files by answering menus.
Rules: reply with exactly ONE digit for ACTIONS menus; reply with numbers
separated by commas when asked which lines to note.
Example:
ACTIONS:
1 = open: notes.txt
2 = go up to the parent folder
Reply with exactly ONE digit.
ANSWER: 1"""


def _is_text_file(path: Path) -> bool:
    """A readable, non-hidden, small, non-binary regular file."""
    if path.name.startswith(".") or path.suffix.lower() in BINARY_SUFFIXES:
        return False
    try:
        return path.is_file() and path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


def _openable_entries(directory: Path) -> list[Path]:
    """Visible dirs first then text files, sorted, capped for a menu."""
    try:
        children = list(directory.iterdir())
    except OSError:
        return []
    dirs = sorted(c for c in children
                  if c.is_dir() and not c.name.startswith("."))
    files = sorted(c for c in children if _is_text_file(c))
    return dirs + files


def _entry_label(path: Path) -> str:
    """One numbered-list line: name plus (dir) or a rounded size."""
    if path.is_dir():
        return f"{path.name}/ (dir)"
    kb = max(1, round(path.stat().st_size / 1024))
    return f"{path.name} ({kb} KB)"


def _file_lines(path: Path) -> list[str]:
    """Non-empty stripped lines of a text file (binary-safe decode)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def dangerous_command(command: str) -> bool:
    """Deterministic deny-list: block anything not plainly read-only.

    Free pre-check before the model judge (design rule: deterministic
    checks first). Rejects shell metacharacters, deny-listed words, and
    any token that could leave the sandbox root ("/...", "~", "..").
    """
    if _SHELL_META_RE.search(command):
        return True
    for token in command.split():
        if token.startswith(("/", "~")) or ".." in token:
            return True
    words = set(re.findall(r"[a-z0-9_]+", command.lower()))
    return bool(words & _SHELL_DENY_WORDS)


def command_is_safe(command: str, policy) -> bool:
    """Fresh-context one-digit safety judge; fails closed.

    A separate Episode sees ONLY the command — never the exploration
    context — so text read from files cannot lobby the judge. Anything
    but an explicit "safe" pick blocks the run.
    """
    episode = Episode(SHELL_JUDGE_PREFIX, f"$ {command}")
    node = MenuNode(f"Judge this command: $ {command}",
                    [OPT_CMD_SAFE, OPT_CMD_UNSAFE], escape=False)
    node.tag = "cmdsafe"
    decision = policy.decide(episode, node)
    return decision.valid and decision.value == OPT_CMD_SAFE


class FileExplorerVertical:
    """Drives one file-exploration episode under a fixed sandbox root.

    ``policy`` powers the fresh-context shell safety judge; without one
    (older callers, tests) the shell option simply never appears.
    """

    def __init__(self, task: str, files_root: Path, notes: NoteStore,
                 policy=None):
        self.task = task
        self.root = Path(files_root).resolve()
        self.cwd = self.root
        self.notes = notes
        self.policy = policy
        self.episode = Episode(PREFIX, task)
        self.steps_left = INITIAL_STEPS_LEFT
        self.files_opened = 0
        self.shell_runs = 0
        self.entries: list[Path] = []
        self.file_path: Path | None = None
        self.file_lines: list[str] = []
        self.chunk_start = 0
        self.note_source = ""
        self.pending = None
        self.answer = None
        self.answer_retried = False
        self._first_bleed = ""
        self._awaiting_curation = False
        self._option_map: dict[str, Path] = {}
        self._show_dir("> started at the root folder")

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
        return self._file_menu() if self.file_path else self._dir_menu()

    def _pending_is_answer(self) -> bool:
        """True when the queued node already belongs to the answer flow."""
        final_text = isinstance(self.pending, ShortTextNode) \
            and self.pending.prefill == FINAL_PREFILL
        return final_text or (self._awaiting_curation
                              and isinstance(self.pending, PickManyNode))

    def apply(self, node, decision):
        """Execute one decision; invalid decisions map to the escape path."""
        value = decision.value if decision.valid else ESCAPE
        if isinstance(node, PickManyNode) and self._awaiting_curation:
            self._apply_curation(decision.value if decision.valid else [])
        elif isinstance(node, PickManyNode):
            self._apply_notes(decision.value if decision.valid else [])
        elif isinstance(node, ShortTextNode) \
                and node.prefill == CMD_PREFILL:
            self._handle_command(decision.value if decision.valid else "")
        elif isinstance(node, ShortTextNode):
            self._accept_answer(decision.value if decision.valid else "")
        else:
            self._apply_menu(value)

    def result(self) -> dict:
        """Episode outcome with note provenance."""
        return {"task": self.task, "answer": self.answer,
                "notes": self.notes.render(),
                "files_opened": self.files_opened,
                "shell_runs": self.shell_runs}

    # ------------------------------------------------------------ menus

    def _dir_menu(self) -> MenuNode:
        options = list(self._build_open_options())
        if self.cwd != self.root:
            options.append(OPT_GO_UP)
        if self._shell_available():
            options.append(OPT_SHELL)
        if self.notes.count():
            options.append(OPT_ANSWER)
        return MenuNode("What next?", options or [OPT_ANSWER])

    def _shell_available(self) -> bool:
        """Shell needs a policy (for the safety judge) and budget left."""
        return self.policy is not None and self.shell_runs < MAX_SHELL_RUNS

    def _build_open_options(self):
        """Openable entries, mapped for opening; capped to fit the menu.

        The cap shrinks by one when the shell option is on the menu so
        the worst case (opens + go up + shell + answer + escape) stays
        within the 9-line menu limit.
        """
        self._option_map = {}
        cap = MAX_OPEN_OPTIONS - (1 if self._shell_available() else 0)
        if self.files_opened >= MAX_FILES_OPENED:
            self.entries = [e for e in self.entries if e.is_dir()]
        for entry in self.entries[:cap]:
            text = f"open: {entry.name[:NAME_CHARS]}"
            self._option_map[text] = entry
            yield text

    def _file_menu(self) -> MenuNode:
        options = []
        if self.chunk_start + CHUNK_LINES < len(self.file_lines):
            options.append(OPT_NEXT_CHUNK)
        options.append(OPT_BACK)
        if self.notes.count():
            options.append(OPT_ANSWER)
        return MenuNode("What next?", options)

    # ------------------------------------------------------------ answer flow

    def _answer_flow(self):
        """Curate the notes (once), then phrase the answer from them.

        Mirrors the web vertical's harvest→curate→synthesise shape:
        quoting raw notes verbatim answered "what are these files?" with
        naked code lines — a short generation grounded ONLY in the notes
        states what they mean, in plain words aimed at the task.
        """
        if self._needs_curation():
            return self._curate_node()
        return self._synthesis_node()

    def _needs_curation(self) -> bool:
        """True when the note store is big enough to be worth ranking."""
        from threetoks.research import CURATE_KEEP  # deferred: keeps import light
        return self.notes.count() > CURATE_KEEP

    def _curate_node(self) -> PickManyNode:
        """PICK_MANY that ranks notes to the best CURATE_KEEP, most first."""
        from threetoks.research import (  # deferred: keeps import light
            build_curate_node, curation_pool)
        self.notes = curation_pool(self.notes)
        self.episode.open_observation(
            f"NOTES COLLECTED:\n{self.notes.render()}",
            "> curating the strongest evidence")
        self.file_path = None
        self._awaiting_curation = True
        return build_curate_node(self.task, self.notes.texts())

    def _apply_curation(self, picked: list[int]) -> None:
        """Replace the store with the ranked selection, then synthesise."""
        from threetoks.research import CURATE_KEEP  # deferred: keeps import light
        self._awaiting_curation = False
        ranking = picked or list(range(1, CURATE_KEEP + 1))
        self.notes = self.notes.select(ranking)
        self.pending = self._synthesis_node()

    def _synthesis_node(self) -> ShortTextNode:
        """Generate the answer from the (curated) notes, or "(no answer)".

        Notes are shown WITHOUT their "(source: ...)" tags — a live run
        showed the 1.5b copying the tags into the answer text.
        """
        body = "\n".join(f"[{i}] {t}"
                         for i, t in enumerate(self.notes.texts(), 1)) \
            or "(none)"
        self.episode.open_observation(f"NOTES COLLECTED:\n{body}",
                                      "> moving to final answer")
        self.file_path = None
        return ShortTextNode(
            f"Using ONLY the notes above, write one or two short plain "
            f"sentences that answer: {self.task}. Reply with words.",
            FINAL_PREFILL, max_tokens=ANSWER_MAX_TOKENS)

    def _accept_answer(self, text: str) -> None:
        """Accept a synthesised answer, retrying once on menu-format bleed.

        On a second bleed the generation is abandoned: with notes the
        answer falls back to quoting the notes closest to the task (still
        real evidence), otherwise "(no answer)". A persistently repeated
        single digit is accepted — it may be a real numeric answer.
        """
        bleed = bool(text) and bool(_MENU_BLEED.fullmatch(text.strip()))
        if bleed and self.answer_retried:
            repeated_digit = (text.strip() == self._first_bleed
                              and len(text.strip()) == 1)
            self.answer = text.strip() if repeated_digit \
                else self._quote_fallback()
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
        """Quote the notes closest to the task when synthesis keeps bleeding."""
        top = rank_by_overlap(self.notes.texts(), self.task,
                              QUOTE_FALLBACK_NOTES)
        return "\n".join(top) if top else "(no answer)"

    # ------------------------------------------------------------ transitions

    def _apply_menu(self, value: str) -> None:
        if value == OPT_ANSWER:
            self.pending = self._answer_flow()
        elif value == OPT_SHELL:
            self.pending = self._command_node()
        elif value == OPT_GO_UP or (value == ESCAPE and not self.file_path):
            self._go_up()
        elif value == OPT_NEXT_CHUNK:
            self.chunk_start += CHUNK_LINES
            self._show_file_chunk("> read further into the file")
            self._queue_note_pass()
        elif value == OPT_BACK or value == ESCAPE:
            self.file_path = None
            self._show_dir(f"> closed '{self._entry_name()}'")
        else:
            self._open_entry(value)

    def _apply_notes(self, picked: list[int]) -> None:
        chunk = self._chunk()
        for idx in picked:
            self.notes.add(chunk[idx - 1], self.note_source,
                           self.chunk_start + idx)
        self.episode.log_session(f"> noted {len(picked)} lines")
        self.pending = None

    def _open_entry(self, value: str) -> None:
        """Open the mapped entry: descend into dirs, read text files."""
        target = self._safe_target(self._option_map.get(value))
        if target is None:
            self._show_dir("> could not open that entry")
        elif target.is_dir():
            self.cwd = target
            self._show_dir(f"> opened folder '{target.name[:NAME_CHARS]}'")
        else:
            self._load_file(target)

    def _load_file(self, path: Path) -> None:
        try:
            lines = _file_lines(path)
        except OSError:
            self.episode.log_session(f"> failed to read '{path.name[:40]}'")
            self.pending = None
            return
        self.file_path, self.file_lines, self.chunk_start = path, lines, 0
        self.note_source = str(path)
        self.files_opened += 1
        self._show_file_chunk(f"> opened file '{path.name[:NAME_CHARS]}'")
        self._queue_note_pass()

    # ------------------------------------------------------------ shell

    def _command_node(self) -> ShortTextNode:
        """Bounded free-text proposal for one read-only shell command."""
        return ShortTextNode(
            "Write ONE read-only shell command (like ls, cat, grep, wc, "
            "find) that helps answer the task. No pipes or redirection.",
            CMD_PREFILL, max_tokens=COMMAND_MAX_TOKENS)

    def _handle_command(self, text: str) -> None:
        """Run a proposed command only past both safety layers.

        Deterministic deny-list first (free), then the fresh-context
        model judge; either rejection just logs. Every proposal burns one
        shell run, so a looping proposer cannot stall the episode.
        """
        self.shell_runs += 1
        command = text.strip().strip("`").strip()
        if not command or dangerous_command(command):
            self.episode.log_session("> command blocked (looked unsafe)")
            self.pending = None
            return
        if not command_is_safe(command, self.policy):
            self.episode.log_session("> command blocked by the safety judge")
            self.pending = None
            return
        self._run_command(command)

    def _run_command(self, command: str) -> None:
        """Execute an approved command in the sandbox root, show its output.

        Output lines are numbered and get the same note pass as a file
        chunk, so command results can be quoted with provenance.
        """
        try:
            proc = subprocess.run(command, shell=True, cwd=self.root,
                                  capture_output=True, text=True,
                                  errors="replace", timeout=SHELL_TIMEOUT_S)
            raw = proc.stdout or proc.stderr
        except (OSError, subprocess.SubprocessError):
            self.episode.log_session(f"> command failed: {command[:40]}")
            self.pending = None
            return
        lines = [line.strip()[:SHELL_LINE_CHARS]
                 for line in raw.splitlines() if line.strip()]
        self.file_path = None
        self.file_lines = lines[:CHUNK_LINES] or ["(no output)"]
        self.chunk_start = 0
        self.note_source = f"$ {command}"
        numbered = "\n".join(f"[{i}] {s}"
                             for i, s in enumerate(self.file_lines, 1))
        self.episode.open_observation(
            f"COMMAND OUTPUT of '$ {command[:NAME_CHARS]}':\n{numbered}",
            f"> ran command '{command[:NAME_CHARS]}'")
        self.pending = None
        self._queue_note_pass()

    def _go_up(self) -> None:
        """Ascend one folder, never above the sandbox root."""
        parent = self._safe_target(self.cwd.parent)
        self.cwd = parent if parent is not None and parent.is_dir() \
            else self.root
        self.file_path = None
        self._show_dir("> went up to the parent folder")

    def _queue_note_pass(self) -> None:
        """Every freshly shown chunk gets an immediate note-picking pass."""
        if self._chunk():
            self.pending = PickManyNode(
                "Which lines contain information needed to answer?",
                len(self._chunk()), items=self._chunk())

    # ------------------------------------------------------------ sandbox / IO

    def _safe_target(self, path) -> Path | None:
        """Resolve a candidate path and keep it inside the sandbox root."""
        if path is None:
            return None
        resolved = Path(path).resolve()
        if resolved == self.root or resolved.is_relative_to(self.root):
            return resolved
        return None

    def _show_dir(self, closing: str) -> None:
        self.entries = _openable_entries(self.cwd)
        shown = self.entries[:MAX_ENTRIES_SHOWN]
        lines = [f"[{i}] {_entry_label(e)}" for i, e in enumerate(shown, 1)]
        body = "\n".join(lines) or "(empty folder)"
        self.episode.open_observation(
            f"FOLDER '{self._folder_name()}':\n{body}", closing)
        self.pending = None

    def _show_file_chunk(self, closing: str) -> None:
        chunk = self._chunk()
        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(chunk, 1))
        section = self.chunk_start // CHUNK_LINES + 1
        self.episode.open_observation(
            f"FILE '{self._entry_name()}' (section {section}):\n{numbered}",
            closing)
        self.pending = None

    def _chunk(self) -> list[str]:
        return self.file_lines[self.chunk_start:self.chunk_start + CHUNK_LINES]

    def _entry_name(self) -> str:
        return (self.file_path.name if self.file_path else "")[:NAME_CHARS]

    def _folder_name(self) -> str:
        name = "." if self.cwd == self.root else self.cwd.name
        return name[:NAME_CHARS]


def run(task: str, services, policy) -> dict:
    """Explore files under services.files_root and answer the task."""
    vertical = FileExplorerVertical(task, services.files_root, NoteStore(),
                                    policy=policy)
    result = run_episode(vertical, policy, max_steps=MAX_STEPS)
    result["agent"] = AGENT_NAME
    return result


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)


if __name__ == "__main__":
    import tempfile

    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig
    from threetoks.services import Services

    class _Backend:
        """Menus answered by option label (permutation-proof); rest scripted."""

        def __init__(self, script):
            self.script = list(script)

        def complete(self, model, raw_prompt, opts):
            kind, payload = self.script.pop(0)
            if kind == "menu":
                match = re.search(rf"(\d+) = {re.escape(payload)}",
                                  raw_prompt)
                return GenResult(f" {match.group(1)}" if match else " 9",
                                 5, 2, 0.0, "stop")
            return GenResult(f" {payload}", 5, 2, 0.0, "stop")

    assert dangerous_command("rm -rf .")
    assert dangerous_command("cat x.txt > y.txt")
    assert dangerous_command("cat ../secret")
    assert not dangerous_command("wc -l data.csv")

    with tempfile.TemporaryDirectory() as root:
        (Path(root) / "facts.txt").write_text(
            "The tower is 42 meters tall.\n", encoding="utf-8")
        script = [("menu", "open: facts.txt"), ("pick", "1"),
                  ("menu", OPT_ANSWER),
                  ("text", "The tower is 42 meters tall.")]
        policy = Policy(_Backend(script),
                        PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = SPEC.run("how tall is the tower?",
                           Services(files_root=Path(root)), policy)
        assert outcome["agent"] == "files", outcome
        assert "42 meters" in (outcome["answer"] or ""), outcome

        judge = Policy(_Backend([("menu", "safe")]),
                       PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        assert command_is_safe("wc -l facts.txt", judge)
        judge = Policy(_Backend([("menu", "unsafe")]),
                       PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        assert not command_is_safe("wc -l facts.txt", judge)
    print("smoke OK")
