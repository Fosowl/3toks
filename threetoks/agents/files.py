"""Files agent: read-only exploration of local files and folders.

A Vertical mirroring the web-research one (threetoks/web/vertical.py):
the harness owns everything deterministic (listing, sandboxing, reading,
chunking, note storage, loop guards); the model only answers menus, picks
line numbers to note, and points at the notes that answer.

State machine: DIR -> FILE -> (auto note pass) -> ... -> ANSWER, rooted at
services.files_root, which the agent never escapes.
"""
import re
from pathlib import Path

from threetoks.agents.base import AgentSpec
from threetoks.engine import run_episode
from threetoks.nodes import ESCAPE, MenuNode, PickManyNode, ShortTextNode
from threetoks.render import Episode
from threetoks.web.notes import NoteStore

AGENT_NAME = "files"
AGENT_DESCRIPTION = ("browse files and folders on this machine — only "
                     "when the request names them")

MAX_ENTRIES_SHOWN = 15
MAX_OPEN_OPTIONS = 6
MAX_FILES_OPENED = 4
CHUNK_LINES = 25
MAX_FILE_BYTES = 200_000
NAME_CHARS = 60
ANSWER_MAX_TOKENS = 48
ANSWER_PICK_POOL = 9
FORCE_ANSWER_AT_STEPS_LEFT = 3
INITIAL_STEPS_LEFT = 40
MAX_STEPS = 20

OPT_GO_UP = "go up to the parent folder"
OPT_ANSWER = "answer the task now"
OPT_BACK = "back to the folder"
OPT_NEXT_CHUNK = "read more of this file"
FINAL_PREFILL = "FINAL ANSWER:"
# A bare menu digit or comma-index list ("1" / "1,2,3") is menu-format
# bleed, not an answer; a real numeric answer like "42" is fine.
_MENU_BLEED = re.compile(r"\d|\d+(\s*,\s*\d+)+\s*,?")

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


class FileExplorerVertical:
    """Drives one read-only file-exploration episode under a fixed root."""

    def __init__(self, task: str, files_root: Path, notes: NoteStore):
        self.task = task
        self.root = Path(files_root).resolve()
        self.cwd = self.root
        self.notes = notes
        self.episode = Episode(PREFIX, task)
        self.steps_left = INITIAL_STEPS_LEFT
        self.files_opened = 0
        self.entries: list[Path] = []
        self.file_path: Path | None = None
        self.file_lines: list[str] = []
        self.chunk_start = 0
        self.pending = None
        self.answer = None
        self.answer_retried = False
        self._awaiting_answer_pick = False
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
        return final_text or (self._awaiting_answer_pick
                              and isinstance(self.pending, PickManyNode))

    def apply(self, node, decision):
        """Execute one decision; invalid decisions map to the escape path."""
        value = decision.value if decision.valid else ESCAPE
        if isinstance(node, PickManyNode) and self._awaiting_answer_pick:
            self._apply_answer_pick(decision.value if decision.valid else [])
        elif isinstance(node, PickManyNode):
            self._apply_notes(decision.value if decision.valid else [])
        elif isinstance(node, ShortTextNode):
            self._accept_answer(decision.value if decision.valid else "")
        else:
            self._apply_menu(value)

    def result(self) -> dict:
        """Episode outcome with note provenance."""
        return {"task": self.task, "answer": self.answer,
                "notes": self.notes.render(),
                "files_opened": self.files_opened}

    # ------------------------------------------------------------ menus

    def _dir_menu(self) -> MenuNode:
        options = list(self._build_open_options())
        if self.cwd != self.root:
            options.append(OPT_GO_UP)
        if self.notes.count():
            options.append(OPT_ANSWER)
        return MenuNode("What next?", options or [OPT_ANSWER])

    def _build_open_options(self):
        """Up to MAX_OPEN_OPTIONS openable entries, mapped for opening."""
        self._option_map = {}
        if self.files_opened >= MAX_FILES_OPENED:
            self.entries = [e for e in self.entries if e.is_dir()]
        for entry in self.entries[:MAX_OPEN_OPTIONS]:
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
        """Answer extractively from notes; free-text only without notes."""
        if self.notes.count():
            self.episode.open_observation(
                f"NOTES COLLECTED:\n{self.notes.render()}",
                "> moving to final answer")
            self.file_path = None
            self._awaiting_answer_pick = True
            pool = self.notes.texts()[:ANSWER_PICK_POOL]
            return PickManyNode(
                f"Which notes directly answer this question: {self.task}",
                len(pool), max_picks=2, items=pool)
        return self._synthesis_node()

    def _synthesis_node(self) -> ShortTextNode:
        """Free-text fallback when there are no notes to quote."""
        self.episode.open_observation("NOTES COLLECTED:\n(none)",
                                      "> moving to final answer")
        self.file_path = None
        self._awaiting_answer_pick = False
        return ShortTextNode(
            f"Answer this question directly in one short sentence "
            f"(words, not a menu number): {self.task}",
            FINAL_PREFILL, max_tokens=ANSWER_MAX_TOKENS)

    def _apply_answer_pick(self, picked: list[int]) -> None:
        """Quote the chosen notes verbatim as the final answer."""
        self._awaiting_answer_pick = False
        texts = self.notes.texts()[:ANSWER_PICK_POOL]
        chosen = [texts[i - 1] for i in picked if 1 <= i <= len(texts)]
        if chosen:
            self.answer = "\n".join(chosen)
            return
        self.pending = self._synthesis_node()

    def _accept_answer(self, text: str) -> None:
        """Accept a free-text answer, retrying once on menu-format bleed."""
        stripped = text.strip()
        if stripped and _MENU_BLEED.fullmatch(stripped) \
                and not self.answer_retried:
            self.answer_retried = True
            self.pending = ShortTextNode(
                f"That was a number, not an answer. In plain words, "
                f"state the answer to: {self.task}",
                FINAL_PREFILL, max_tokens=ANSWER_MAX_TOKENS)
            return
        self.answer = text or "(no answer)"

    # ------------------------------------------------------------ transitions

    def _apply_menu(self, value: str) -> None:
        if value == OPT_ANSWER:
            self.pending = self._answer_flow()
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
            self.notes.add(chunk[idx - 1], str(self.file_path),
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
        self.files_opened += 1
        self._show_file_chunk(f"> opened file '{path.name[:NAME_CHARS]}'")
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
    vertical = FileExplorerVertical(task, services.files_root, NoteStore())
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
        def __init__(self, texts):
            self.texts = list(texts)

        def complete(self, model, raw_prompt, opts):
            return GenResult(self.texts.pop(0) if self.texts else " 9",
                             5, 2, 0.0, "stop")

    with tempfile.TemporaryDirectory() as root:
        (Path(root) / "facts.txt").write_text(
            "The tower is 42 meters tall.\n", encoding="utf-8")
        policy = Policy(_Backend([" 1", " 1", " 2", " 1"]),
                        PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = SPEC.run("how tall is the tower?",
                           Services(files_root=Path(root)), policy)
        assert outcome["agent"] == "files", outcome
        assert "42 meters" in (outcome["answer"] or ""), outcome
    print("smoke OK")
