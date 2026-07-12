"""Offline tests for the files agent (real temp trees, scripted decisions)."""
import tempfile
import unittest
from pathlib import Path

from threetoks.agents import files
from threetoks.agents.files import (MAX_FILE_BYTES, OPT_GO_UP,
                                   FileExplorerVertical)
from threetoks.nodes import Decision
from threetoks.web.notes import NoteStore

FACT = "The Eiffel Tower is 330 meters tall."


def _menu(value):
    return Decision("menu", value, value, valid=True)


def _pick(indices):
    return Decision("pick_many", indices, str(indices), valid=True)


class TempTree:
    """Context manager yielding a populated temp directory as a Path."""

    def __init__(self, files_map):
        self._files_map = files_map
        self._tmp = tempfile.TemporaryDirectory()

    def __enter__(self) -> Path:
        root = Path(self._tmp.name)
        for name, content in self._files_map.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
        return root

    def __exit__(self, *exc):
        self._tmp.cleanup()


class WalkTest(unittest.TestCase):
    def test_open_autonote_answer_quotes_fact_verbatim(self):
        with TempTree({"facts.txt": FACT + "\nBuilt in 1889.\n"}) as root:
            vertical = FileExplorerVertical("how tall is the tower?",
                                            root, NoteStore())
            menu = vertical.next_node()
            self.assertTrue(menu.options[0].startswith("open: facts.txt"))
            vertical.apply(menu, _menu(menu.options[0]))
            vertical.apply(vertical.next_node(), _pick([1]))  # auto-note pass
            page_menu = vertical.next_node()
            vertical.apply(page_menu, _menu("answer the task now"))
            vertical.apply(vertical.next_node(), _pick([1]))  # answer pick
            self.assertIsNone(vertical.next_node())
            self.assertEqual(vertical.result()["answer"], FACT)

    def test_note_provenance_records_file_path_and_line(self):
        with TempTree({"facts.txt": FACT + "\n"}) as root:
            notes = NoteStore()
            vertical = FileExplorerVertical("q", root, notes)
            menu = vertical.next_node()
            vertical.apply(menu, _menu(menu.options[0]))
            vertical.apply(vertical.next_node(), _pick([1]))
            self.assertEqual(notes.texts(), [FACT])

    def test_run_via_agent_spec_tags_result(self):
        from threetoks.backend.base import (FAMILY_CHATML, GenResult,
                                           ModelSpec)
        from threetoks.policy import Policy, PolicyConfig
        from threetoks.services import Services

        class Backend:
            def __init__(self, texts):
                self.texts = list(texts)

            def complete(self, model, raw_prompt, opts):
                return GenResult(self.texts.pop(0) if self.texts else " 9",
                                 5, 2, 0.0, "stop")

        with TempTree({"facts.txt": FACT + "\n"}) as root:
            policy = Policy(Backend([" 1", " 1", " 2", " 1"]),
                            PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
            result = files.SPEC.run("how tall?",
                                    Services(files_root=root), policy)
            self.assertEqual(result["agent"], "files")
            self.assertEqual(result["answer"], FACT)


class GuardTest(unittest.TestCase):
    def test_hidden_file_is_not_listed(self):
        with TempTree({"visible.txt": "x\n", ".secret.txt": "x\n"}) as root:
            vertical = FileExplorerVertical("q", root, NoteStore())
            options = vertical.next_node().options
            self.assertTrue(any("visible.txt" in o for o in options))
            self.assertFalse(any("secret" in o for o in options))

    def test_oversized_file_is_not_listed(self):
        big = "x" * (MAX_FILE_BYTES + 1)
        with TempTree({"small.txt": "x\n", "big.txt": big}) as root:
            vertical = FileExplorerVertical("q", root, NoteStore())
            options = vertical.next_node().options
            self.assertTrue(any("small.txt" in o for o in options))
            self.assertFalse(any("big.txt" in o for o in options))

    def test_binary_extension_is_not_listed(self):
        with TempTree({"note.txt": "x\n",
                       "model.gguf": b"\x00\x01"}) as root:
            vertical = FileExplorerVertical("q", root, NoteStore())
            options = vertical.next_node().options
            self.assertFalse(any("gguf" in o for o in options))

    def test_go_up_absent_at_root(self):
        with TempTree({"a.txt": "x\n"}) as root:
            vertical = FileExplorerVertical("q", root, NoteStore())
            self.assertNotIn(OPT_GO_UP, vertical.next_node().options)

    def test_go_up_present_below_root(self):
        with TempTree({"sub/inner.txt": "x\n"}) as root:
            vertical = FileExplorerVertical("q", root, NoteStore())
            menu = vertical.next_node()
            sub_option = next(o for o in menu.options if "sub" in o)
            vertical.apply(menu, _menu(sub_option))
            self.assertIn(OPT_GO_UP, vertical.next_node().options)

    def test_cannot_escape_root_via_safe_target(self):
        with TempTree({"a.txt": "x\n"}) as root:
            vertical = FileExplorerVertical("q", root, NoteStore())
            self.assertIsNone(vertical._safe_target(Path("/etc")))
            self.assertIsNone(vertical._safe_target(root.parent))

    def test_go_up_at_root_stays_at_root(self):
        with TempTree({"a.txt": "x\n"}) as root:
            vertical = FileExplorerVertical("q", root, NoteStore())
            vertical._go_up()
            self.assertEqual(vertical.cwd, vertical.root)


class ChunkTest(unittest.TestCase):
    def test_second_chunk_offered_for_long_file(self):
        body = "\n".join(f"Line number {i}." for i in range(1, 40)) + "\n"
        with TempTree({"long.txt": body}) as root:
            vertical = FileExplorerVertical("q", root, NoteStore())
            menu = vertical.next_node()
            vertical.apply(menu, _menu(menu.options[0]))
            vertical.apply(vertical.next_node(), _pick([1]))  # note pass
            page_menu = vertical.next_node()
            self.assertIn(files.OPT_NEXT_CHUNK, page_menu.options)


if __name__ == "__main__":
    unittest.main()
