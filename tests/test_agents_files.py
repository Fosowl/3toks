"""Offline tests for the files agent (real temp trees, scripted decisions)."""
import re
import tempfile
import unittest
from pathlib import Path

from threetoks.agents import files
from threetoks.agents.files import (CMD_PREFILL, FINAL_PREFILL,
                                   MAX_FILE_BYTES, MAX_SHELL_RUNS,
                                   OPT_ANSWER, OPT_GO_UP, OPT_SHELL,
                                   FileExplorerVertical, command_is_safe,
                                   dangerous_command)
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.nodes import Decision, PickManyNode, ShortTextNode
from threetoks.policy import Policy, PolicyConfig
from threetoks.web.notes import NoteStore

FACT = "The Eiffel Tower is 330 meters tall."


def _menu(value):
    return Decision("menu", value, value, valid=True)


def _pick(indices):
    return Decision("pick_many", indices, str(indices), valid=True)


def _text(value):
    return Decision("short_text", value, value, valid=True)


class LabelBackend:
    """Answers menus by option label (permutation-proof); rest scripted.

    Script entries are ("menu", label_prefix) or ("text", payload); an
    exhausted script raises, proving no unexpected model call happened.
    """

    def __init__(self, script):
        self.script = list(script)

    def complete(self, model, raw_prompt, opts):
        kind, payload = self.script.pop(0)
        if kind == "menu":
            match = re.search(rf"(\d+) = {re.escape(payload)}", raw_prompt)
            return GenResult(f" {match.group(1)}" if match else " 9",
                             5, 2, 0.0, "stop")
        return GenResult(f" {payload}", 5, 2, 0.0, "stop")


def make_policy(script):
    return Policy(LabelBackend(script),
                  PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


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
    def test_open_autonote_then_synthesised_answer(self):
        with TempTree({"facts.txt": FACT + "\nBuilt in 1889.\n"}) as root:
            vertical = FileExplorerVertical("how tall is the tower?",
                                            root, NoteStore())
            menu = vertical.next_node()
            self.assertTrue(menu.options[0].startswith("open: facts.txt"))
            vertical.apply(menu, _menu(menu.options[0]))
            vertical.apply(vertical.next_node(), _pick([1]))  # auto-note pass
            page_menu = vertical.next_node()
            vertical.apply(page_menu, _menu(OPT_ANSWER))
            synth = vertical.next_node()
            self.assertIsInstance(synth, ShortTextNode)
            self.assertEqual(synth.prefill, FINAL_PREFILL)
            self.assertIn("how tall is the tower?", synth.question)
            self.assertIn(FACT, vertical.episode.render_base())  # grounded
            vertical.apply(synth, _text("It is 330 meters tall."))
            self.assertIsNone(vertical.next_node())
            self.assertEqual(vertical.result()["answer"],
                             "It is 330 meters tall.")

    def test_note_provenance_records_file_path_and_line(self):
        with TempTree({"facts.txt": FACT + "\n"}) as root:
            notes = NoteStore()
            vertical = FileExplorerVertical("q", root, notes)
            menu = vertical.next_node()
            vertical.apply(menu, _menu(menu.options[0]))
            vertical.apply(vertical.next_node(), _pick([1]))
            self.assertEqual(notes.texts(), [FACT])
            self.assertTrue(
                notes.entries()[0].source_url.endswith("facts.txt"))

    def test_run_via_agent_spec_tags_result(self):
        from threetoks.services import Services

        with TempTree({"facts.txt": FACT + "\n"}) as root:
            policy = make_policy([("menu", "open: facts.txt"),
                                  ("text", "1"),
                                  ("menu", OPT_ANSWER),
                                  ("text", "It is 330 meters tall.")])
            result = files.SPEC.run("how tall?",
                                    Services(files_root=root), policy)
            self.assertEqual(result["agent"], "files")
            self.assertEqual(result["answer"], "It is 330 meters tall.")


class AnswerFlowTest(unittest.TestCase):
    """The final answer is phrased from notes, not a verbatim dump."""

    def _vertical_with_notes(self, texts, task="how tall is the tower?"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        notes = NoteStore()
        for index, text in enumerate(texts, 1):
            notes.add(text, "file.txt", index)
        return FileExplorerVertical(task, Path(tmp.name), notes)

    def test_big_store_is_curated_before_synthesis(self):
        from threetoks.research import CURATE_KEEP
        texts = [f"Distinct fact number {i}." for i in range(CURATE_KEEP + 3)]
        vertical = self._vertical_with_notes(texts)
        vertical.apply(vertical.next_node(), _menu(OPT_ANSWER))
        curate = vertical.next_node()
        self.assertIsInstance(curate, PickManyNode)
        self.assertEqual(getattr(curate, "tag", ""), "curate")
        vertical.apply(curate, _pick([2, 1]))
        self.assertEqual(vertical.notes.texts(),
                         ["Distinct fact number 1.",
                          "Distinct fact number 0."])
        synth = vertical.next_node()
        self.assertIsInstance(synth, ShortTextNode)
        self.assertEqual(synth.prefill, FINAL_PREFILL)

    def test_double_bleed_falls_back_to_goal_closest_notes(self):
        vertical = self._vertical_with_notes(
            ["Paris is in France.", FACT])
        vertical.apply(vertical.next_node(), _menu(OPT_ANSWER))
        vertical.apply(vertical.next_node(), _text("3"))   # bleed once
        vertical.apply(vertical.next_node(), _text("7"))   # bleed again
        answer = vertical.result()["answer"]
        self.assertTrue(answer.startswith(FACT), answer)  # tower note first

    def test_persistent_single_digit_is_accepted_as_answer(self):
        vertical = self._vertical_with_notes(["There are 3 files here."],
                                             task="how many files?")
        vertical.apply(vertical.next_node(), _menu(OPT_ANSWER))
        vertical.apply(vertical.next_node(), _text("3"))
        vertical.apply(vertical.next_node(), _text("3"))
        self.assertEqual(vertical.result()["answer"], "3")


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


class DenyListTest(unittest.TestCase):
    """Layer 1: the deterministic deny-list blocks non-read-only shapes."""

    def test_write_delete_and_execute_words_blocked(self):
        for command in ("rm -rf .", "mv a b", "chmod 777 x", "sudo ls",
                        "python3 x.py", "curl example.com", "git status",
                        "touch new.txt", "mkdir sub"):
            self.assertTrue(dangerous_command(command), command)

    def test_metacharacters_blocked(self):
        for command in ("cat a > b", "cat a | grep x", "ls; ls",
                        "echo `ls`", "echo $(ls)", "ls & ls"):
            self.assertTrue(dangerous_command(command), command)

    def test_sandbox_escaping_paths_blocked(self):
        for command in ("cat ../secret", "ls /", "cat /etc/passwd",
                        "ls ~/private"):
            self.assertTrue(dangerous_command(command), command)

    def test_plain_read_only_commands_pass(self):
        for command in ("ls", "cat notes.txt", "wc -l data.csv",
                        "grep -r flag .", "find . -name x.py",
                        "head -5 log.txt"):
            self.assertFalse(dangerous_command(command), command)


class ShellTest(unittest.TestCase):
    def _vertical(self, root, policy):
        return FileExplorerVertical("what are these files?", root,
                                    NoteStore(), policy=policy)

    def _propose(self, vertical, command):
        """Walk menu -> shell option -> command proposal."""
        menu = vertical.next_node()
        vertical.apply(menu, _menu(OPT_SHELL))
        node = vertical.next_node()
        assert node.prefill == CMD_PREFILL, node.prefill
        vertical.apply(node, _text(command))

    def test_shell_option_needs_a_policy(self):
        with TempTree({"a.txt": "x\n"}) as root:
            with_policy = self._vertical(root, make_policy([]))
            self.assertIn(OPT_SHELL, with_policy.next_node().options)
            without = FileExplorerVertical("q", root, NoteStore())
            self.assertNotIn(OPT_SHELL, without.next_node().options)

    def test_dangerous_command_blocked_without_any_model_call(self):
        with TempTree({"a.txt": "x\n"}) as root:
            vertical = self._vertical(root, make_policy([]))  # empty script:
            self._propose(vertical, "rm -rf .")  # a judge call would raise
            self.assertEqual(vertical.shell_runs, 1)
            self.assertIn("blocked (looked unsafe)",
                          vertical.episode.render_base())

    def test_judge_rejection_blocks_execution(self):
        with TempTree({"a.txt": "x\n"}) as root:
            vertical = self._vertical(root, make_policy([("menu", "unsafe")]))
            self._propose(vertical, "cat a.txt")
            self.assertEqual(vertical.notes.count(), 0)
            self.assertIn("blocked by the safety judge",
                          vertical.episode.render_base())

    def test_approved_command_output_feeds_note_pass(self):
        with TempTree({"facts.txt": FACT + "\n"}) as root:
            vertical = self._vertical(root, make_policy([("menu", "safe")]))
            self._propose(vertical, "cat facts.txt")
            note_pass = vertical.next_node()
            self.assertIsInstance(note_pass, PickManyNode)
            self.assertEqual(note_pass.items, [FACT])
            vertical.apply(note_pass, _pick([1]))
            self.assertEqual(vertical.notes.texts(), [FACT])
            self.assertEqual(vertical.notes.entries()[0].source_url,
                             "$ cat facts.txt")

    def test_shell_option_disappears_after_budget_spent(self):
        with TempTree({"a.txt": "x\n"}) as root:
            vertical = self._vertical(root, make_policy([]))
            for _ in range(MAX_SHELL_RUNS):
                self._propose(vertical, "rm -rf .")  # burns a run, no judge
            self.assertNotIn(OPT_SHELL, vertical.next_node().options)

    def test_fresh_judge_context_sees_only_the_command(self):
        recorded = []

        class SpyBackend:
            def complete(self, model, raw_prompt, opts):
                recorded.append(raw_prompt)
                return GenResult(" 1", 5, 2, 0.0, "stop")

        policy = Policy(SpyBackend(),
                        PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        command_is_safe("wc -l a.txt", policy)
        self.assertEqual(len(recorded), 1)
        self.assertIn("wc -l a.txt", recorded[0])
        self.assertNotIn("FOLDER", recorded[0])  # no exploration context


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
