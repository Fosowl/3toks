"""Offline tests for the code agent's four-mode dispatch."""
import tempfile
import unittest
from pathlib import Path

from threetoks.agents import code as code_agent
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.policy import Policy, PolicyConfig
from threetoks.services import Services


class QueueBackend:
    """Pops canned completions; repeats the last one when exhausted."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def complete(self, model, raw_prompt, opts):
        self.calls += 1
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return GenResult(text, 8, 4, 0.0, "stop")


def _policy(backend):
    return Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


def _corpus(tree):
    holder = tempfile.TemporaryDirectory()
    root = Path(holder.name)
    for rel, source in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return holder, root


class AuthorModeTest(unittest.TestCase):
    def test_write_request_routes_free_and_returns_the_module(self):
        backend = QueueBackend(["square(n): multiply n by itself",
                                "return n * n"])
        outcome = code_agent.run("write a function that squares a number",
                                 Services(), _policy(backend))
        self.assertEqual(outcome["mode"], "author")
        self.assertTrue(outcome["routed_by"].startswith("pre:"))
        self.assertIn("return n * n", outcome["answer"])


class ComputeModeTest(unittest.TestCase):
    def test_compute_answers_with_the_scripts_output(self):
        backend = QueueBackend(["main(): print the sum of 2 and 3",
                                "print(2 + 3)"])
        outcome = code_agent.run("sum the numbers in this data for me",
                                 Services(), _policy(backend))
        self.assertEqual(outcome["mode"], "compute")
        self.assertEqual(outcome["answer"], "5")
        self.assertIn("print(2 + 3)", outcome["script"])

    def test_compute_falls_back_to_the_script_when_output_is_empty(self):
        backend = QueueBackend(["quiet(): compute silently",
                                "return 2 + 3"])
        outcome = code_agent.run("sum the numbers in this data for me",
                                 Services(), _policy(backend))
        self.assertEqual(outcome["mode"], "compute")
        self.assertIn("did not produce output", outcome["answer"])


class NavigateModeTest(unittest.TestCase):
    def test_unique_symbol_answers_verbatim_with_zero_model_calls(self):
        holder, root = _corpus({"pkg/mod.py": (
            'def greet(name):\n    """Say hi."""\n    return f"hi {name}"\n')})
        self.addCleanup(holder.cleanup)
        backend = QueueBackend(["unused"])
        outcome = code_agent.run("where is the greet function defined?",
                                 Services(files_root=root), _policy(backend))
        self.assertEqual(outcome["mode"], "navigate")
        self.assertEqual(backend.calls, 0)          # fully free
        self.assertIn("pkg/mod.py:1", outcome["answer"])
        self.assertIn('return f"hi {name}"', outcome["answer"])

    def test_no_match_is_an_honest_miss(self):
        holder, root = _corpus({"pkg/mod.py": "def greet():\n    return 1\n"})
        self.addCleanup(holder.cleanup)
        outcome = code_agent.run("where is the frobnicate function defined?",
                                 Services(files_root=root),
                                 _policy(QueueBackend(["unused"])))
        self.assertEqual(outcome["mode"], "navigate")
        self.assertIn("no matching", outcome["answer"])


class EditModeTest(unittest.TestCase):
    def test_edit_applies_and_reports_the_new_source(self):
        holder, root = _corpus(
            {"pkg/mod.py": "def add(a, b):\n    return a - b\n"})
        self.addCleanup(holder.cleanup)

        class MenuOrBody(QueueBackend):
            def complete(self, model, raw_prompt, opts):
                self.calls += 1
                if "ACTIONS:" in raw_prompt:
                    for line in raw_prompt.splitlines():
                        if line[:1].isdigit() and "replace the target" in line:
                            return GenResult(f" {line.split(' = ')[0]}",
                                             5, 1, 0.0, "stop")
                    return GenResult(" 1", 5, 1, 0.0, "stop")
                return GenResult("return a + b", 8, 4, 0.0, "stop")

        outcome = code_agent.run("fix the bug in add in pkg/mod.py",
                                 Services(files_root=root),
                                 _policy(MenuOrBody(["unused"])))
        self.assertEqual(outcome["mode"], "edit")
        self.assertTrue(outcome["success"], outcome)
        self.assertFalse(outcome["verified"])       # no test in the agent path
        self.assertIn("return a + b", outcome["answer"])
        self.assertIn("a + b", (root / "pkg" / "mod.py").read_text())


if __name__ == "__main__":
    unittest.main()
