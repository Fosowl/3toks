"""The delivered script's stdout reaches the user (runner -> result -> TUI).

The runner's zero-arg smoke call executes exactly what the ``__main__``
guard will; these tests pin that its captured stdout flows into the
result payload and renders as an output section in the result panel,
instead of being silently discarded.
"""
import unittest

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code import runner
from threetoks.code.vertical import CodeVertical
from threetoks.engine import run_episode
from threetoks.policy import Policy, PolicyConfig
from threetoks.tui import OUTPUT_MAX_LINES, render_result


class QueueBackend:
    def __init__(self, texts):
        self.texts = list(texts)

    def complete(self, model, raw_prompt, opts):
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return GenResult(text, 8, 4, 0.0, "stop")


class RunnerStdoutTest(unittest.TestCase):
    def test_smoke_call_stdout_is_captured(self):
        module = "def main():\n    print('21C in Antibes')\n"
        [result] = runner.run_checks(
            module, [runner.Check("main", "main()", runner.KIND_CALL)])
        self.assertTrue(result.passed)
        self.assertEqual(result.stdout, "21C in Antibes\n")


class VerticalOutputTest(unittest.TestCase):
    def test_result_carries_the_entry_scripts_output(self):
        backend = QueueBackend(["main(): print the weather",
                                "print('21C in Antibes')"])
        vertical = CodeVertical("print the weather", gen_tests=False)
        policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = run_episode(vertical, policy, max_steps=20)
        self.assertIn("print('21C in Antibes')", outcome["answer"])
        self.assertEqual(outcome["output"], "21C in Antibes\n")

    def test_no_runnable_entry_means_empty_output(self):
        backend = QueueBackend(["double(n): twice n", "return n * 2"])
        vertical = CodeVertical("double(n) doubles a number",
                                gen_tests=False)
        policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = run_episode(vertical, policy, max_steps=20)
        self.assertEqual(outcome["output"], "")


class PanelOutputTest(unittest.TestCase):
    def test_output_section_renders_under_the_answer(self):
        panel = render_result({"agent": "code", "answer": "def main(): ...",
                               "output": "21C in Antibes\n"}, enabled=False)
        self.assertIn("output when run:", panel)
        self.assertIn("21C in Antibes", panel)

    def test_long_output_is_clipped_with_an_honest_count(self):
        output = "\n".join(f"line {i}" for i in range(OUTPUT_MAX_LINES + 5))
        panel = render_result({"agent": "code", "answer": "x",
                               "output": output}, enabled=False)
        self.assertIn("line 0", panel)
        self.assertIn("… 5 more line(s)", panel)
        self.assertNotIn(f"line {OUTPUT_MAX_LINES + 1}", panel)

    def test_no_output_key_renders_no_section(self):
        panel = render_result({"agent": "code", "answer": "x"},
                              enabled=False)
        self.assertNotIn("output when run:", panel)


if __name__ == "__main__":
    unittest.main()
