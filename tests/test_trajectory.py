"""Offline tests for the trajectory data and the texts agents read."""
import unittest

from threetoks.trajectory import (CONTEXT_HEADER, STEP_ANSWER_CLIP, Step,
                                  Trajectory)


def two_step_trajectory() -> Trajectory:
    """Goal with one done web step and one code step still to run."""
    trajectory = Trajectory("print the population of France",
                            [Step("code", "print that number")])
    trajectory.done.append(Step("web", "find the population", "68 million"))
    return trajectory


class ContextTest(unittest.TestCase):
    def test_first_step_gets_the_goal_only(self):
        trajectory = Trajectory("goal", [Step("web", "find it")])
        self.assertEqual(trajectory.context(),
                         f"{CONTEXT_HEADER}\nOVERALL GOAL: goal")

    def test_later_step_lists_each_done_answer(self):
        self.assertEqual(two_step_trajectory().context().splitlines(), [
            CONTEXT_HEADER,
            "OVERALL GOAL: print the population of France",
            "EARLIER STEPS:",
            "- web: find the population -> 68 million"])

    def test_long_answers_are_flattened_and_clipped(self):
        trajectory = two_step_trajectory()
        trajectory.done[0].answer = "line one\n" + "x" * 1000
        line = trajectory.render_done()
        self.assertNotIn("\n", line)
        answer = line.split(" -> ", 1)[1]
        self.assertEqual(len(answer), STEP_ANSWER_CLIP)
        self.assertTrue(answer.endswith("…"))

    def test_missing_answer_reads_as_no_answer(self):
        trajectory = two_step_trajectory()
        trajectory.done[0].answer = ""
        self.assertIn("-> (no answer)", trajectory.render_done())


class RenderTest(unittest.TestCase):
    def test_plan_numbers_done_then_remaining_steps(self):
        self.assertEqual(two_step_trajectory().render_plan().splitlines(), [
            "1. web — find the population (done)",
            "2. code — print that number"])

    def test_latest_names_the_last_done_step(self):
        self.assertEqual(two_step_trajectory().render_latest(),
                         "LATEST RESULT (web: find the population):\n"
                         "68 million")

    def test_latest_is_empty_before_any_step(self):
        self.assertEqual(Trajectory("goal", []).render_latest(), "")


class SummaryTest(unittest.TestCase):
    def test_one_step_gives_its_answer_unchanged(self):
        trajectory = two_step_trajectory()
        self.assertEqual(trajectory.summary(), "68 million")

    def test_several_steps_keep_every_answer_in_order(self):
        trajectory = two_step_trajectory()
        trajectory.done.append(Step("code", "print it", "print(68)"))
        self.assertEqual(trajectory.summary(),
                         "1. web — find the population\n68 million\n\n"
                         "2. code — print it\nprint(68)")


if __name__ == "__main__":
    unittest.main()
