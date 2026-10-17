"""Offline tests for the planner agent: draft, follow, correct a trajectory."""
import unittest
from types import SimpleNamespace

from threetoks.agents import planner
from threetoks.agents.base import AgentSpec
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.policy import Policy, PolicyConfig


class ScriptedBackend:
    """Answers each node kind from its own script; the last answer repeats.

    ``plans`` feed the agent-order node, ``purposes`` the step sentences,
    ``updates`` the replan menu (True = yes), ``routes`` the router menu
    (an agent name). Prompts are kept for assertions.
    """

    def __init__(self, plans, purposes=(" do it",), updates=(False,),
                 routes=("casual",)):
        self.scripts = {"plan": list(plans), "purpose": list(purposes),
                        "update": list(updates), "route": list(routes)}
        self.prompts: list[str] = []

    def complete(self, model, raw_prompt, opts):
        self.prompts.append(raw_prompt)
        if raw_prompt.endswith("PLAN:"):
            return self._reply(self._next("plan"))
        if not raw_prompt.endswith("ANSWER:"):
            return self._reply(self._next("purpose"))
        if planner.UPDATE_QUESTION in raw_prompt:
            wanted = planner.OPT_UPDATE if self._next("update") \
                else planner.OPT_KEEP
            return self._reply(_digit_for(raw_prompt, wanted))
        return self._reply(_digit_for(raw_prompt, self._next("route")))

    def _next(self, kind):
        script = self.scripts[kind]
        return script.pop(0) if len(script) > 1 else script[0]

    @staticmethod
    def _reply(text):
        return GenResult(f" {text}", 5, 2, 0.0, "stop")


def _digit_for(prompt: str, option_start: str) -> str:
    """The digit of the live menu (last ACTIONS block) for option_start."""
    live_menu = prompt.rpartition("ACTIONS:")[2]
    for line in live_menu.splitlines():
        digit, _, text = line.partition(" = ")
        if digit.isdecimal() and text.startswith(option_start):
            return digit
    raise AssertionError(f"option {option_start!r} not shown")


def recording_agent(name: str, calls: list, judged_good=None):
    """An AgentSpec recording (name, task, plan context); answers 'name answer'.

    judged_good, when not None, is put in the result like the web judge.
    """
    def run(task, services, policy):
        calls.append((name, task, policy.context))
        verdict = {} if judged_good is None else {"judged_good": judged_good}
        return {"answer": f"{name} answer", "agent": name, **verdict}
    return AgentSpec(name, f"{name} things", run)


def run_planner(backend, task="print the population of France",
                services=None, rejected=()):
    """Run a planner over casual/web/code; return (result, calls).

    Agents named in ``rejected`` report judged_good False.
    """
    calls: list = []
    specs = [recording_agent(name, calls,
                             False if name in rejected else None)
             for name in ("casual", "web", "code")]
    policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    result = planner.make_spec(specs).run(task, services, policy)
    return result, calls


def agent_names(calls: list) -> list[str]:
    """The agent names in call order."""
    return [name for name, _, _ in calls]


class FollowTest(unittest.TestCase):
    def test_runs_the_planned_agents_in_order_repeats_allowed(self):
        _, calls = run_planner(ScriptedBackend(plans=["2,3,2"]))
        self.assertEqual(agent_names(calls), ["web", "code", "web"])

    def test_task_is_only_the_purpose_and_context_holds_the_goal(self):
        backend = ScriptedBackend(plans=["2,3"],
                                  purposes=["find the number",
                                            "print the number"])
        _, calls = run_planner(backend)
        self.assertEqual([task for _, task, _ in calls],
                         ["find the number", "print the number"])
        for _, _, context in calls:
            self.assertIn("OVERALL GOAL: print the population of France",
                          context)

    def test_later_step_context_reads_the_earlier_answer(self):
        _, calls = run_planner(ScriptedBackend(plans=["2,3"]))
        self.assertIn("- web: do it -> web answer", calls[1][2])
        self.assertNotIn("EARLIER STEPS", calls[0][2])

    def test_result_joins_every_step_answer_under_the_planner_name(self):
        result, _ = run_planner(ScriptedBackend(plans=["2,3"]))
        self.assertEqual(result["agent"], "planner")
        self.assertIn("web answer", result["answer"])
        self.assertIn("code answer", result["answer"])
        self.assertEqual([step["agent"] for step in result["trajectory"]],
                         ["web", "code"])
        self.assertNotIn("judged_good", result)

    def test_a_rejected_step_marks_the_whole_result_unverified(self):
        result, _ = run_planner(ScriptedBackend(plans=["2,3"]),
                                rejected=("web",))
        self.assertIs(result["judged_good"], False)

    def test_invalid_purpose_falls_back_to_the_goal(self):
        backend = ScriptedBackend(plans=["2"], purposes=["<|im_end|>"])
        _, calls = run_planner(backend)
        self.assertEqual(calls[0][1], "print the population of France")

    def test_progress_lines_show_the_plan_and_each_step(self):
        lines: list[str] = []
        services = SimpleNamespace(on_progress=lines.append)
        run_planner(ScriptedBackend(plans=["2,3"]), services=services)
        self.assertEqual(lines[0], "plan ›")
        self.assertIn("  1. web — do it", lines)
        self.assertIn("step 2 › code: do it", lines)


class CorrectTest(unittest.TestCase):
    def test_yes_replaces_the_remaining_steps(self):
        backend = ScriptedBackend(plans=["2,3", "1"], updates=[True, False])
        result, calls = run_planner(backend)
        self.assertEqual(agent_names(calls), ["web", "casual"])
        self.assertEqual(len(result["trajectory"]), 2)

    def test_replan_prompt_lists_the_done_steps(self):
        backend = ScriptedBackend(plans=["2,3", "3"], updates=[True, False])
        run_planner(backend)
        replan = [prompt for prompt in backend.prompts
                  if prompt.endswith("PLAN:")][1]
        self.assertIn(planner.DONE_HEADER, replan)
        self.assertIn("- web: do it -> web answer", replan)

    def test_update_question_shows_the_latest_result(self):
        backend = ScriptedBackend(plans=["2,3"])
        run_planner(backend)
        update = next(prompt for prompt in backend.prompts
                      if planner.UPDATE_QUESTION in prompt)
        self.assertIn("LATEST RESULT (web: do it):\nweb answer", update)
        self.assertIn("1. web — do it (done)", update)

    def test_unparseable_replan_keeps_the_old_steps(self):
        backend = ScriptedBackend(plans=["2,3", "none"],
                                  updates=[True, False])
        _, calls = run_planner(backend)
        self.assertEqual(agent_names(calls), ["web", "code"])

    def test_replans_stop_at_the_cap(self):
        backend = ScriptedBackend(plans=["2"], updates=[True])
        _, calls = run_planner(backend)
        self.assertEqual(len(calls), 1 + planner.MAX_REPLANS)
        asked = sum(planner.UPDATE_QUESTION in prompt
                    for prompt in backend.prompts)
        self.assertEqual(asked, planner.MAX_REPLANS)

    def test_steps_stop_at_the_trajectory_cap_and_say_so(self):
        lines: list[str] = []
        backend = ScriptedBackend(plans=["1,2,3,1"],
                                  updates=[False, False, True, True, False])
        _, calls = run_planner(backend,
                               services=SimpleNamespace(
                                   on_progress=lines.append))
        self.assertEqual(len(calls), planner.MAX_TRAJECTORY_STEPS)
        self.assertIn("step cap reached › 2 step(s) not run", lines)


class FallbackTest(unittest.TestCase):
    def test_unparseable_plan_routes_to_one_agent(self):
        backend = ScriptedBackend(plans=["sure!"], routes=["code"])
        result, calls = run_planner(backend, task="reverse a string")
        self.assertEqual(calls, [("code", "reverse a string", "")])
        self.assertEqual(result["agent"], "code")


if __name__ == "__main__":
    unittest.main()
