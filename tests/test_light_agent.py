"""Offline tests for the light agent; no RPi.GPIO is imported.

Deterministic phrasings must drive a stub relay AND leave the backend
untouched (regex, not model). One ambiguous phrasing exercises the single
MenuNode decision; an escape falls through to the didn't-understand answer.
"""
import unittest

from threetoks.agents import light
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.nodes import MenuNode
from threetoks.policy import Policy, PolicyConfig
from threetoks.services import Services

MENU_OPTIONS = [light.OPT_ON, light.OPT_OFF, light.OPT_TOGGLE, light.OPT_STATUS]
ESCAPE_DIGIT = f" {len(MENU_OPTIONS) + 1}"  # escape is the last numbered option


class ScriptedBackend:
    """Pops scripted completions; records prompts (empty == never consulted)."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts = []

    def complete(self, model, raw_prompt, opts):
        self.prompts.append(raw_prompt)
        return GenResult(self.texts.pop(0), 5, 2, 0.0, "stop")


def make_policy(texts):
    """Policy over a scripted backend; returns both for prompt assertions."""
    backend = ScriptedBackend(texts)
    return Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML))), backend


def menu_reply(option, options):
    """Digit selecting ``option`` under the policy's real permutation."""
    node = MenuNode("q", list(options), escape=True)
    policy, _ = make_policy([])
    perm = policy._permutation(node, 0, 0)
    return f" {perm.index(options.index(option)) + 1}"


class StubRelay:
    """Minimal relay double recording each call and tracking on/off."""

    def __init__(self, on=False):
        self.on = on
        self.calls = []

    def set_light(self, on):
        self.calls.append(f"set_light({bool(on)})")
        self.on = bool(on)
        return "on" if self.on else "off"

    def get_status(self):
        self.calls.append("get_status")
        return "on" if self.on else "off"

    def toggle(self):
        self.calls.append("toggle")
        self.on = not self.on
        return "on" if self.on else "off"


def services_with(relay):
    """A Services carrying the injected relay (wired separately in prod)."""
    services = Services()
    services.relay = relay
    return services


class DeterministicTest(unittest.TestCase):
    """Clear phrasings settle with a regex; the model is never consulted."""

    def _run(self, task, relay):
        policy, backend = make_policy([])
        out = light.run(task, services_with(relay), policy)
        self.assertEqual(backend.prompts, [])  # no model call
        return out["answer"]

    def test_turn_on_drives_relay_on(self):
        relay = StubRelay(on=False)
        self.assertEqual(self._run("turn on the light", relay),
                         "The light is now on.")
        self.assertEqual(relay.calls, ["set_light(True)"])
        self.assertTrue(relay.on)

    def test_lights_off_drives_relay_off(self):
        relay = StubRelay(on=True)
        self.assertEqual(self._run("lights off please", relay),
                         "The light is now off.")
        self.assertEqual(relay.calls, ["set_light(False)"])
        self.assertFalse(relay.on)

    def test_toggle_flips_relay(self):
        relay = StubRelay(on=False)
        self.assertEqual(self._run("toggle the lamp", relay),
                         "The light is now on.")
        self.assertEqual(relay.calls, ["toggle"])

    def test_status_reports_without_changing(self):
        relay = StubRelay(on=False)
        self.assertEqual(self._run("is the light on?", relay),
                         "The light is off.")
        self.assertEqual(relay.calls, ["get_status"])

    def test_state_questions_never_drive_the_relay(self):
        for phrasing in ("is it on?", "are the lights on?",
                         "is the lamp off right now?"):
            relay = StubRelay(on=False)
            self.assertEqual(self._run(phrasing, relay),
                             "The light is off.", phrasing)
            self.assertEqual(relay.calls, ["get_status"], phrasing)


class MenuTest(unittest.TestCase):
    """Ambiguous phrasing spends exactly one MenuNode decision."""

    AMBIGUOUS = "do something with the lamp brightness"

    def test_menu_choice_drives_the_chosen_action(self):
        relay = StubRelay(on=False)
        policy, backend = make_policy([menu_reply(light.OPT_ON, MENU_OPTIONS)])
        out = light.run(self.AMBIGUOUS, services_with(relay), policy)
        self.assertEqual(out["answer"], "The light is now on.")
        self.assertTrue(relay.on)
        self.assertEqual(len(backend.prompts), 1)  # exactly one decision

    def test_escape_gives_didnt_understand_answer(self):
        relay = StubRelay(on=False)
        policy, _ = make_policy([ESCAPE_DIGIT])
        out = light.run(self.AMBIGUOUS, services_with(relay), policy)
        self.assertEqual(out["answer"], light.CONFUSED)
        self.assertEqual(relay.calls, [])  # nothing driven

    def test_invalid_decision_gives_didnt_understand_answer(self):
        relay = StubRelay(on=False)
        policy, backend = make_policy(["x", "x", "x"])  # never parses
        out = light.run(self.AMBIGUOUS, services_with(relay), policy)
        self.assertEqual(out["answer"], light.CONFUSED)
        self.assertEqual(len(backend.prompts), 3)  # retry ladder exhausted


class NoRelayTest(unittest.TestCase):
    def test_missing_relay_reports_no_relay(self):
        policy, backend = make_policy([])
        out = light.run("turn on the light", Services(), policy)
        self.assertEqual(out["answer"], light.NO_RELAY)
        self.assertEqual(out["agent"], "light")
        self.assertEqual(backend.prompts, [])  # guard clause, no model call


if __name__ == "__main__":
    unittest.main()
