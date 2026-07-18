"""Offline tests for the agent registry (default and optional agents)."""
import unittest

from threetoks.agents import (OPTIONAL_AGENT_NAMES, AgentSpec,
                              default_agents, optional_agents)
from threetoks.services import Services


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.specs = default_agents(Services())

    def test_returns_all_builtin_specs(self):
        self.assertEqual(len(self.specs), 4)

    def test_all_are_agent_specs(self):
        self.assertTrue(all(isinstance(s, AgentSpec) for s in self.specs))

    def test_casual_is_first_router_fallback(self):
        self.assertEqual(self.specs[0].name, "casual")

    def test_names_are_the_expected_set_in_order(self):
        self.assertEqual([s.name for s in self.specs],
                         ["casual", "web", "files", "code"])

    def test_names_are_unique(self):
        names = [s.name for s in self.specs]
        self.assertEqual(len(set(names)), len(names))

    def test_each_spec_has_description_and_callable_run(self):
        for spec in self.specs:
            self.assertTrue(spec.description)
            self.assertTrue(callable(spec.run))


class OptionalAgentsTest(unittest.TestCase):
    """Capability agents register only when their capability is wired."""

    def test_bare_services_register_nothing(self):
        self.assertEqual(optional_agents(Services(), "llava:7b"), [])

    def test_relay_registers_light_regardless_of_model(self):
        wired = Services(relay=object())
        names = [s.name for s in optional_agents(wired, "qwen2.5:1.5b")]
        self.assertEqual(names, ["light"])

    def test_camera_registers_look_only_for_vision_models(self):
        wired = Services(capture_frame=lambda: "b64")
        seeing = [s.name for s in optional_agents(wired, "llava:7b")]
        blind = [s.name for s in optional_agents(wired, "qwen2.5:1.5b")]
        self.assertEqual((seeing, blind), (["look"], []))

    def test_optional_agent_names_match_the_spec_names(self):
        wired = Services(relay=object(), capture_frame=lambda: "b64")
        names = tuple(s.name for s in optional_agents(wired, "llava:7b"))
        self.assertEqual(names, OPTIONAL_AGENT_NAMES)  # sync guard

    def test_optional_names_never_collide_with_defaults(self):
        defaults = {s.name for s in default_agents(Services())}
        self.assertFalse(defaults & set(OPTIONAL_AGENT_NAMES))


if __name__ == "__main__":
    unittest.main()
