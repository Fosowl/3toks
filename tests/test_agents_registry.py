"""Offline tests for the agent registry (default_agents)."""
import unittest

from threetoks.agents import AgentSpec, default_agents
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


if __name__ == "__main__":
    unittest.main()
