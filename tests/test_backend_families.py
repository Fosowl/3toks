"""Offline tests for model-family templates, detection, and stop merging.

The failure class under test is silent: a model driven with the wrong
raw-mode template still answers menus (one digit survives junk tokens)
while every multi-line generation derails — live, gemma3 under a ChatML
template stubbed every method of a factorial. These tests pin each
family's exact rendering, the tag -> family detection (including the
honest known=False fallback), and that the family's end-of-turn token
always rides along with node stop sequences.
"""
import unittest

from threetoks.backend.base import (FAMILY_CHATML, FAMILY_GEMMA,
                                    FAMILY_LLAMA3, FAMILY_MISTRAL,
                                    FAMILY_PHI3, FAMILY_R1, GenResult,
                                    ModelSpec, build_raw_prompt,
                                    detect_family, family_stops)
from threetoks.nodes import MenuNode, ShortTextNode
from threetoks.policy import Policy, PolicyConfig
from threetoks.render import Episode
from threetoks.tui import unknown_family_notice


class TemplateRenderingTest(unittest.TestCase):
    def test_gemma_folds_system_into_the_user_turn(self):
        prompt = build_raw_prompt(ModelSpec("gemma3:4b", FAMILY_GEMMA),
                                  "pick one", system="sys", prefill="ANSWER:")
        self.assertEqual(prompt,
                         "<start_of_turn>user\nsys\n\npick one<end_of_turn>\n"
                         "<start_of_turn>model\nANSWER:")
        self.assertNotIn("<|im_start|>", prompt)

    def test_llama3_uses_header_turns_with_a_system_role(self):
        prompt = build_raw_prompt(ModelSpec("llama3.2:1b", FAMILY_LLAMA3),
                                  "pick one", system="sys", prefill="ANSWER:")
        self.assertIn("<|start_header_id|>system<|end_header_id|>\n\n"
                      "sys<|eot_id|>", prompt)
        self.assertTrue(prompt.endswith(
            "<|start_header_id|>assistant<|end_header_id|>\n\nANSWER:"))

    def test_mistral_is_a_single_inst_block(self):
        prompt = build_raw_prompt(ModelSpec("mistral:7b", FAMILY_MISTRAL),
                                  "pick one", system="sys", prefill="A:")
        self.assertEqual(prompt, "[INST] sys\n\npick one [/INST]A:")

    def test_phi3_uses_pipe_role_tags(self):
        prompt = build_raw_prompt(ModelSpec("phi3:mini", FAMILY_PHI3),
                                  "pick one", system="sys", prefill="A:")
        self.assertEqual(prompt, "<|system|>\nsys<|end|>\n<|user|>\n"
                                 "pick one<|end|>\n<|assistant|>\nA:")

    def test_no_system_omits_the_system_block_everywhere(self):
        for family in (FAMILY_CHATML, FAMILY_GEMMA, FAMILY_LLAMA3,
                       FAMILY_MISTRAL, FAMILY_PHI3):
            prompt = build_raw_prompt(ModelSpec("m", family), "u",
                                      prefill="p")
            self.assertNotIn("system", prompt.lower(), family)
            self.assertTrue(prompt.endswith("p"), family)

    def test_unknown_family_raises_loudly(self):
        with self.assertRaises(ValueError):
            build_raw_prompt(ModelSpec("m", "nope"), "u")


class DetectFamilyTest(unittest.TestCase):
    KNOWN = {"qwen2.5:1.5b-instruct": FAMILY_CHATML,
             "smollm2:135m": FAMILY_CHATML,
             "gemma3:4b": FAMILY_GEMMA,
             "gemma2:2b": FAMILY_GEMMA,
             "llama3.2:1b": FAMILY_LLAMA3,
             "llama3:8b": FAMILY_LLAMA3,
             "llama2:7b": FAMILY_MISTRAL,
             "mistral:7b-instruct": FAMILY_MISTRAL,
             "mixtral:8x7b": FAMILY_MISTRAL,
             "phi3:mini": FAMILY_PHI3,
             "deepseek-r1:1.5b": FAMILY_R1}

    def test_known_tags_map_to_their_family(self):
        for tag, family in self.KNOWN.items():
            self.assertEqual(detect_family(tag), (family, True), tag)

    def test_unknown_tags_fall_back_to_chatml_flagged(self):
        for tag in ("granite4:tiny", "some-model:7b", "phi4:latest"):
            family, known = detect_family(tag)
            self.assertEqual(family, FAMILY_CHATML, tag)
            self.assertFalse(known, tag)

    def test_unknown_tags_produce_a_notice_and_known_do_not(self):
        self.assertIsNone(unknown_family_notice("qwen2.5:1.5b-instruct"))
        notice = unknown_family_notice("granite4:tiny")
        self.assertIn("granite4:tiny", notice)
        self.assertIn("ChatML", notice)


class _CapturingBackend:
    def __init__(self):
        self.opts = None

    def complete(self, model, raw_prompt, opts):
        self.opts = opts
        return GenResult(" 1", 5, 2, 0.0, "stop")


class FamilyStopMergingTest(unittest.TestCase):
    def test_node_stops_ride_with_the_family_end_token(self):
        # Node stops OVERRIDE Modelfile defaults in raw mode, so the
        # family end-of-turn must always be merged in.
        backend = _CapturingBackend()
        policy = Policy(backend, PolicyConfig(
            ModelSpec("gemma3:4b", FAMILY_GEMMA)))
        node = ShortTextNode("q", prefill="A:")     # stop=("\n",)
        policy.decide(Episode("sys", "task"), node)
        self.assertIn("\n", backend.opts.stop)
        self.assertIn("<end_of_turn>", backend.opts.stop)

    def test_menu_nodes_still_carry_the_family_stop(self):
        backend = _CapturingBackend()
        policy = Policy(backend, PolicyConfig(
            ModelSpec("qwen2.5:1.5b-instruct", FAMILY_CHATML)))
        policy.decide(Episode("sys", "task"), MenuNode("q", ["a", "b"]))
        self.assertEqual(backend.opts.stop, ("<|im_end|>",))

    def test_leaked_family_markers_are_stripped_from_short_text(self):
        node = ShortTextNode("q", prefill="A:")
        for leaked in ("answer<end_of_turn>", "answer<|eot_id|>",
                       "answer<|end|>"):
            self.assertEqual(node.parse(leaked, ()).value, "answer", leaked)


if __name__ == "__main__":
    unittest.main()
