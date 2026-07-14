"""Offline tests for the look vision agent (cv2 never imported).

A recording backend captures the GenOpts each decision sends so the tests
can assert the captured frame rode along as ``opts.images``. Capture is
always injected via a services stub, so no real camera is ever opened.
"""
import unittest

from threetoks.agents.look import (AGENT_NAME, NO_CAMERA_ANSWER,
                                   UNCLEAR_ANSWER, run)
from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.policy import Policy, PolicyConfig

FRAME_B64 = "ZmFrZS1qcGVn"          # stand-in raw base64 JPEG


class RecordingBackend:
    """Scripted completions that also record every GenOpts seen."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.opts = []
        self.calls = 0

    def complete(self, model, raw_prompt, opts):
        self.calls += 1
        self.opts.append(opts)
        return GenResult(self.texts.pop(0), 5, 2, 0.0, "stop")


class _Services:
    """Minimal services stub exposing an injected capture callable."""

    def __init__(self, frame):
        self.capture_frame = lambda: frame


def _policy(backend):
    return Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_CHATML)))


class LookAgentTest(unittest.TestCase):
    def test_answer_and_image_reach_the_backend(self):
        backend = RecordingBackend([" a cat on a chair"])
        result = run("what do you see?", _Services(FRAME_B64), _policy(backend))
        self.assertEqual(result["agent"], AGENT_NAME)
        self.assertEqual(result["answer"], "a cat on a chair")
        self.assertEqual(backend.opts[0].images, (FRAME_B64,))

    def test_no_frame_answers_with_zero_model_calls(self):
        backend = RecordingBackend([])
        result = run("what do you see?", _Services(None), _policy(backend))
        self.assertEqual(result["answer"], NO_CAMERA_ANSWER)
        self.assertEqual(result["agent"], AGENT_NAME)
        self.assertEqual(backend.calls, 0)

    def test_garbage_completions_give_the_unclear_answer(self):
        backend = RecordingBackend(["", "", ""])   # invalid parse each attempt
        result = run("what do you see?", _Services(FRAME_B64), _policy(backend))
        self.assertEqual(result["answer"], UNCLEAR_ANSWER)


if __name__ == "__main__":
    unittest.main()
