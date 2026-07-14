"""Look agent: answer a question about what the camera currently sees.

One capture, one vision decision. The frame is grabbed as a raw base64
JPEG and attached to a ``ShortTextNode``; the policy forwards
``node.images`` into ``GenOpts`` (see threetoks/policy.py), so a vision
call is just an ordinary decision with an image set. The model answers
in a single spoken-length sentence.
"""
from threetoks.agents.base import AgentSpec
from threetoks.nodes import ShortTextNode
from threetoks.render import Episode

AGENT_NAME = "look"
AGENT_DESCRIPTION = "look through the camera and answer what is visible right now"
ANSWER_MAX_TOKENS = 60          # one spoken-length sentence, token-frugal
NO_CAMERA_ANSWER = "The camera is not available."
UNCLEAR_ANSWER = "I could not make out the image."
PREFIX = "You describe what a camera sees in one short sentence."


def _capture_fn(services):
    """The injected ``capture_frame`` callable, or the webcam default."""
    injected = getattr(services, "capture_frame", None)
    if injected is not None:
        return injected
    from threetoks.camera import capture_jpeg_b64   # optional cv2, lazy
    return capture_jpeg_b64


def _look(task: str, image: str, policy) -> str:
    """One vision decision over the frame; answer text or a fallback."""
    episode = Episode(PREFIX, task)
    node = ShortTextNode(
        f"Look at the image and answer in one short sentence: {task}",
        "", max_tokens=ANSWER_MAX_TOKENS)
    node.images = (image,)                 # policy forwards this into GenOpts
    decision = policy.decide(episode, node)
    return decision.value if decision.valid else UNCLEAR_ANSWER


def run(task: str, services, policy) -> dict:
    """Capture a frame and answer the task about it (one model call max)."""
    image = _capture_fn(services)()
    if not image:                          # no camera -> zero model calls
        return {"agent": AGENT_NAME, "answer": NO_CAMERA_ANSWER}
    return {"agent": AGENT_NAME, "answer": _look(task, image, policy)}


SPEC = AgentSpec(AGENT_NAME, AGENT_DESCRIPTION, run)


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
    from threetoks.policy import Policy, PolicyConfig

    class _Backend:
        def complete(self, model, raw_prompt, opts):
            return GenResult("a red mug on a desk", 5, 4, 0.0, "stop")

    class _Services:
        def __init__(self, frame):
            self.capture_frame = lambda: frame

    policy = Policy(_Backend(), PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
    seen = SPEC.run("what do you see?", _Services("ZmFrZQ=="), policy)
    assert seen == {"agent": "look", "answer": "a red mug on a desk"}, seen
    blind = SPEC.run("what do you see?", _Services(None), policy)
    assert blind["answer"] == NO_CAMERA_ANSWER, blind
    print("smoke OK")
