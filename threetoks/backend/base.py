"""Model-family prompt templates and the backend protocol.

ThreeToks builds chat templates itself (raw mode) so it can pre-close the
<think> block of reasoning distills and steer output with a prefill —
the two tricks that make 1-token decisions work on tiny models.
"""
from dataclasses import dataclass, field
from typing import Protocol

R1_BOS = "<｜begin▁of▁sentence｜>"
R1_USER = "<｜User｜>"
R1_ASSISTANT = "<｜Assistant｜>"
R1_THINK_CLOSED = "<think>\n\n</think>\n\n"

FAMILY_R1 = "r1"
FAMILY_CHATML = "chatml"


@dataclass(frozen=True)
class ModelSpec:
    """An Ollama model tag plus the template family it speaks."""
    name: str
    family: str


@dataclass(frozen=True)
class GenOpts:
    """Generation options for one completion."""
    max_tokens: int
    temperature: float = 0.0
    stop: tuple[str, ...] = ()
    num_ctx: int = 2048
    seed: int | None = None


@dataclass
class GenResult:
    """One generation: text plus token/latency accounting."""
    text: str
    prompt_tokens: int
    out_tokens: int
    wall_s: float
    done_reason: str
    raw: dict = field(repr=False, default_factory=dict)


def build_raw_prompt(spec: ModelSpec, user: str, system: str = "",
                     think: bool = False, prefill: str = "") -> str:
    """Assemble a raw prompt for the model family.

    For R1 the system text is folded into the user turn (DeepSeek guidance);
    think=False pre-closes the reasoning block. For ChatML, system is a real
    system turn and think is ignored.
    """
    if spec.family == FAMILY_R1:
        merged = f"{system}\n\n{user}" if system else user
        base = f"{R1_BOS}{R1_USER}{merged}{R1_ASSISTANT}"
        opened = f"<think>\n{prefill}"
        return base + (opened if think else R1_THINK_CLOSED + prefill)
    if spec.family == FAMILY_CHATML:
        sys_block = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        return (f"{sys_block}<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{prefill}")
    raise ValueError(f"unknown model family: {spec.family}")


class LLMBackend(Protocol):
    """Transport that turns a raw prompt into a completion."""

    def complete(self, model: str, raw_prompt: str, opts: GenOpts) -> GenResult:
        """Run one raw-mode completion."""
        ...


if __name__ == "__main__":
    spec = ModelSpec("deepseek-r1:1.5b", FAMILY_R1)
    prompt = build_raw_prompt(spec, "Pick 1 or 2.", prefill="ANSWER:")
    assert prompt.endswith("</think>\n\nANSWER:") and R1_USER in prompt
    chatml = build_raw_prompt(ModelSpec("qwen2.5:1.5b-instruct", FAMILY_CHATML),
                              "Pick 1 or 2.", system="s", prefill="ANSWER:")
    assert chatml.endswith("assistant\nANSWER:") and "<|im_start|>system" in chatml
    print("smoke OK")
