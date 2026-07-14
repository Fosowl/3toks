"""Model-family prompt templates and the backend protocol.

ThreeToks builds chat templates itself (raw mode) so it can pre-close the
<think> block of reasoning distills and steer output with a prefill —
the two tricks that make 1-token decisions work on tiny models.

Raw mode means the harness owns the whole template: a model driven with
the wrong family sees literal special-token junk, and while menu digits
survive that, multi-line generations derail (live failure: gemma3 under
a ChatML template stubbed every method of a factorial). ``detect_family``
maps known Ollama tag patterns to their template; an unknown tag falls
back to ChatML with ``known=False`` so callers can warn instead of
failing silently. The settled policy model remains qwen2.5:1.5b-instruct
(DESIGN.md §11) — other families are supported, not recommended.
"""
import re
from dataclasses import dataclass, field
from typing import Protocol

R1_BOS = "<｜begin▁of▁sentence｜>"
R1_USER = "<｜User｜>"
R1_ASSISTANT = "<｜Assistant｜>"
R1_THINK_CLOSED = "<think>\n\n</think>\n\n"

FAMILY_R1 = "r1"
FAMILY_CHATML = "chatml"
FAMILY_GEMMA = "gemma"
FAMILY_LLAMA3 = "llama3"
FAMILY_MISTRAL = "mistral"
FAMILY_PHI3 = "phi3"

# The end-of-turn token each family emits when it is done talking. Nodes
# set their own stop sequences, which OVERRIDE the Modelfile's defaults in
# raw mode — so the family stop must always ride along, or a generation
# node's custom stops let the model run past its own end of turn and leak
# template markers into the completion.
FAMILY_STOPS = {
    FAMILY_R1: ("<｜end▁of▁sentence｜>",),
    FAMILY_CHATML: ("<|im_end|>",),
    FAMILY_GEMMA: ("<end_of_turn>",),
    FAMILY_LLAMA3: ("<|eot_id|>",),
    FAMILY_MISTRAL: ("</s>",),
    FAMILY_PHI3: ("<|end|>",),
}

# Ollama tag substrings -> family, checked in order (first hit wins).
# Only patterns whose template is actually known belong here; anything
# else is driven as ChatML with known=False so the caller can warn.
_FAMILY_PATTERNS = (
    ("r1", FAMILY_R1),
    ("gemma", FAMILY_GEMMA),
    ("llama3", FAMILY_LLAMA3),
    ("llama-3", FAMILY_LLAMA3),
    ("llama2", FAMILY_MISTRAL),      # nearest [INST]-style template
    ("mistral", FAMILY_MISTRAL),
    ("mixtral", FAMILY_MISTRAL),
    ("phi3", FAMILY_PHI3),
    ("phi-3", FAMILY_PHI3),
    ("qwen", FAMILY_CHATML),
    ("smollm", FAMILY_CHATML),
)


@dataclass(frozen=True)
class ModelSpec:
    """An Ollama model tag plus the template family it speaks."""
    name: str
    family: str


@dataclass(frozen=True)
class GenOpts:
    """Generation options for one completion.

    ``images`` are raw base64 JPEGs for multimodal models; each backend
    adds its own wire framing (Ollama field, data URI, source block).
    """
    max_tokens: int
    temperature: float = 0.0
    stop: tuple[str, ...] = ()
    num_ctx: int = 2048
    seed: int | None = None
    images: tuple[str, ...] = ()


@dataclass
class GenResult:
    """One generation: text plus token/latency accounting."""
    text: str
    prompt_tokens: int
    out_tokens: int
    wall_s: float
    done_reason: str
    raw: dict = field(repr=False, default_factory=dict)


def detect_family(model_name: str) -> tuple[str, bool]:
    """(family, known) for an Ollama model tag.

    ``known=False`` means the tag matched no pattern and ChatML is a
    guess — the caller should surface a warning, because a wrong
    template fails silently (menus keep working, generations break).
    """
    lowered = model_name.lower()
    for pattern, family in _FAMILY_PATTERNS:
        if pattern in lowered:
            return family, True
    return FAMILY_CHATML, False


def family_stops(spec: ModelSpec) -> tuple[str, ...]:
    """The family's end-of-turn stop tokens (empty for unknown family)."""
    return FAMILY_STOPS.get(spec.family, ())


def build_raw_prompt(spec: ModelSpec, user: str, system: str = "",
                     think: bool = False, prefill: str = "") -> str:
    """Assemble a raw prompt for the model family.

    ``think`` only means something for R1 (False pre-closes the reasoning
    block). Families without a system role (R1, Gemma, Mistral) fold the
    system text into the user turn, per each vendor's guidance.
    """
    if spec.family == FAMILY_R1:
        base = f"{R1_BOS}{R1_USER}{_fold(system, user)}{R1_ASSISTANT}"
        opened = f"<think>\n{prefill}"
        return base + (opened if think else R1_THINK_CLOSED + prefill)
    if spec.family == FAMILY_CHATML:
        sys_block = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        return (f"{sys_block}<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{prefill}")
    if spec.family == FAMILY_GEMMA:
        return (f"<start_of_turn>user\n{_fold(system, user)}<end_of_turn>\n"
                f"<start_of_turn>model\n{prefill}")
    if spec.family == FAMILY_LLAMA3:
        sys_block = (f"<|start_header_id|>system<|end_header_id|>\n\n"
                     f"{system}<|eot_id|>") if system else ""
        return (f"{sys_block}<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{prefill}")
    if spec.family == FAMILY_MISTRAL:
        return f"[INST] {_fold(system, user)} [/INST]{prefill}"
    if spec.family == FAMILY_PHI3:
        sys_block = f"<|system|>\n{system}<|end|>\n" if system else ""
        return (f"{sys_block}<|user|>\n{user}<|end|>\n"
                f"<|assistant|>\n{prefill}")
    raise ValueError(f"unknown model family: {spec.family}")


def _fold(system: str, user: str) -> str:
    """System text merged into the user turn (no-system-role families)."""
    return f"{system}\n\n{user}" if system else user


class LLMBackend(Protocol):
    """Transport that turns a raw prompt into a completion."""

    def complete(self, model: str, raw_prompt: str, opts: GenOpts) -> GenResult:
        """Run one raw-mode completion."""
        ...


@dataclass(frozen=True)
class ChatPrompt:
    """Un-templated prompt parts for chat-API transports.

    Chat providers own their template, so the harness hands them parts
    instead of a rendered raw string; how faithfully ``prefill`` can be
    honoured is each backend's business (see threetoks/backend/providers.py).
    """
    system: str
    user: str
    prefill: str = ""


class ChatBackend(Protocol):
    """Transport that speaks a chat API instead of raw prompts.

    The policy dispatches to ``chat`` (instead of build_raw_prompt +
    ``complete``) whenever a backend exposes it as a callable.
    """

    def chat(self, model: str, prompt: ChatPrompt, opts: GenOpts) -> GenResult:
        """Run one chat completion from un-templated parts."""
        ...


# Model-name marks meaning "accepts image input" (no API introspection
# exists across providers; a name match is the practical test).
_VISION_MODEL_MARKS = ("vision", "llava", "gpt-4o", "gpt-4-turbo", "gpt-4.1",
                       "gpt-5", "gemini", "claude-3", "claude-4",
                       "claude-opus", "claude-sonnet", "claude-haiku",
                       "pixtral", "yi-vision", "glm-4v", "internvl",
                       "molmo", "minicpm-v", "moondream")
_QWEN_VL_PATTERN = re.compile(r"qwen[\w.-]*vl(?![a-z])")


def is_vision_model(model_name: str) -> bool:
    """Whether ``model_name`` names a model that accepts image input."""
    lowered = model_name.lower()
    if any(mark in lowered for mark in _VISION_MODEL_MARKS):
        return True
    return bool(_QWEN_VL_PATTERN.search(lowered))


if __name__ == "__main__":
    spec = ModelSpec("deepseek-r1:1.5b", FAMILY_R1)
    prompt = build_raw_prompt(spec, "Pick 1 or 2.", prefill="ANSWER:")
    assert prompt.endswith("</think>\n\nANSWER:") and R1_USER in prompt
    chatml = build_raw_prompt(ModelSpec("qwen2.5:1.5b-instruct", FAMILY_CHATML),
                              "Pick 1 or 2.", system="s", prefill="ANSWER:")
    assert chatml.endswith("assistant\nANSWER:") and "<|im_start|>system" in chatml

    gemma = build_raw_prompt(ModelSpec("gemma3:4b", FAMILY_GEMMA),
                             "Pick 1 or 2.", system="s", prefill="ANSWER:")
    assert gemma.startswith("<start_of_turn>user\ns\n\nPick")
    assert gemma.endswith("<start_of_turn>model\nANSWER:")
    llama = build_raw_prompt(ModelSpec("llama3.2:1b", FAMILY_LLAMA3),
                             "u", system="s", prefill="ANSWER:")
    assert "system<|end_header_id|>\n\ns<|eot_id|>" in llama
    assert llama.endswith("assistant<|end_header_id|>\n\nANSWER:")
    mistral = build_raw_prompt(ModelSpec("mistral:7b", FAMILY_MISTRAL),
                               "u", system="s", prefill="A:")
    assert mistral == "[INST] s\n\nu [/INST]A:"
    phi = build_raw_prompt(ModelSpec("phi3:mini", FAMILY_PHI3), "u",
                           system="s", prefill="A:")
    assert phi.endswith("<|assistant|>\nA:") and "<|system|>\ns<|end|>" in phi

    assert detect_family("qwen2.5:1.5b-instruct") == (FAMILY_CHATML, True)
    assert detect_family("gemma3:4b") == (FAMILY_GEMMA, True)
    assert detect_family("llama3.2:1b") == (FAMILY_LLAMA3, True)
    assert detect_family("mixtral:8x7b") == (FAMILY_MISTRAL, True)
    assert detect_family("deepseek-r1:1.5b") == (FAMILY_R1, True)
    assert detect_family("phi3:mini") == (FAMILY_PHI3, True)
    assert detect_family("granite4:tiny") == (FAMILY_CHATML, False)  # unknown
    assert family_stops(ModelSpec("g", FAMILY_GEMMA)) == ("<end_of_turn>",)

    assert is_vision_model("gpt-4o-mini")
    assert is_vision_model("claude-sonnet-4-5")
    assert is_vision_model("llava:7b")
    assert is_vision_model("qwen2.5vl:7b") and is_vision_model("qwen2-vl-72b")
    assert not is_vision_model("qwen2.5:1.5b-instruct")
    assert not is_vision_model("mistral:7b")
    assert GenOpts(max_tokens=3).images == ()
    assert ChatPrompt("s", "u").prefill == ""
    print("smoke OK")
