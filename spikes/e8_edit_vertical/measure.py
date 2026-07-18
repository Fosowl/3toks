"""Measurement helpers: decisions, output tokens, and delta-economics.

The delta-economics question this spike exists to answer: how many output
tokens does a selection-dominated edit spend, compared to what a naive
"regenerate the whole file" approach would have to emit? The fairest token
count for "the whole file" is the model's OWN tokenizer, so this module
makes one throwaway Ollama call per file (max_tokens=1) and reads
`prompt_eval_count` back -- not a chars/4 guess. When Ollama is unreachable
it falls back to a labelled heuristic so the numbers still print.
"""
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_DIR.parent.parent
for _p in (str(SPIKE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from threetoks.backend.base import FAMILY_CHATML, GenOpts, ModelSpec  # noqa: E402
from threetoks.backend.ollama import OllamaBackend  # noqa: E402

MODEL_NAME = "qwen3.5:2b"
_HEURISTIC_CHARS_PER_TOKEN = 4

_backend = OllamaBackend()
_availability_checked = False
_available = False


def _ollama_available() -> bool:
    global _availability_checked, _available
    if not _availability_checked:
        try:
            _backend.complete(MODEL_NAME, "ping", GenOpts(max_tokens=1))
            _available = True
        except Exception:
            _available = False
        _availability_checked = True
    return _available


def whole_file_tokens(text: str) -> tuple[int, str]:
    """(token_count, method) for a whole file's text.

    method is "ollama-tokenizer" (real prompt_eval_count from the settled
    policy model) or "heuristic-chars/4" when Ollama can't be reached.
    """
    if _ollama_available():
        result = _backend.complete(MODEL_NAME, text, GenOpts(max_tokens=1))
        if result.prompt_tokens:
            return result.prompt_tokens, "ollama-tokenizer"
    return max(1, len(text) // _HEURISTIC_CHARS_PER_TOKEN), "heuristic-chars/4"


def summarize_events(events: list[dict]) -> dict:
    """Decision points, retries, and total output tokens from a trace."""
    decision_points = sum(1 for e in events if e.get("attempt") == 0)
    total_attempts = len(events)
    out_tokens = sum(int(e.get("out_tokens") or 0) for e in events)
    prompt_tokens = sum(int(e.get("prompt_tokens") or 0) for e in events)
    return {
        "decision_points": decision_points,
        "total_attempts": total_attempts,
        "retries": total_attempts - decision_points,
        "output_tokens": out_tokens,
        "prompt_tokens": prompt_tokens,
    }


def delta_economics(output_tokens: int, whole_file_token_count: int) -> dict:
    ratio = output_tokens / whole_file_token_count if whole_file_token_count else 0.0
    return {
        "output_tokens": output_tokens,
        "whole_file_tokens": whole_file_token_count,
        "ratio": round(ratio, 4),
        "savings_pct": round((1 - ratio) * 100, 1) if whole_file_token_count else 0.0,
    }


if __name__ == "__main__":
    events = [
        {"attempt": 0, "out_tokens": 3, "prompt_tokens": 40},
        {"attempt": 1, "out_tokens": 2, "prompt_tokens": 40},   # retry
        {"attempt": 0, "out_tokens": 5, "prompt_tokens": 60},
    ]
    summary = summarize_events(events)
    assert summary["decision_points"] == 2, summary
    assert summary["total_attempts"] == 3
    assert summary["retries"] == 1
    assert summary["output_tokens"] == 10

    econ = delta_economics(10, 200)
    assert econ["ratio"] == 0.05
    assert econ["savings_pct"] == 95.0

    tokens, method = whole_file_tokens("def f():\n    return 1\n")
    assert tokens > 0 and method in {"ollama-tokenizer", "heuristic-chars/4"}
    print(f"(measured via {method})")
    print("smoke OK")
