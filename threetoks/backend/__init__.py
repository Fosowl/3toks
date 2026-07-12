"""LLM backends: prompt templating per model family + transport adapters."""
from threetoks.backend.base import GenOpts, GenResult, LLMBackend, ModelSpec, build_raw_prompt
from threetoks.backend.ollama import OllamaBackend

__all__ = ["GenOpts", "GenResult", "LLMBackend", "ModelSpec",
           "build_raw_prompt", "OllamaBackend"]
