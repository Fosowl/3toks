"""Ollama transport adapter (stdlib only, blocking).

KV-cache reuse is implicit: Ollama re-uses the prompt cache when a new
prompt shares a prefix with the previous one, which the append-only
renderer guarantees (see threetoks/render.py).
"""
import json
import time
import urllib.request

from threetoks.backend.base import GenOpts, GenResult

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 300


class OllamaBackend:
    """LLMBackend implementation over the Ollama HTTP API."""

    def __init__(self, host: str = DEFAULT_HOST,
                 timeout_s: int = DEFAULT_TIMEOUT_S):
        self.host = host
        self.timeout_s = timeout_s

    def complete(self, model: str, raw_prompt: str, opts: GenOpts) -> GenResult:
        """Run one raw-mode completion against /api/generate."""
        options = {"num_predict": opts.max_tokens,
                   "temperature": opts.temperature, "num_ctx": opts.num_ctx}
        if opts.stop:
            options["stop"] = list(opts.stop)
        if opts.seed is not None:
            options["seed"] = opts.seed
        payload = {"model": model, "prompt": raw_prompt, "raw": True,
                   "stream": False, "options": options}
        start = time.time()
        data = self._post("/api/generate", payload)
        return GenResult(text=data.get("response", ""),
                         prompt_tokens=data.get("prompt_eval_count", 0),
                         out_tokens=data.get("eval_count", 0),
                         wall_s=time.time() - start,
                         done_reason=data.get("done_reason", ""), raw=data)

    def _post(self, path: str, payload: dict) -> dict:
        """POST JSON to the Ollama API and decode the response."""
        req = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())


if __name__ == "__main__":
    from threetoks.backend.base import FAMILY_R1, ModelSpec, build_raw_prompt
    spec = ModelSpec("deepseek-r1:1.5b", FAMILY_R1)
    prompt = build_raw_prompt(
        spec, "ACTIONS:\n1 = continue\n2 = stop\nReply with ONE digit.",
        prefill="ANSWER:")
    result = OllamaBackend().complete(spec.name, prompt, GenOpts(max_tokens=3))
    print(f"got {result.text!r} in {result.wall_s:.2f}s")
    assert result.text.strip(), "empty completion"
    print("smoke OK")
