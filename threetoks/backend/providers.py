"""Chat-API transports: OpenAI-compatible services and Anthropic (stdlib only).

Chat providers are SUPPORTED, NOT RECOMMENDED — same footing as non-qwen
template families. Raw mode's assistant prefill is what makes 1–3-token
decisions reliable, and a chat API cannot pre-seed the assistant turn:
the OpenAI-shaped path degrades to a prefill *hint* in the user turn plus
stripping any echoed prefill from the reply. Anthropic is the exception —
its Messages API honours a trailing assistant message as a true prefill.
Decisions made over a chat transport are traced with ``mode: chat`` so a
degraded-mode failure is diagnosable, never mysterious.

API keys come from environment variables only, never from config files.
KV-cache append-only discipline has no client-side equivalent here; every
call re-sends the episode at the provider's price.
"""
import json
import os
import time
import urllib.error
import urllib.request

from threetoks.backend.base import ChatPrompt, GenOpts, GenResult
from threetoks.backend.ollama import DEFAULT_HOST, OllamaBackend

DEFAULT_TIMEOUT_S = 120
MAX_OPENAI_STOPS = 4  # the OpenAI API rejects more than four stop strings
ERROR_BODY_CLIP = 200
PREFILL_HINT = 'Begin your reply with "{prefill}".'
ANTHROPIC_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

# provider -> (chat-completions base URL, API-key env var, key required).
# "custom" points anywhere OpenAI-shaped via [llm] api_base.
OPENAI_COMPAT = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", True),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", True),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", True),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", True),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai",
               "GOOGLE_API_KEY", True),
    "lm-studio": ("http://localhost:1234/v1", "LLM_API_KEY", False),
    "custom": ("", "LLM_API_KEY", False),
}
PROVIDERS = ("ollama", "anthropic") + tuple(OPENAI_COMPAT)


def make_backend(llm):
    """The configured transport for an ``LlmConfig``-shaped object.

    ``ollama`` gives the raw backend; every other provider name gives a
    chat backend. Raises RuntimeError for an unknown provider, a missing
    API key, or ``custom`` without ``api_base``.
    """
    if llm.provider == "ollama":
        return OllamaBackend(host=llm.host or DEFAULT_HOST)
    if llm.provider == "anthropic":
        return AnthropicBackend(_require_key("ANTHROPIC_API_KEY", "anthropic"),
                                base_url=llm.api_base or ANTHROPIC_BASE)
    if llm.provider not in OPENAI_COMPAT:
        raise RuntimeError(f"unknown llm provider '{llm.provider}' — "
                           f"choose from: {', '.join(PROVIDERS)}")
    default_base, key_env, key_required = OPENAI_COMPAT[llm.provider]
    base_url = llm.api_base or default_base
    if not base_url:
        raise RuntimeError("provider 'custom' needs api_base under [llm] "
                           "in config.ini")
    key = _require_key(key_env, llm.provider) if key_required \
        else os.environ.get(key_env, "")
    return OpenAIChatBackend(base_url, api_key=key)


def _require_key(env_var: str, provider: str) -> str:
    """The API key from ``env_var``, or a clear error naming it."""
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    raise RuntimeError(f"provider '{provider}' needs the {env_var} "
                       "environment variable")


def _http_post_json(url: str, headers: dict, payload: dict,
                    timeout_s: int) -> dict:
    """POST JSON and decode the reply; HTTP errors carry the body text."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read()[:ERROR_BODY_CLIP].decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") \
            from error
    except urllib.error.URLError as error:  # DNS, refused, timeout
        raise RuntimeError(f"cannot reach {url}: {error.reason}") from error


def _hinted_user(prompt: ChatPrompt) -> str:
    """User text plus a one-line prefill hint (chat has no true prefill)."""
    if not prompt.prefill:
        return prompt.user
    return f"{prompt.user}\n{PREFILL_HINT.format(prefill=prompt.prefill)}"


def _strip_prefill(text: str, prefill: str) -> str:
    """Drop an echoed prefill so parsing sees raw-mode-shaped text."""
    if prefill and text.lstrip().startswith(prefill):
        return text.lstrip()[len(prefill):]
    return text


def _openai_content(text: str, images: tuple):
    """Plain string, or typed content parts when images ride along."""
    if not images:
        return text
    parts = [{"type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
             for image in images]
    return [{"type": "text", "text": text}] + parts


class OpenAIChatBackend:
    """ChatBackend over any /chat/completions-shaped API.

    One class covers openai/openrouter/together/deepseek/google-compat/
    lm-studio/custom — they differ only in base URL and key env var (see
    OPENAI_COMPAT). Prefill is hinted and stripped, never native.
    """

    def __init__(self, base_url: str, api_key: str = "",
                 timeout_s: int = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat(self, model: str, prompt: ChatPrompt, opts: GenOpts) -> GenResult:
        """Run one chat completion and map it onto GenResult."""
        start = time.time()
        data = self._post("/chat/completions", self._payload(model, prompt,
                                                             opts))
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        msg = choice.get("message") or {}
        raw_text = msg.get("content") or msg.get("reasoning_content") or ""
        return GenResult(text=_strip_prefill(raw_text, prompt.prefill),
                         prompt_tokens=usage.get("prompt_tokens", 0),
                         out_tokens=usage.get("completion_tokens", 0),
                         wall_s=time.time() - start,
                         done_reason=choice.get("finish_reason") or "",
                         raw=data)

    def _payload(self, model: str, prompt: ChatPrompt, opts: GenOpts) -> dict:
        """Request body; stops capped at four, num_ctx has no analogue."""
        messages = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user",
                         "content": _openai_content(_hinted_user(prompt),
                                                    opts.images)})
        payload = {"model": model, "messages": messages,
                   "max_tokens": opts.max_tokens,
                   "temperature": opts.temperature}
        if opts.stop:
            payload["stop"] = list(opts.stop[:MAX_OPENAI_STOPS])
        if opts.seed is not None:
            payload["seed"] = opts.seed
        if "openrouter.ai" in self.base_url:
            payload["include_reasoning"] = False
        return payload

    def _post(self, path: str, payload: dict) -> dict:
        """POST to the service; bearer auth only when a key is set."""
        headers = {"Authorization": f"Bearer {self.api_key}"} \
            if self.api_key else {}
        return _http_post_json(self.base_url + path, headers, payload,
                               self.timeout_s)


def _anthropic_content(text: str, images: tuple):
    """Plain string, or image source blocks followed by the text."""
    if not images:
        return text
    parts = [{"type": "image", "source": {"type": "base64",
                                          "media_type": "image/jpeg",
                                          "data": image}}
             for image in images]
    return parts + [{"type": "text", "text": text}]


class AnthropicBackend:
    """ChatBackend over the Anthropic Messages API with NATIVE prefill.

    A trailing assistant message pre-seeds the reply exactly like raw
    mode, so this is the chat provider that keeps the measured prefill
    discipline intact (no hint, no stripping needed).
    """

    def __init__(self, api_key: str, base_url: str = ANTHROPIC_BASE,
                 timeout_s: int = DEFAULT_TIMEOUT_S):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def chat(self, model: str, prompt: ChatPrompt, opts: GenOpts) -> GenResult:
        """Run one message turn and map it onto GenResult."""
        start = time.time()
        data = self._post("/v1/messages", self._payload(model, prompt, opts))
        usage = data.get("usage") or {}
        text = "".join(block.get("text", "")
                       for block in data.get("content") or [])
        return GenResult(text=text,
                         prompt_tokens=usage.get("input_tokens", 0),
                         out_tokens=usage.get("output_tokens", 0),
                         wall_s=time.time() - start,
                         done_reason=data.get("stop_reason") or "", raw=data)

    def _payload(self, model: str, prompt: ChatPrompt, opts: GenOpts) -> dict:
        """Request body; prefill rides as a trailing assistant message.

        The API rejects whitespace-only stop sequences (a node's "\\n"
        stop), so those are dropped — first-line parsing and the token
        cap already bound those nodes.
        """
        messages = [{"role": "user",
                     "content": _anthropic_content(prompt.user, opts.images)}]
        if prompt.prefill.strip():  # trailing whitespace is rejected too
            messages.append({"role": "assistant",
                             "content": prompt.prefill.rstrip()})
        payload = {"model": model, "messages": messages,
                   "max_tokens": opts.max_tokens,
                   "temperature": opts.temperature}
        if prompt.system:
            payload["system"] = prompt.system
        stops = [stop for stop in opts.stop if stop.strip()]
        if stops:
            payload["stop_sequences"] = stops
        return payload

    def _post(self, path: str, payload: dict) -> dict:
        """POST to the Messages API with Anthropic's auth headers."""
        headers = {"x-api-key": self.api_key,
                   "anthropic-version": ANTHROPIC_VERSION}
        return _http_post_json(self.base_url + path, headers, payload,
                               self.timeout_s)


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _Llm:
        provider: str
        model: str = "m"
        host: str = ""
        api_base: str = ""

    class _CannedOpenAI(OpenAIChatBackend):
        def _post(self, path, payload):
            self.sent = (path, payload)
            return {"choices": [{"message": {"content": "ANSWER: 2"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 3}}

    backend = _CannedOpenAI("https://api.example.test/v1", api_key="k")
    result = backend.chat("m", ChatPrompt("sys", "pick one", "ANSWER:"),
                          GenOpts(max_tokens=3, stop=("\n", "a", "b", "c",
                                                      "d")))
    assert result.text == " 2" and result.out_tokens == 3, result
    path, payload = backend.sent
    assert path == "/chat/completions"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert 'Begin your reply with "ANSWER:"' in payload["messages"][1]["content"]
    assert len(payload["stop"]) == MAX_OPENAI_STOPS

    class _CannedAnthropic(AnthropicBackend):
        def _post(self, path, payload):
            self.sent = (path, payload)
            return {"content": [{"type": "text", "text": " 1"}],
                    "usage": {"input_tokens": 9, "output_tokens": 2},
                    "stop_reason": "end_turn"}

    claude = _CannedAnthropic("k")
    result = claude.chat("m", ChatPrompt("sys", "pick one", "ANSWER:"),
                         GenOpts(max_tokens=3, stop=("\n",)))
    assert result.text == " 1" and result.prompt_tokens == 9, result
    path, payload = claude.sent
    assert payload["messages"][-1] == {"role": "assistant",
                                       "content": "ANSWER:"}
    assert payload["system"] == "sys" and "stop_sequences" not in payload

    assert isinstance(make_backend(_Llm("ollama", host="")), OllamaBackend)
    os.environ.pop("LLM_API_KEY", None)
    assert isinstance(make_backend(_Llm("lm-studio")), OpenAIChatBackend)
    for bad in (_Llm("custom"), _Llm("nope")):
        try:
            make_backend(bad)
            raise AssertionError(f"{bad.provider} should have raised")
        except RuntimeError:
            pass
    os.environ.pop("OPENAI_API_KEY", None)
    try:
        make_backend(_Llm("openai"))
        raise AssertionError("missing key should have raised")
    except RuntimeError as error:
        assert "OPENAI_API_KEY" in str(error)
    print("smoke OK")
