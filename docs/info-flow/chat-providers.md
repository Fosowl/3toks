# Info flow: chat-API providers

What reaches the model when `[llm] provider` is not `ollama`, and where
each piece comes from. Transport code: `threetoks/backend/providers.py`;
dispatch: `Policy._call` in `threetoks/policy.py`.

## Source → prompt map

| Piece | Source | Reaches the chat API as |
|---|---|---|
| system text | `PolicyConfig.system` (cli.py `SYSTEM_CHATML`), or `node.system` override | `system` message (OpenAI-shaped) / top-level `system` (Anthropic) |
| episode + node | `episode.render_base() + node.render(perm)` — identical to raw mode | the single `user` message |
| prefill (`ANSWER:` / `RELEVANT:`) | `node.prefill` | Anthropic: trailing assistant message (native prefill). OpenAI-shaped: a one-line hint appended to the user text, and any echoed prefill is stripped from the reply |
| stops | `node.stop` ONLY (family end-of-turn stops are a raw-mode artifact) | `stop` (capped at 4, OpenAI) / `stop_sequences` (whitespace-only entries dropped — the API rejects them) |
| max_tokens / temperature | node + retry ladder, same as raw | passed through |
| seed | `GenOpts.seed` (the policy never sets it today) | forwarded when set (OpenAI-shaped only; Anthropic has no seed) |
| images | `node.images` (set by the `look` agent) | data-URI content part (OpenAI-shaped) / base64 source block (Anthropic) |
| API key | `{PROVIDER}_API_KEY` env var only | auth header; never logged, never in config files |

## Deliberately dropped

- `num_ctx` — no chat-API analogue; the provider owns context limits.
- Family templates and family stop tokens — the provider owns its template.
- R1 think phase — needs raw prompt surgery; silently skipped on chat.

Nothing else is dropped: every field the raw path sends has a mapped
equivalent above. Decisions taken over a chat transport carry
`mode: chat` in trace records so degraded prefill behavior is
distinguishable from raw-mode behavior when debugging accuracy.
