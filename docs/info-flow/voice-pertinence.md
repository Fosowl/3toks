# Info flow: voice pertinence gate

How a microphone transcript is judged "meant for the assistant" before
any agent runs. Code: `threetoks/voice/stt.py` (`is_pertinent`),
composed in `threetoks/voice/session.py` (`VoiceSession.listen`).

## Pipeline (cheapest gate first)

1. **Vosk transcript** — raw text from `SpeechListener.listen`. Zero
   model cost.
2. **Deterministic gates** (`gate_transcript`, zero model cost):
   empty → drop; whole text in `KNOWN_HALLUCINATIONS` → drop; echo of
   the assistant's own last `SPOKEN_HISTORY` answers
   (`threetoks/voice/echo.py`, n-gram window both directions) → drop.
   It returns the drop reason (`GATE_EMPTY`/`GATE_HALLUCINATION`/
   `GATE_ECHO`/`GATE_OK`) so `/voice debug` can name it;
   `accept_transcript` is the reason-less wrapper.
3. **Pertinence menu** (`is_pertinent`, ~1 output token): one
   `MenuNode` with two semantically described options, `escape=False`,
   decided by the session policy.

## Source → prompt map for step 3

| Piece | Value |
|---|---|
| episode system | `_GATE_PREFIX` — frames the always-listening-mic setting and the fail-closed instruction |
| episode task | `_GATE_TASK` (constant) |
| question | `Which best describes this transcript? "<transcript>"` — the transcript is the ONLY runtime data in the prompt |
| options | `DIRECTED_OPTION` / `NOISE_OPTION` (two described options, never bare yes/no — house rule 3) |

## Failure behavior

Invalid parse after the retry ladder, or the noise option → `False`
(fail closed, matching the reference project's default-to-GIBBERISH).
The spoken-history list feeding the echo filter is appended by
`VoiceSession.speak` BEFORE audio playback, so a transcript of our own
answer can never race past the filter. The reference project's second
"fix the misheard text" LLM call was deliberately not ported (it was
persona-specific and doubles cost per utterance).
