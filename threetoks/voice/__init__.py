"""Optional voice layer: speech-to-text in, text-to-speech out.

Both directions are optional extras (``pip install '3toks[stt]'`` /
``'3toks[tts]'``) and degrade independently: STT alone gives voice input
with typed output, TTS alone speaks answers to typed input. Third-party
imports stay inside functions so a bare install still imports this package.
"""
