"""Colors, glyphs, and tiny ANSI helpers shared by every screen.

The chat TUI, the config screen, and the CLI all draw with this palette,
so none of them has to import the 1400-line REPL module. ``NO_COLOR`` and
non-tty output fall back to plain text.
"""
import os
import sys

# --- palette (xterm-256) -------------------------------------------------
CYAN = 51        # decision values, prompt chevron
VIOLET = 141     # agent identity, headings
PINK = 205       # answer labels, slow decisions
GREEN = 84       # fast decisions
AMBER = 214      # warnings / errors
GRAY = 240       # structural chrome
GRAY_LIGHT = 246 # secondary text
WHITE = 253      # answer text

# Signature ramp: neon dawn, cyan → lavender → hot pink.
RAMP = (51, 81, 111, 141, 171, 205)
PANEL_WIDTH = 64  # result panels, hairline rules, and text wrapping

# --- glyphs --------------------------------------------------------------
SPINE = "▍"                 # ▍ panel gutter
NODE = "◈"                  # ◈ hairline node
ZAP = "⌁"                   # ⌁ mesh events
CHEVRON = "❯"               # ❯ prompt
NOTE_BULLET = "▪"           # ▪
CROSS = "✖"                 # ✖

_ESC = "\x1b"


def color_enabled(stream=None) -> bool:
    """True when ANSI color is appropriate for the given stream."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def fg(code: int, text: str, enabled: bool = True) -> str:
    """Wrap ``text`` in a 256-color foreground escape (or return it plain)."""
    if not enabled:
        return text
    return f"{_ESC}[38;5;{code}m{text}{_ESC}[0m"


def dim(text: str, enabled: bool = True) -> str:
    """Render ``text`` in dim gray, or plain when color is disabled."""
    return fg(GRAY, text, enabled)


def ramp_at(position: int, total: int, ramp=RAMP) -> int:
    """Color code for step ``position`` of ``total`` along ``ramp``."""
    if total <= 1:
        return ramp[0]
    return ramp[round(position * (len(ramp) - 1) / (total - 1))]


def gradient(text: str, enabled: bool = True, ramp=RAMP) -> str:
    """Color each character through ``ramp`` (plain when disabled)."""
    if not enabled:
        return text
    return "".join(fg(ramp_at(index, len(text), ramp), char)
                   for index, char in enumerate(text))


def hairline(node_at: int, enabled: bool = True, ramp=RAMP,
             width: int = PANEL_WIDTH) -> str:
    """A gradient hairline rule with a ◈ node at column ``node_at``."""
    line = "─" * width
    return gradient(line[:node_at] + NODE + line[node_at + 1:], enabled, ramp)


if __name__ == "__main__":
    assert fg(CYAN, "x", False) == "x"
    assert gradient("ab", False) == "ab"
    assert fg(CYAN, "x").startswith(_ESC)
    print("smoke OK")
