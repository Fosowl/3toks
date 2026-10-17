"""Startup banner: a flying fish and its sea shadow drawn as a "3".

The fish leaps head first: forked tail at the waist, where it left the
water, back arcing up the right side and over the top, head at the top
left tip with its mouth open and its eye on the upper side. That arc is
the upper bowl of the 3. Under the horizon its shadow is the same shape
mirrored and shaded, so the two halves read as one glyph. The veined
pectoral wing, drawn in light shades so the water shows through, sweeps
back under the body; a small pelvic fin sits near the tail. The TOKS
wordmark stands on the horizon and casts a shorter reflection that fades
with depth.

Every row is pure text; :func:`build_banner` adds the color.
"""
import random

from threetoks.theme import GRAY_LIGHT, ZAP, dim, fg, gradient, hairline

# Upper bowl of the 3. Open mouth: thin upper jaw ▀, longer lower jaw ◢▄;
# eye ▞◉▚; body █; see-through wing and pelvic fin ░▒▓ with ribs; forked
# tail ◢◤ on the water.
FISH = (
    "         ▄▄▄▄▟███████▙▄      ",
    "     ▀▀▀█▞◉▚███████████▙▄    ",
    "   ◢▄▄▄▄███████████████▙     ",
    "         ▓▓▓▒▓▒░▒░   ▜████▙  ",
    "           ▓▓▒▓▒░▒░   ▜███▌  ",
    "             ▓▒▓▒░▒░  ▐███▌  ",
    "         ░▒    ▒▓▒░▒  ▟███▛  ",
    "          ░▒▓▓   ▒░  ▟███▛   ",
    "    ◢███████████████████▛    ",
    "   ◢▛▀▀▀▀▜█████████████▛▀    ",
    "   ◤      ▀▀▀▜███████▛▀      ",
)
# Wordmark; :func:`banner_rows` stands it on the horizon beside the fish.
WORDMARK = (
    "▀▀▀████▀▀▀  ▄██████▄  ████  ▄██▀  ▄███████▄",
    "   ████    ████  ████ ████ ▄██▀  ████    ▀▀",
    "   ████    ████  ████ ████▄██▀   ████      ",
    "   ████    ████  ████ ███████▄    ▀██████▄ ",
    "   ████    ████  ████ ████ ▀██▄        ▀███",
    "   ████    ████  ████ ████  ▀██▄ ▄▄    ▄███",
    "   ████     ▀██████▀  ████   ▀██  ▀██████▀ ",
)
GAP = "  "
INDENT = "  "
HORIZON = "▔"
WIDTH = len(INDENT) + len(FISH[0]) + len(GAP) + len(WORDMARK[0])
FULL = "█▙▜▟▛▌▐"              # solid glyphs: take the depth shade in water
HALF = "▀▄▘▗▝▖"               # thin edges: one step lighter than the row
LADDER = "█▓▒░ "              # shade steps, dark to gone
MIRROR = {"▄": "▀", "▀": "▄", "▙": "▜", "▜": "▙", "▟": "▛", "▛": "▟",
          "◣": "◤", "◤": "◣", "◥": "◢", "◢": "◥", "▞": "▚", "▚": "▞",
          "▘": "▗", "▗": "▘", "▝": "▖", "▖": "▝", "◉": "○"}
FISH_SHADES = "▓▓▒"            # shadow shade per depth row; stays visible
WORDMARK_SHADES = "▒░░ "       # reflection fades to nothing by row four
WATER_DEPTH = 4                # ramp colors the water keeps (dark end)
SPLASH = "˚∘·"
SPLASH_ROWS = (6, 7, 9, 10)    # fish rows whose blank lead gets a droplet
TAGLINE = "3 tokens per action or bust — local · frugal · fast"

# Launch ramps: build_banner draws one per call; all run dark to neon so
# the art brightens rightward and the water keeps the dark end.
BANNER_RAMPS = (
    (54, 55, 56, 92, 93, 129),     # midnight: deep purple → electric violet
    (19, 25, 26, 32, 38, 44),      # blade rain: navy → azure → turquoise
    (23, 29, 30, 36, 37, 43),      # deep sea: dark teal → aqua
    (53, 89, 125, 161, 162, 197),  # noir crimson: wine → neon crimson
    (22, 28, 29, 35, 41, 42),      # ghost circuit: dark green → sea green
)


def _lighter(shade: str) -> str:
    """The next step of ``LADDER`` after ``shade`` (blank stays blank)."""
    return LADDER[min(LADDER.index(shade) + 1, len(LADDER) - 1)]


def _sink(char: str, shade: str) -> str:
    """One glyph as seen in the water at a row whose shade is ``shade``."""
    char = MIRROR.get(char, char)
    if char in FULL:
        return shade
    if char in HALF:
        return _lighter(shade)
    return char


def reflect(rows, shades: str) -> list:
    """Mirror ``rows`` top-to-bottom into the water.

    Args:
        rows: Art rows, top first.
        shades: One shade glyph per depth row; the last one repeats.

    Returns:
        The rows in reverse order, each glyph flipped through ``MIRROR``
        and solid glyphs replaced by that row's shade.
    """
    deep = []
    for depth, row in enumerate(reversed(rows)):
        shade = shades[min(depth, len(shades) - 1)]
        deep.append("".join(_sink(char, shade) for char in row))
    return deep


def _splashed(rows, rng) -> list:
    """Copy of ``rows`` with one droplet in the blank lead of each
    ``SPLASH_ROWS`` row, where the tail left the water."""
    out = list(rows)
    for index in SPLASH_ROWS:
        row = out[index]
        blank = len(row) - len(row.lstrip())
        column = rng.randrange(1, blank - 1)
        out[index] = row[:column] + rng.choice(SPLASH) + row[column + 1:]
    return out


def _standing_wordmark() -> list:
    """``WORDMARK`` padded with blank rows so it sits on the horizon."""
    blank = " " * len(WORDMARK[0])
    return [blank] * (len(FISH) - len(WORDMARK)) + list(WORDMARK)


def banner_rows(rng) -> tuple:
    """The plain art: ``(sky_rows, horizon, sea_rows)``, all one width."""
    wordmark = _standing_wordmark()
    sky = [fish + GAP + word
           for fish, word in zip(_splashed(FISH, rng), wordmark)]
    sea = [fish + GAP + word
           for fish, word in zip(reflect(FISH, FISH_SHADES),
                                 reflect(wordmark, WORDMARK_SHADES))]
    return sky, HORIZON * len(sky[0]), sea


def build_banner(model: str, agent_count: int, enabled: bool = True,
                 rng=None) -> str:
    """The startup banner between two hairline rules.

    Args:
        model: Model name shown in the meta line.
        agent_count: Number of agents shown in the meta line.
        enabled: False renders plain text with no escapes.
        rng: Seeded ``random.Random`` for a reproducible banner; each call
            otherwise draws a fresh ramp from ``BANNER_RAMPS`` and splash.

    Returns:
        The banner text, ``WIDTH`` columns wide. The sky uses the whole
        ramp; the horizon and the water stop at ``WATER_DEPTH`` of it, so
        the shadow reads as a reflection yet stays legible on black.
    """
    rng = rng or random.Random()
    ramp = rng.choice(BANNER_RAMPS)
    water = ramp[:WATER_DEPTH]
    sky, horizon, sea = banner_rows(rng)
    meta = f"{ZAP} {model} · {agent_count} agents online"
    return "\n".join([
        hairline(6, enabled, ramp, WIDTH),
        *(INDENT + gradient(row, enabled, ramp) for row in sky),
        INDENT + gradient(horizon, enabled, water),
        *(INDENT + gradient(row, enabled, water) for row in sea),
        INDENT + dim(TAGLINE, enabled),
        INDENT + fg(GRAY_LIGHT, meta, enabled),
        hairline(WIDTH - 7, enabled, ramp, WIDTH),
    ])


if __name__ == "__main__":
    print(build_banner("qwen3.5:2b", 4))
