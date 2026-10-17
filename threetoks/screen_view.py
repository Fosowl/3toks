"""Draw the config screen: a neon harbor-control panel (stdlib ANSI only).

:func:`render` turns a :class:`~threetoks.screen_state.Screen` into a list
of text lines. It reads the screen and draws; it never changes it and
never touches the terminal, so tests can call it and read the lines.
Theme: tide lines of waves that run from deep teal to neon pink, and
harbor names for the sections (HELM, NETS, SONAR, ...).
"""
from threetoks.fields import (APPLY_LIVE, APPLY_RESTART, APPLY_VOICE,
                              KIND_BOOL, KIND_TEXT, SECTIONS, Field, fields_of,
                              format_value, get_value)
from threetoks.hints import (LEVEL_BAD, LEVEL_OK, LEVEL_WARN, Hint,
                             hint_for)
from threetoks.screen_state import (Screen, current_field, unsaved)
from threetoks.theme import (AMBER, CROSS, CYAN, GRAY, GRAY_LIGHT, GREEN, NODE,
                             PINK, SPINE, VIOLET, WHITE, ZAP, dim, fg,
                             gradient)

# Deep sea teal, then the neon-dawn ramp: a sunset over dark water.
SEA_RAMP = (23, 30, 37, 44, 51, 81, 111, 141, 171, 205)
WAVES = ("≋≈", "≈∿")
TITLE = "HARBOR CONTROL"
MAX_WIDTH = 78
MIN_WIDTH = 64
PATH_MIN_ROOM = 8     # least columns kept for the file path
TAB_MARGIN = 4        # columns the full tab bar needs beside its tabs
LABEL_W, VALUE_W, TAG_W = 18, 30, 10
MAX_ROWS = max(len(fields_of(section.name)) for section in SECTIONS)
LEVEL_STYLE = {LEVEL_OK: (GREEN, "●"), LEVEL_WARN: (AMBER, "▲"),
                LEVEL_BAD: (PINK, CROSS)}
_APPLY_STYLE = {APPLY_LIVE: GREEN, APPLY_RESTART: GRAY, APPLY_VOICE: VIOLET}
_BROWSE_KEYS = (("↑↓", "move"), ("←→", "change"), ("⏎", "edit"),
                ("⇥", "section"), ("r", "reset"), ("s", "save"),
                ("esc", "leave"))
_EDIT_KEYS = (("type", "value"), ("⏎", "accept"), ("esc", "cancel"),
              ("^U", "clear"))


def clip(text: str, width: int) -> str:
    """Cut ``text`` to ``width`` characters, ending in an ellipsis.

    Args: text (str): plain text; width (int): the most characters.
    Returns: str.
    """
    return text if len(text) <= width else text[:max(width - 1, 0)] + "…"


def wave_line(width: int, phase: int, enabled: bool) -> str:
    """One tide line: waves that run from deep teal to neon pink.

    Args: width (int): characters; phase (int): 0 or 1, picks the wave
    pattern; enabled (bool): colors on. Returns: str.
    """
    pattern = WAVES[phase % len(WAVES)]
    return gradient((pattern * width)[:width], enabled, SEA_RAMP)


def header_line(screen: Screen, width: int, enabled: bool) -> str:
    """The title row: harbor name on the left, the file path on the right.

    Args: screen (Screen); width (int); enabled (bool). Returns: str.
    """
    left = f" {NODE} {TITLE} {ZAP} config"
    room = max(width - len(left) - 2, PATH_MIN_ROOM)
    path = screen.path
    if len(path) > room:
        path = "…" + path[-(room - 1):]
    gap = " " * max(width - len(left) - len(path) - 1, 1)
    title = gradient(f" {NODE} {TITLE}", enabled, SEA_RAMP)
    tag = fg(VIOLET, f" {ZAP} config", enabled)
    return title + tag + gap + dim(path, enabled)


def _dirty_sections(screen: Screen) -> set:
    """Names of the sections that hold an unsaved change."""
    return {item.section for item in unsaved(screen)}


def _tab(index: int, screen: Screen, enabled: bool) -> str:
    """One tab label; the active one has a node, edited ones a dot."""
    section = SECTIONS[index]
    mark = "•" if section.name in _dirty_sections(screen) else ""
    if index == screen.section:
        label = fg(CYAN, f"{NODE} {section.title}", enabled)
        return label + fg(PINK, mark, enabled)
    return dim(section.title, enabled) + fg(PINK, mark, enabled)


def tab_bar(screen: Screen, width: int, enabled: bool) -> str:
    """The row of section tabs; on a narrow screen only the active one.

    Args: screen (Screen); width (int); enabled (bool). Returns: str.
    """
    full = "  ".join(SECTIONS[i].title for i in range(len(SECTIONS)))
    if len(full) + TAB_MARGIN > width:
        title = SECTIONS[screen.section].title
        place = f"{screen.section + 1}/{len(SECTIONS)}"
        return " " + fg(CYAN, f"‹ {place} {title} ›", enabled)
    tabs = (_tab(i, screen, enabled) for i in range(len(SECTIONS)))
    return " " + "  ".join(tabs)


def _cell(text: str, style: tuple, enabled: bool) -> str:
    """Clip and pad ``text`` to the style's width, then color it.

    Args: text (str); style (tuple): (color code, width); enabled (bool).
    Returns: str.
    """
    color, width = style
    return fg(color, clip(text, width).ljust(width), enabled)


def _value_cell(screen: Screen, field: Field, enabled: bool) -> str:
    """The value column: box while editing, arrows on a cycling row."""
    selected = field == current_field(screen)
    if selected and screen.edit is not None:
        shown = (screen.edit + "█")[-VALUE_W:]
        return _cell(shown, (CYAN, VALUE_W), enabled)
    value = get_value(screen.config, field)
    text = format_value(field, value)
    if field.kind == KIND_BOOL:
        text = ("● " if value else "○ ") + text
    if selected and field.kind != KIND_TEXT:
        text = f"‹ {text} ›"
    edited = value != get_value(screen.saved, field)
    return _cell(text, (PINK if edited else VIOLET, VALUE_W), enabled)


def _glyph(hint: Hint | None, enabled: bool) -> str:
    """The one-character health marker at the end of a row."""
    if hint is None:
        return " "
    color, mark = LEVEL_STYLE[hint.level]
    return fg(color, mark, enabled)


def row_line(screen: Screen, field: Field, enabled: bool) -> str:
    """One setting row: marker, label, value, apply tag, health glyph.

    Args: screen (Screen); field (Field); enabled (bool). Returns: str.
    """
    selected = field == current_field(screen)
    marker = fg(CYAN, "▸", enabled) if selected else " "
    label = _cell(field.label, (WHITE if selected else GRAY_LIGHT, LABEL_W),
                  enabled)
    tag = _cell(field.apply, (_APPLY_STYLE[field.apply], TAG_W), enabled)
    hint = hint_for(field, screen.config, screen.probes)
    value = _value_cell(screen, field, enabled)
    return f" {marker} {label}{value}{tag}{_glyph(hint, enabled)}"


def section_title(screen: Screen, enabled: bool) -> str:
    """The sub-title of the open section: spine, harbor name, tagline.

    Args: screen (Screen); enabled (bool). Returns: str.
    """
    section = SECTIONS[screen.section]
    return (" " + fg(VIOLET, SPINE, enabled) + " "
            + fg(CYAN, section.title, enabled)
            + dim(f"  ·  {section.tagline}", enabled))


def detail_lines(screen: Screen, enabled: bool) -> list:
    """Help text and health verdict of the row under the cursor.

    Args: screen (Screen); enabled (bool). Returns: list of 2 str.
    """
    field = current_field(screen)
    hint = hint_for(field, screen.config, screen.probes)
    verdict = ""
    if hint is not None:
        color, mark = LEVEL_STYLE[hint.level]
        verdict = fg(color, f"{mark} {hint.text}", enabled)
    return [dim(f"   {field.help}", enabled), "   " + verdict]


def message_line(screen: Screen, enabled: bool) -> str:
    """The one-line notice (saved, error, warning), or an empty line.

    Args: screen (Screen); enabled (bool). Returns: str.
    """
    if not screen.message:
        return ""
    color, mark = LEVEL_STYLE[screen.level]
    return " " + fg(color, f"{mark} {screen.message}", enabled)


def key_bar(screen: Screen, width: int, enabled: bool) -> str:
    """The key help row and the unsaved counter on the right.

    Args: screen (Screen); width (int); enabled (bool). Returns: str.
    """
    pairs = _EDIT_KEYS if screen.edit is not None else _BROWSE_KEYS
    plain = " " + "  ".join(f"{key} {name}" for key, name in pairs)
    drawn = " " + "  ".join(fg(CYAN, key, enabled) + dim(" " + name, enabled)
                            for key, name in pairs)
    count = len(unsaved(screen))
    status = f"◉ {count} unsaved" if count else "✓ in sync"
    gap = " " * max(width - len(plain) - len(status) - 1, 1)
    return drawn + gap + fg(PINK if count else GREEN, status, enabled)


def render(screen: Screen, enabled: bool = True,
           width: int = MAX_WIDTH) -> list:
    """Draw the whole screen as a list of lines.

    Used by the session loop each frame and by tests. Args: screen
    (Screen): the state to draw; enabled (bool): colors on; width (int):
    columns to use. Returns: list of str, always the same height.
    """
    width = max(MIN_WIDTH, min(width, MAX_WIDTH))
    rows = [row_line(screen, item, enabled)
            for item in fields_of(SECTIONS[screen.section].name)]
    rows += [""] * (MAX_ROWS - len(rows))
    return ([wave_line(width, 0, enabled), header_line(screen, width, enabled),
             tab_bar(screen, width, enabled), "",
             section_title(screen, enabled)] + rows + [""]
            + detail_lines(screen, enabled)
            + [message_line(screen, enabled), "",
               key_bar(screen, width, enabled), wave_line(width, 1, enabled)])


if __name__ == "__main__":
    from threetoks.config import ThreetoksConfig
    from threetoks.probes import Probes
    demo = Screen(ThreetoksConfig(), ThreetoksConfig(), Probes(), "~/c.ini")
    frame = render(demo, enabled=False)
    assert any("HARBOR CONTROL" in line for line in frame), frame
    assert len(frame) == len(render(demo, False, 70)), "height must not jump"
    print("\n".join(render(demo, enabled=True)))
    print("smoke OK")
