"""State and key handling of the config screen (no drawing, no terminal).

:func:`handle_key` takes one key name, changes the :class:`Screen`, and
returns an action for the session loop: ``save``, ``quit``, ``probe`` or
None. Because nothing here touches the terminal, tests can feed it a
list of keys.
"""
from dataclasses import dataclass

from threetoks.config import ThreetoksConfig
from threetoks.fields import (KIND_BOOL, KIND_CHOICE, KIND_TEXT, SECTIONS,
                              changed_fields, cycle_value, default_value,
                              fields_of, format_value, get_value,
                              parse_value, serialize_value, set_value)
from threetoks.hints import LEVEL_BAD, LEVEL_OK, LEVEL_WARN
from threetoks.probes import Probes

ACTION_SAVE = "save"
ACTION_QUIT = "quit"
ACTION_PROBE = "probe"

# Changing one of these makes the old probe answers stale.
PROBE_FIELDS = frozenset({"llm.provider", "llm.host", "search.searxng_url"})

_ESC_KEY, _QUIT_KEY = "esc", "q"
_DIGIT_KEYS = "0123456789"
_TENTH_SECTION = 9   # the 0 key jumps to the tenth section


@dataclass
class Screen:
    """Everything the screen shows and edits.

    ``config`` is the working copy, ``saved`` is what is on disk.
    ``edit`` is the text box buffer, or None while browsing.
    """

    config: ThreetoksConfig
    saved: ThreetoksConfig
    probes: Probes
    path: str
    section: int = 0
    row: int = 0
    edit: str | None = None
    message: str = ""
    level: str = LEVEL_OK
    confirm_quit: bool = False


def current_section(screen: Screen):
    """Return the Section the cursor is in.

    Args: screen (Screen). Returns: Section.
    """
    return SECTIONS[screen.section]


def current_field(screen: Screen):
    """Return the Field under the cursor.

    Args: screen (Screen). Returns: Field.
    """
    return fields_of(current_section(screen).name)[screen.row]


def unsaved(screen: Screen) -> tuple:
    """List the fields that differ from the saved file.

    Args: screen (Screen). Returns: tuple of Field.
    """
    return changed_fields(screen.saved, screen.config)


def _say(screen: Screen, level: str, text: str) -> None:
    """Set the one-line notice under the rows."""
    screen.level, screen.message = level, text


def _assign(screen: Screen, field, value) -> str | None:
    """Store ``value`` in the working copy; ask for a probe when stale."""
    screen.config = set_value(screen.config, field, value)
    return ACTION_PROBE if field.dotted in PROBE_FIELDS else None


def suggestions(screen: Screen, field) -> tuple:
    """Return values that left and right can cycle for a text field.

    Only the Ollama model has suggestions: the pulled models.
    Args: screen (Screen); field (Field). Returns: tuple of str.
    """
    if field.dotted == "llm.model" and screen.config.llm.provider == "ollama":
        return screen.probes.models or ()
    return ()


def _move(screen: Screen, step: int) -> None:
    """Move the cursor ``step`` rows, wrapping round the section."""
    count = len(fields_of(current_section(screen).name))
    screen.row = (screen.row + step) % count


def _switch(screen: Screen, step: int) -> None:
    """Jump ``step`` sections along, wrapping, and reset the row."""
    screen.section = (screen.section + step) % len(SECTIONS)
    screen.row = 0


def _suggest(screen: Screen, field, step: int) -> str | None:
    """Cycle a text field through its suggestions, if there are any."""
    options = suggestions(screen, field)
    if not options:
        _say(screen, LEVEL_WARN, "free text: press enter to type")
        return None
    now = get_value(screen.config, field)
    index = options.index(now) if now in options else -step
    return _assign(screen, field, options[(index + step) % len(options)])


def _nudge(screen: Screen, step: int) -> str | None:
    """Left or right key: cycle the value of the row under the cursor."""
    field = current_field(screen)
    if field.kind == KIND_TEXT:
        return _suggest(screen, field, step)
    value = cycle_value(field, get_value(screen.config, field), step)
    return _assign(screen, field, value)


def _activate(screen: Screen) -> str | None:
    """Enter key: flip bools and choices, open the text box otherwise."""
    field = current_field(screen)
    if field.kind in (KIND_BOOL, KIND_CHOICE):
        return _nudge(screen, 1)
    screen.edit = serialize_value(field, get_value(screen.config, field))
    return None


def _reset(screen: Screen) -> str | None:
    """R key: put the row back to the built-in default."""
    field = current_field(screen)
    _say(screen, LEVEL_OK, f"{field.label} back to default")
    return _assign(screen, field, default_value(field))


def _leave(screen: Screen) -> str | None:
    """Esc key: quit, but warn once when there are unsaved changes."""
    if not unsaved(screen) or screen.confirm_quit:
        return ACTION_QUIT
    screen.confirm_quit = True
    _say(screen, LEVEL_WARN, "unsaved changes: esc again to drop, s to save")
    return None


_BROWSE = {
    "up": lambda s: _move(s, -1), "k": lambda s: _move(s, -1),
    "down": lambda s: _move(s, 1), "j": lambda s: _move(s, 1),
    "tab": lambda s: _switch(s, 1), "pgdn": lambda s: _switch(s, 1),
    "btab": lambda s: _switch(s, -1), "pgup": lambda s: _switch(s, -1),
    "left": lambda s: _nudge(s, -1), "h": lambda s: _nudge(s, -1),
    "right": lambda s: _nudge(s, 1), "l": lambda s: _nudge(s, 1),
    " ": lambda s: _nudge(s, 1), "enter": _activate, "r": _reset,
    "s": lambda s: ACTION_SAVE, "ctrl-s": lambda s: ACTION_SAVE,
    "p": lambda s: ACTION_PROBE, "esc": _leave, "q": _leave,
}


def _jump_to_section(screen: Screen, key: str) -> None:
    """Digit keys 1..9 and 0 jump straight to a section."""
    index = _TENTH_SECTION if key == "0" else int(key) - 1
    if index < len(SECTIONS):
        screen.section, screen.row = index, 0


def _commit_edit(screen: Screen) -> str | None:
    """Enter in the text box: check the text, store it or show why not."""
    field = current_field(screen)
    value, problem = parse_value(field, screen.edit)
    if problem:
        _say(screen, LEVEL_BAD, f"{field.label}: {problem}")
        return None
    screen.edit = None
    _say(screen, LEVEL_OK, f"{field.label} = {format_value(field, value)}")
    return _assign(screen, field, value)


def _edit_key(screen: Screen, key: str) -> str | None:
    """Handle one key while the text box is open."""
    if key == "enter":
        return _commit_edit(screen)
    if key == "esc":
        screen.edit = None
    elif key == "backspace":
        screen.edit = screen.edit[:-1]
    elif key in ("ctrl-u", "ctrl-w"):
        screen.edit = ""
    elif len(key) == 1 and key.isprintable():
        screen.edit += key
    return None


def handle_key(screen: Screen, key: str) -> str | None:
    """Apply one key press to the screen.

    Used by the session loop with real keys and by tests with scripted
    keys. Args: screen (Screen): changed in place; key (str): a name from
    :mod:`threetoks.keys` or one character. Returns: ``save``, ``quit``,
    ``probe`` or None.
    """
    if screen.edit is not None:
        return _edit_key(screen, key)
    if key not in (_ESC_KEY, _QUIT_KEY):
        screen.confirm_quit = False
    if len(key) == 1 and key in _DIGIT_KEYS:
        _jump_to_section(screen, key)
        return None
    handler = _BROWSE.get(key)
    return handler(screen) if handler else None


if __name__ == "__main__":
    demo = Screen(ThreetoksConfig(), ThreetoksConfig(), Probes(), "x.ini")
    for press in ("down", "enter", "backspace", "backspace", "z", "enter"):
        handle_key(demo, press)
    assert demo.config.llm.model == "qwen3.5:z", demo.config.llm
    assert handle_key(demo, "esc") is None and demo.confirm_quit
    assert handle_key(demo, "esc") == ACTION_QUIT
    print("smoke OK")
