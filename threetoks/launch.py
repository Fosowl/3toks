"""The ``threetoks`` command: flags, ``config`` tools, and ``doctor``.

``threetoks`` alone opens the chat. ``threetoks config`` opens the config
screen (or prints the config when there is no terminal), and
``config path|show|get|set`` work in scripts. ``threetoks doctor`` checks
Ollama, the model, keys, SearXNG, paths, and extras, and exits 1 when
something is broken.
"""
import argparse
import importlib.util
import os
import sys

from threetoks.config import (CONFIG_ENV_VAR, ThreetoksConfig, load_config,
                              resolve_path)
from threetoks.hints import LEVEL_BAD
from threetoks.fields import (FIELDS, SECTIONS, config_warnings,
                              default_value, fields_of, find_field,
                              format_value, get_value, parse_value,
                              serialize_value, set_value)

EXIT_BROKEN = 1        # the file cannot be used, or a check failed
EXIT_BAD_INPUT = 2
CONFIG_ACTIONS = ("path", "show", "get", "set", "edit")
# (module to import, pip extra)
EXTRAS = (("curl_cffi", "web"), ("selenium", "browser"), ("vosk", "stt"),
          ("piper", "tts"), ("cv2", "camera"))
SHOW_NAME_W = 26       # `config show` key column
DOCTOR_NAME_W, DOCTOR_VALUE_W = 24, 32
RULE_WIDTH = 60        # doctor wave lines


def _version() -> str:
    """Return the installed version, or the package constant if not installed.

    Returns: str, for example ``0.2.0``.
    """
    from importlib import metadata
    try:
        return metadata.version("3toks")
    except metadata.PackageNotFoundError:
        from threetoks import __version__
        return __version__


def _add_config_parser(commands) -> None:
    """Add the ``config`` command and its actions to the sub-parsers."""
    config = commands.add_parser("config", help="open or edit the config")
    config.add_argument("action", nargs="?", choices=CONFIG_ACTIONS,
                        help="default: the screen (or show, with no terminal)")
    config.add_argument("key", nargs="?", help="section.key, e.g. llm.model")
    config.add_argument("value", nargs="?", help="new value (set only)")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the ``threetoks`` command.

    Returns: argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="threetoks", description="Token-frugal agents for small models.")
    parser.add_argument("--version", action="version",
                        version=f"threetoks {_version()}")
    parser.add_argument("-m", "--model", help="model for this session only")
    parser.add_argument("-c", "--config", metavar="PATH",
                        help="use this config file (sets $THREETOKS_CONFIG)")
    parser.add_argument("--no-color", action="store_true",
                        help="plain text, no colors")
    commands = parser.add_subparsers(dest="command")
    _add_config_parser(commands)
    commands.add_parser("doctor", help="check the setup and say what to fix")
    return parser


def _style():
    """Return the colour helpers and the on/off flag.

    Returns: (fg, dim, enabled) from the TUI module.
    """
    from threetoks.theme import color_enabled, dim, fg
    return fg, dim, color_enabled()


def _config_path(args) -> int:
    """Print the config file that is read, and whether it exists."""
    path = resolve_path()
    print(path + ("" if os.path.isfile(path) else "  (not created yet)"))
    return 0


def _show_section(section, config: ThreetoksConfig) -> None:
    """Print one section: harbor title, then each key and value."""
    fg, dim, on = _style()
    from threetoks.theme import CYAN, PINK, VIOLET
    title = fg(CYAN, f" ◈ {section.title}", on)
    print(title + dim(f"  {section.tagline}", on))
    for item in fields_of(section.name):
        value = get_value(config, item)
        color = VIOLET if value == default_value(item) else PINK
        print("   " + dim(item.dotted.ljust(SHOW_NAME_W), on)
              + fg(color, format_value(item, value), on))


def _warn_on_stderr() -> None:
    """Print what in the config file is not used, on stderr.

    ``config show`` and ``config get`` call this before their output, so
    a value that silently fell back to its default is visible.
    """
    for line in config_warnings():
        print(line, file=sys.stderr)


def _config_show(args) -> int:
    """Print every setting, grouped by harbor section; pink = not default."""
    _warn_on_stderr()
    config = load_config()
    for section in SECTIONS:
        _show_section(section, config)
    return 0


def _lookup(key: str | None):
    """Find a field by name; print the error and return None if unknown."""
    item = find_field(key or "")
    if item is None:
        print(f"unknown setting: {key!r} (see: threetoks config show)",
              file=sys.stderr)
    return item


def _config_get(args) -> int:
    """Print one value, in the form it has in the INI file."""
    item = _lookup(args.key)
    if item is None:
        return EXIT_BAD_INPUT
    _warn_on_stderr()
    print(serialize_value(item, get_value(load_config(), item)))
    return 0


def _checked_value(args) -> tuple:
    """Parse the ``config set`` value; print why when it is bad.

    Returns: (Field, value), or (None, None) after an error message.
    """
    item = _lookup(args.key)
    if item is None:
        return None, None
    if args.value is None:
        print("usage: threetoks config set section.key VALUE",
              file=sys.stderr)
        return None, None
    value, problem = parse_value(item, args.value)
    if problem:
        print(f"{item.dotted}: {problem}", file=sys.stderr)
        return None, None
    return item, value


def _config_set(args) -> int:
    """Check and save one value; a running chat applies it at once.

    A value the loader ignored is named on stderr first: the save
    writes its default in its place.
    """
    item, value = _checked_value(args)
    if item is None:
        return EXIT_BAD_INPUT
    _warn_on_stderr()
    from threetoks.ini_edit import save_changes
    try:
        path = save_changes(set_value(load_config(), item, value))
    except (OSError, ValueError) as error:  # unwritable, or a broken file
        print(f"cannot save: {error}", file=sys.stderr)
        return EXIT_BROKEN
    print(f"{item.dotted} = {serialize_value(item, value)} -> {path}")
    return 0


def _config_edit(args) -> int:
    """Open the config screen; without a terminal, print the config."""
    from threetoks.keys import is_interactive
    if not is_interactive():
        return _config_show(args)
    from threetoks.config_ui import run_screen
    from threetoks.theme import color_enabled
    run_screen(color_enabled())
    return 0


_CONFIG_RUNNERS = {"path": _config_path, "show": _config_show,
                   "get": _config_get, "set": _config_set,
                   "edit": _config_edit, None: _config_edit}


def _doctor_rows(config: ThreetoksConfig) -> list:
    """Return (field, hint) pairs for every setting that has a verdict."""
    from threetoks.hints import hint_for
    from threetoks.probes import sweep
    probes = sweep(config)
    pairs = ((item, hint_for(item, config, probes)) for item in FIELDS)
    return [(item, hint) for item, hint in pairs if hint is not None]


def _print_doctor_row(item, hint, config: ThreetoksConfig) -> None:
    """Print one verdict line: glyph, setting, value, hint text."""
    from threetoks.screen_view import LEVEL_STYLE
    fg, dim, on = _style()
    color, mark = LEVEL_STYLE[hint.level]
    value = format_value(item, get_value(config, item))
    shown = value[:DOCTOR_VALUE_W - 2].ljust(DOCTOR_VALUE_W)
    print(f" {fg(color, mark, on)} {item.dotted.ljust(DOCTOR_NAME_W)}"
          f"{dim(shown, on)}{fg(color, hint.text, on)}")


def _extras_line() -> str:
    """One text line: each pip extra with a tick or a dash."""
    parts = []
    for module, extra in EXTRAS:
        found = importlib.util.find_spec(module) is not None
        parts.append(f"{extra} {'✓' if found else '–'}")
    return "extras: " + "  ".join(parts)


def _config_doctor(args) -> int:
    """Check the setup; return 1 when a check is bad."""
    from threetoks.screen_view import wave_line
    fg, dim, on = _style()
    config = load_config()
    print(wave_line(RULE_WIDTH, 0, on))
    print(dim(f" config: {resolve_path()}", on))
    warnings = config_warnings()
    for line in warnings:
        print(line)
    rows = _doctor_rows(config)
    for item, hint in rows:
        _print_doctor_row(item, hint, config)
    print(dim(" " + _extras_line(), on))
    print(wave_line(RULE_WIDTH, 1, on))
    broken = any(hint.level == LEVEL_BAD for _, hint in rows)
    return EXIT_BROKEN if broken or warnings else 0


def main(argv=None) -> int:
    """Run the ``threetoks`` command with ``argv`` (default: the real ones).

    Used by the console script and ``python -m threetoks``. Args: argv
    (list of str or None): arguments without the program name. Returns:
    int, the exit code.
    """
    args = build_parser().parse_args(argv)
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
    if args.config:
        os.environ[CONFIG_ENV_VAR] = args.config
    if args.command == "config":
        return _CONFIG_RUNNERS[args.action](args)
    if args.command == "doctor":
        return _config_doctor(args)
    from threetoks.tui import main as chat
    chat(args.model)
    return 0


if __name__ == "__main__":
    parsed = build_parser().parse_args(["config", "get", "llm.model"])
    assert parsed.action == "get" and parsed.key == "llm.model"
    assert build_parser().parse_args(["-m", "x"]).model == "x"
    assert main(["config", "get", "llm.model"]) == 0
    assert main(["config", "get", "nope.nope"]) == EXIT_BAD_INPUT
    print("smoke OK")
