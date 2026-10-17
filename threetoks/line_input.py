"""Line editing for the chat prompt: history, arrow keys, Tab completion.

The REPL used plain ``input()``. With ``readline`` loaded, ``input()`` also
gives history, cursor movement, and Tab completion of ``/commands`` and
their arguments. Windows has no ``readline``; there the reader stays
plain ``input``. History is kept between sessions in the config folder.
"""
import atexit
import os

from threetoks.config import user_config_dir

HISTORY_NAME = "history"
HISTORY_LIMIT = 500
_ESC = "\x1b"
_MARK_ON, _MARK_OFF = "\x01", "\x02"   # readline: "zero-width text"

# Arguments each command accepts, offered after the command and a space.
ARG_CHOICES = {
    "/browser": ("http", "plain", "http on-screen", "plain on-screen",
                 "plain headless"),
    "/debug": ("on", "off", "full"),
    "/voice": ("on", "off", "debug"),
}


def wrap_prompt(prompt: str) -> str:
    """Mark colour escapes in ``prompt`` as zero-width for readline.

    Without the marks readline counts the escape bytes as visible text
    and breaks the cursor position on long lines. Args: prompt (str):
    the prompt, with ANSI colour codes. Returns: str, same text with each
    ``ESC[...m`` code wrapped in ``\\x01`` and ``\\x02``.
    """
    out, index = [], 0
    while index < len(prompt):
        if prompt[index] != _ESC:
            out.append(prompt[index])
            index += 1
            continue
        end = prompt.find("m", index)
        end = len(prompt) - 1 if end < 0 else end
        out.append(_MARK_ON + prompt[index:end + 1] + _MARK_OFF)
        index = end + 1
    return "".join(out)


def candidates(buffer: str, commands) -> list:
    """List the completions for what is typed so far.

    Used by the Tab key. Args: buffer (str): the whole line typed so far;
    commands (iterable of str): known ``/command`` names. Returns: list
    of full replacements for the word being typed, sorted. Empty when
    the line is not a slash command.
    """
    if not buffer.startswith("/"):
        return []
    verb, space, rest = buffer.partition(" ")
    if not space:
        return sorted(name for name in commands if name.startswith(verb))
    options = ARG_CHOICES.get(verb.lower(), ())
    return sorted(item for item in options if item.startswith(rest))


def _current_words(matches: list, buffer: str) -> list:
    """Reduce each match to the word being typed; drop duplicates.

    Args: matches (list of str): from :func:`candidates`; buffer (str):
    the line typed so far. Returns: list of str, one word per match.
    """
    done = buffer.partition(" ")[2].count(" ")
    return list(dict.fromkeys(item.split(" ")[done] for item in matches))


def make_completer(commands, get_buffer):
    """Build the function readline calls for each Tab press.

    Args: commands (iterable of str): known ``/command`` names;
    get_buffer (callable): returns the current line text. Returns: a
    ``completer(text, state)`` function as readline expects.
    """
    names = tuple(commands)

    def completer(text: str, state: int):
        """Return the ``state``-th completion of ``text``, or None."""
        buffer = get_buffer()
        found = _current_words(candidates(buffer, names), buffer)
        found = [item for item in found if item.startswith(text)]
        return found[state] if state < len(found) else None
    return completer


def history_path() -> str:
    """Return the history file path in the user config folder.

    Returns: str.
    """
    return os.path.join(user_config_dir(), HISTORY_NAME)


def _is_libedit(readline) -> bool:
    """True for macOS libedit, which ignores the zero-width prompt marks."""
    return "libedit" in (readline.__doc__ or "")


def _bind_tab(readline) -> None:
    """Bind Tab to complete, in both GNU readline and macOS libedit."""
    if _is_libedit(readline):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def _load_history(readline, path: str) -> None:
    """Read old history and arrange to save it at exit; never raise."""
    try:
        readline.read_history_file(path)
    except OSError:
        pass
    readline.set_history_length(HISTORY_LIMIT)
    atexit.register(_save_history, readline, path)


def _save_history(readline, path: str) -> None:
    """Write history to ``path``; a read-only folder is not an error."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        readline.write_history_file(path)
    except OSError:
        pass


def make_reader(commands, path: str = None):
    """Build the ``prompt -> line`` function the REPL reads with.

    Used by the TUI start-up on a real terminal. Args: commands
    (iterable of str): ``/command`` names for Tab; path (str): history
    file, default in the config folder. Returns: a callable that takes a
    prompt and returns one typed line; plain ``input`` when readline is
    missing or is macOS libedit (which would drop the prompt colours).
    """
    try:
        import readline
    except ImportError:
        return input
    readline.set_completer_delims(" \t\n")
    readline.set_completer(make_completer(commands,
                                          readline.get_line_buffer))
    _bind_tab(readline)
    _load_history(readline, path or history_path())
    if _is_libedit(readline):  # libedit prints marked spans uncoloured
        return input
    return lambda prompt: input(wrap_prompt(prompt))


def terminal_reader(commands):
    """Readline reader on a real terminal, plain ``input`` otherwise.

    Used by ``tui.main``. Args: commands (iterable of str): ``/command``
    names for Tab. Returns: a callable that takes a prompt and returns a line.
    """
    from threetoks.keys import is_interactive
    return make_reader(commands) if is_interactive() else input


if __name__ == "__main__":
    cmds = ("/help", "/hello", "/model", "/browser")
    assert candidates("/he", cmds) == ["/hello", "/help"]
    assert candidates("hi", cmds) == []
    assert candidates("/browser pl", cmds) == ["plain", "plain headless",
                                               "plain on-screen"]
    assert wrap_prompt("\x1b[38;5;51mx\x1b[0m") == (
        "\x01\x1b[38;5;51m\x02x\x01\x1b[0m\x02")
    assert wrap_prompt("plain") == "plain"
    print("smoke OK")
