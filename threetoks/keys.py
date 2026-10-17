"""Read single key presses from the terminal, with no third-party package.

POSIX uses ``termios`` cbreak mode. Windows uses ``msvcrt``. A key comes
back as a short name (``up``, ``enter``, ``esc``, ``ctrl-s``...) or as the
typed character itself. Ctrl-C still raises ``KeyboardInterrupt``.
"""
import os
import sys

IS_WINDOWS = os.name == "nt"
ESCAPE_WAIT_S = 0.04          # a lone Esc key sends nothing after it
MAX_SEQUENCE = 8
ESC = "\x1b"

CONTROL_KEYS = {
    "\r": "enter", "\n": "enter", "\t": "tab", "\x7f": "backspace",
    "\x08": "backspace", "\x13": "ctrl-s", "\x15": "ctrl-u",
    "\x17": "ctrl-w", ESC: "esc",
}
ESCAPE_KEYS = {
    "[A": "up", "[B": "down", "[C": "right", "[D": "left", "[Z": "btab",
    "[5~": "pgup", "[6~": "pgdn", "OA": "up", "OB": "down", "OC": "right",
    "OD": "left",
}
WINDOWS_KEYS = {
    "H": "up", "P": "down", "K": "left", "M": "right", "I": "pgup",
    "Q": "pgdn", "\x0f": "btab",
}
_UTF8_LEADS = ((0xF0, 3), (0xE0, 2), (0xC0, 1))   # (lowest lead, extra)


def decode_escape(tail: str) -> str:
    """Name the key that sent ``ESC`` followed by ``tail``.

    Args: tail (str): the characters after ESC, for example ``[A``.
    Returns: a key name; ``esc`` for no tail, ``unknown`` for a stranger.
    """
    if not tail:
        return "esc"
    return ESCAPE_KEYS.get(tail, "unknown")


def is_interactive() -> bool:
    """True when stdin and stdout are a real terminal that can draw.

    Used before the config screen opens. Returns: bool.
    """
    dumb = os.environ.get("TERM") == "dumb"
    return sys.stdin.isatty() and sys.stdout.isatty() and not dumb


class Keyboard:
    """Context manager that reads one key per :meth:`read` call.

    Use ``with Keyboard() as keyboard: keyboard.read()``. The terminal
    mode is put back on exit, even after an error.
    """

    def __init__(self) -> None:
        """Remember the stdin file descriptor; nothing is changed yet."""
        self._fd = None if IS_WINDOWS else sys.stdin.fileno()
        self._saved = None
        self._pending = ""   # an ESC that arrived straight after an ESC

    def __enter__(self) -> "Keyboard":
        """Switch the terminal to cbreak mode (POSIX); no-op on Windows."""
        if not IS_WINDOWS:
            import termios
            import tty
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_exc) -> None:
        """Restore the terminal mode that was saved on entry."""
        if self._saved is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self) -> str:
        """Block until a key is pressed and return its name or character.

        Returns: str, for example ``up``, ``enter``, ``ctrl-s`` or ``a``.
        """
        if IS_WINDOWS:
            return self._read_windows()
        char = self._pending or self._read_char()
        self._pending = ""
        if char == ESC:
            return decode_escape(self._read_escape_tail())
        return CONTROL_KEYS.get(char, char)

    def _read_windows(self) -> str:
        """Read one key through msvcrt and name it."""
        import msvcrt
        char = msvcrt.getwch()
        if char == "\x03":
            raise KeyboardInterrupt
        if char in ("\x00", "\xe0"):
            return WINDOWS_KEYS.get(msvcrt.getwch(), "unknown")
        return CONTROL_KEYS.get(char, char)

    def _read_char(self) -> str:
        """Read one UTF-8 character from stdin."""
        first = os.read(self._fd, 1)
        extra = _utf8_extra_bytes(first[0]) if first else 0
        data = first + (os.read(self._fd, extra) if extra else b"")
        return data.decode("utf-8", errors="replace")

    def _ready(self) -> bool:
        """True when more input is waiting right after an ESC."""
        import select
        return bool(select.select([self._fd], [], [], ESCAPE_WAIT_S)[0])

    def _read_escape_tail(self) -> str:
        """Collect the characters of an escape sequence after ESC.

        A second ESC is not part of the sequence: it is kept for the
        next :meth:`read`, so two quick Esc presses stay two keys.
        """
        tail = ""
        while self._ready() and len(tail) < MAX_SEQUENCE:
            char = self._read_char()
            if char == ESC:
                self._pending = char
                break
            tail += char
            if _ends_sequence(tail, char):
                break
        return tail


def _utf8_extra_bytes(lead: int) -> int:
    """How many bytes follow a UTF-8 lead byte."""
    for lowest, count in _UTF8_LEADS:
        if lead >= lowest:
            return count
    return 0


def _ends_sequence(tail: str, char: str) -> bool:
    """True when ``char`` is the last character of an escape sequence."""
    if tail[0] == "O":
        return len(tail) == 2
    return len(tail) > 1 and (char.isalpha() or char == "~")


if __name__ == "__main__":
    assert decode_escape("") == "esc" and decode_escape("[A") == "up"
    assert decode_escape("[5~") == "pgup" and decode_escape("[9z") == "unknown"
    assert _utf8_extra_bytes("é".encode()[0]) == 1
    assert _ends_sequence("[5~", "~") and not _ends_sequence("[5", "5")
    print("smoke OK")
