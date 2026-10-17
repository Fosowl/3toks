"""Save config changes into the existing INI file without losing comments.

The old save wrote the whole file again from a template, so hand-written
comments and key order were lost. This module changes only the
``key = value`` lines that differ, and keeps every other line as it was.
It reads lines the way ``configparser`` does (indented keys, headers with
trailing text, continuation lines), so a saved file always loads back.
A missing file is created from the commented template instead.
"""
import os
import shutil
import tempfile

from threetoks.config import (ThreetoksConfig, load_config, resolve_path,
                              save_config)
from threetoks.fields import changed_fields, get_value, serialize_value

_DELIMITERS = "=:"
_COMMENT_MARKS = ("#", ";")
_CRLF, _LF = "\r\n", "\n"


def _section_name(line: str) -> str | None:
    """Return the section name of a ``[header]`` line, else None.

    Like configparser: the name runs to the last ``]``, text after it is
    ignored, and the case is kept (the loader is case-sensitive).
    Args: line (str): one raw INI line. Returns: the name or None.
    """
    text = line.strip()
    end = text.rfind("]")
    if not text.startswith("[") or end < 2:
        return None
    return text[1:end]


def _is_comment(line: str) -> bool:
    """True when ``line`` is a whole-line comment."""
    return line.lstrip().startswith(_COMMENT_MARKS)


def _delimiter_at(line: str) -> int:
    """Return the index of the first ``=`` or ``:`` in ``line``, or -1.

    Args: line (str): one raw INI line. Returns: int index.
    """
    spots = [line.find(mark) for mark in _DELIMITERS if mark in line]
    return min(spots) if spots else -1


def _key_name(line: str) -> str | None:
    """Return the key of a ``key = value`` line, else None.

    Blank lines, comments and headers have no key. Indented keys count.
    Args: line (str): one raw INI line. Returns: lowercase key or None.
    """
    if not line.strip() or _is_comment(line) or _section_name(line):
        return None
    spot = _delimiter_at(line)
    return line[:spot].strip().lower() if spot > 0 else None


def _with_value(line: str, value: str) -> str:
    """Return ``line`` with its value replaced, keeping key and delimiter.

    Args: line (str): the old line; value (str): the new text.
    Returns: the new line.
    """
    spot = _delimiter_at(line)
    return f"{line[:spot].rstrip()} {line[spot]} {value}"


def _indent(line: str) -> int:
    """Count the leading white space of ``line``."""
    return len(line) - len(line.lstrip())


class _Scan:
    """Walks the lines once and rewrites the keys that must change.

    ``out`` is the new text, ``last`` maps a section to the index of its
    last header or key line, ``pending`` holds changes that found no line.
    """

    def __init__(self, changes: dict) -> None:
        """Start with the wanted ``changes``: (section, key) -> text."""
        self.out, self.last, self.pending = [], {}, dict(changes)
        self.section = ""
        self.in_value, self.key_indent, self.dropping = False, 0, False

    def _is_continuation(self, line: str) -> bool:
        """True when ``line`` continues the value of the key above it."""
        return (self.in_value and bool(line.strip()) and not _is_comment(line)
                and _indent(line) > self.key_indent)

    def _header(self, name: str, line: str) -> None:
        """Enter a new section."""
        self.section, self.in_value = name, False
        self.last[name] = len(self.out)
        self.out.append(line)

    def _key(self, key: str, line: str) -> None:
        """Keep or rewrite one key line, and start tracking its value."""
        self.in_value, self.key_indent = True, _indent(line)
        self.dropping = (self.section, key) in self.pending
        if self.dropping:
            line = _with_value(line, self.pending.pop((self.section, key)))
        self.last[self.section] = len(self.out)
        self.out.append(line)

    def feed(self, line: str) -> None:
        """Handle one raw line."""
        if self._is_continuation(line):
            if not self.dropping:
                self.out.append(line)
            return
        name, key = _section_name(line), _key_name(line)
        if name is not None:
            self._header(name, line)
        elif key is not None:
            self._key(key, line)
        else:
            self.out.append(line)


def _insert_missing(scan: _Scan) -> list:
    """Add unmatched changes: into their section, or in a new section.

    Args: scan (_Scan): the finished scan. Returns: the new lines.
    """
    out = scan.out
    known = [(s, k, v) for (s, k), v in scan.pending.items()
             if s in scan.last]
    known.sort(key=lambda row: -scan.last[row[0]])   # bottom-up, stable
    for section, key, value in known:
        out.insert(scan.last[section] + 1, f"{key} = {value}")
    opened = set()
    for (section, key), value in scan.pending.items():
        if section in scan.last:
            continue
        while out and not out[-1].strip() and section not in opened:
            out.pop()
        head = [] if section in opened else ["", f"[{section}]"]
        opened.add(section)
        out.extend(head + [f"{key} = {value}"])
    return out


def update_ini(text: str, changes: dict) -> str:
    """Return ``text`` with ``changes`` applied and all else kept.

    Used by :func:`save_changes` and tests. Args: text (str): the file
    content; changes (dict): (section, key) -> new value text.
    Returns: the new file content, ending in one newline.
    """
    crlf = _CRLF in text
    scan = _Scan(changes)
    for line in text.replace(_CRLF, _LF).split(_LF):
        scan.feed(line)
    result = _LF.join(_insert_missing(scan)).rstrip(_LF) + _LF
    return result.replace(_LF, _CRLF) if crlf else result


def _atomic_write(path: str, text: str) -> None:
    """Write ``text`` to ``path`` through a temp file and a rename.

    A symlinked ``path`` is followed, so the real file is the one
    replaced; its permissions are kept. Args: path (str): the target;
    text (str): the content. Returns: None.
    """
    real = os.path.realpath(path)
    handle, temp = tempfile.mkstemp(dir=os.path.dirname(real), prefix=".tt-")
    with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
        stream.write(text)
    shutil.copymode(real, temp)
    os.replace(temp, real)


def _changes_against(old: ThreetoksConfig, new: ThreetoksConfig) -> dict:
    """Map each changed field to its INI text.

    Args: old, new (ThreetoksConfig). Returns: dict (section, key) -> str.
    """
    return {(item.section, item.key): serialize_value(item,
                                                      get_value(new, item))
            for item in changed_fields(old, new)}


def save_changes(new: ThreetoksConfig, path: str = None) -> str:
    """Save ``new`` into the active config file, changing only what differs.

    Used by the config screen and ``threetoks config set``. A missing
    file is created from the commented template. Args: new
    (ThreetoksConfig): the wanted config; path (str): the file, default
    the file the loader reads. Returns: the path written.
    """
    target = path or resolve_path()
    if not os.path.isfile(target):
        return save_config(new, target)
    changes = _changes_against(load_config(target), new)
    if not changes:
        return target
    with open(target, encoding="utf-8", newline="") as stream:
        text = stream.read()
    _atomic_write(target, update_ini(text, changes))
    return target


if __name__ == "__main__":
    sample = ("# keep me\n[llm]\nmodel: old\n  more\n# note\nhost = h\n\n"
              "[browser]\nmode = http\n")
    done = update_ini(sample, {("llm", "model"): "new",
                               ("llm", "vote_k"): "2",
                               ("browser", "mode"): "plain",
                               ("voice", "lang"): "fr"})
    assert "# keep me" in done and "# note" in done and "more" not in done
    assert "model : new" in done and "vote_k = 2" in done, done
    assert done.rstrip().endswith("[voice]\nlang = fr"), done
    print("smoke OK")
