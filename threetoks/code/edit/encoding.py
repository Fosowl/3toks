"""Source bytes in and out for the edit vertical and the code agent.

``decode_source`` reads file bytes the way Python does (BOM, coding
cookie) as LF text; ``encode_like`` writes new LF text back with the
original's line endings, trailing newline run and encoding.
"""
import io
import tokenize

DEFAULT_ENCODING = "utf-8"
LF = "\n"
CRLF = "\r\n"
CR = "\r"


def encode_like(original: bytes, new_text: str) -> bytes:
    """Encode new file text with the conventions of the original bytes.

    The code agent calls this before it writes an accepted edit. The line
    ending (LF, CRLF or CR) is the first one of the original. The run of
    newlines at the end is the original's run, so b"a\\n\\n" keeps two.
    The encoding is the one Python reads the original with: utf-8,
    utf-8 with BOM, or a coding cookie.

    Args:
        original: bytes, the file bytes read from disk.
        new_text: str, the candidate file text (LF), from the vertical.

    Returns:
        bytes: the bytes to write. Raises UnicodeEncodeError when the
        text does not fit a non-utf-8 coding cookie.
    """
    text = new_text.rstrip(LF) + LF * _trailing_newlines(original)
    newline = _newline_of(original)
    if newline != LF:
        text = text.replace(LF, newline)
    return text.encode(_source_encoding(original) or DEFAULT_ENCODING)


def decode_source(raw: bytes) -> str | None:
    """Decode source bytes the way Python reads them, as LF text.

    The symbol index uses it on every file, and the code agent on the
    bytes it reads again before a write. A BOM and a coding cookie are
    honoured; CRLF and CR become LF. encode_like is the inverse.

    Args:
        raw: bytes, the file bytes read from disk.

    Returns:
        str | None: the text, or None when it cannot be decoded.
    """
    encoding = _source_encoding(raw)
    if encoding is None:
        return None
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        return None
    return text.replace(CRLF, LF).replace(CR, LF)


def _newline_of(data: bytes) -> str:
    """Give the first line ending of a file.

    Args:
        data: bytes, the original file bytes.

    Returns:
        str: CRLF, CR, or LF (also when the file has no line ending).
    """
    cr, lf = data.find(CR.encode()), data.find(LF.encode())
    if cr < 0 or 0 <= lf < cr:
        return LF
    return CRLF if lf == cr + 1 else CR


def _trailing_newlines(data: bytes) -> int:
    """Count the line endings at the end of a file.

    Args:
        data: bytes, the original file bytes.

    Returns:
        int: how many line endings (of any style) end the file.
    """
    flat = data.replace(CRLF.encode(), LF.encode()).replace(CR.encode(),
                                                            LF.encode())
    return len(flat) - len(flat.rstrip(LF.encode()))


def _source_encoding(data: bytes) -> str | None:
    """Give the encoding Python uses to read a source file.

    Args:
        data: bytes, the original file bytes.

    Returns:
        str | None: the codec name ("utf-8-sig" keeps a BOM), or None
        when the BOM, cookie or first lines are not valid.
    """
    try:
        return tokenize.detect_encoding(io.BytesIO(data).readline)[0]
    except SyntaxError:
        return None


if __name__ == "__main__":
    raw = b"\xef\xbb\xbfa = 1\r\n\r\n"
    assert encode_like(raw, decode_source(raw)) == raw
    assert encode_like(b"a\rb", "a\nc\n") == b"a\rc"
    assert encode_like(b"a\n\n", "b\n") == b"b\n\n"
    assert decode_source(b"\xff\xfe = 1\n") is None
    print("smoke OK")
