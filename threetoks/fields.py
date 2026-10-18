"""Field registry: one table that describes every config knob.

The config screen, the ``threetoks config`` command, and the doctor read
this table. It lists each knob once: its section, kind, label, and help
text. A test checks that the table covers every field of every config
dataclass, so a new knob cannot be forgotten. A running chat applies
every knob as soon as the file is saved (see :mod:`threetoks.live_config`).
"""
from dataclasses import dataclass, fields as dataclass_fields, replace

from threetoks.config import (BROWSER_MODES, MINIMUMS, PROVIDER_CHOICES,
                              ThreetoksConfig, config_problem, in_range,
                              read_parser)

KIND_CHOICE = "choice"
KIND_BOOL = "bool"
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_TEXT = "text"

_FLOAT_STEP = 0.5
_ON, _OFF = "on", "off"
_FLOAT_DIGITS = 6     # rounding after a step drops binary noise
_EMPTY_SHOWN = "(empty)"


@dataclass(frozen=True)
class Section:
    """One ``[section]`` of the INI file with its harbor-themed title."""

    name: str
    title: str
    tagline: str


@dataclass(frozen=True)
class Field:
    """One config knob: where it lives and how to edit it."""

    section: str
    key: str
    kind: str
    label: str
    help: str
    choices: tuple = ()

    @property
    def minimum(self) -> float | None:
        """The least accepted number, from ``config.MINIMUMS``, or None."""
        return MINIMUMS.get((self.section, self.key))

    @property
    def dotted(self) -> str:
        """The ``section.key`` name used by ``threetoks config get/set``."""
        return f"{self.section}.{self.key}"


SECTIONS = (
    Section("llm", "HELM", "model and transport"),
    Section("browser", "NETS", "how pages are fetched"),
    Section("search", "SONAR", "where searches go"),
    Section("research", "DEPTH", "how deep it dives"),
    Section("files", "DOCK", "where files live"),
    Section("memory", "LOGBOOK", "what it remembers"),
    Section("code", "FORGE", "the code agent"),
    Section("voice", "COMMS", "talk and listen"),
    Section("camera", "SCOPE", "periscope camera"),
    Section("relay", "BEACON", "lighthouse relay"),
)

FIELDS = (
    Field("llm", "provider", KIND_CHOICE, "provider",
          "ollama = local raw mode (measured). The rest are chat APIs.",
          choices=PROVIDER_CHOICES),
    Field("llm", "model", KIND_TEXT, "model",
          "Model that makes every decision node."),
    Field("llm", "host", KIND_TEXT, "ollama host",
          "Where Ollama listens (ollama provider only)."),
    Field("llm", "api_base", KIND_TEXT, "api base",
          "Overrides the provider URL. Required for custom."),
    Field("llm", "vote_k", KIND_INT, "vote k",
          "Votes per menu decision. 1 is the measured default."),
    Field("llm", "send_seed", KIND_BOOL, "send seed",
          "Off for servers that reject the sampling seed."),
    Field("llm", "disable_think", KIND_BOOL, "disable think",
          "Pre-close the think block. Thinking Qwen needs it on."),
    Field("browser", "mode", KIND_CHOICE, "fetch mode",
          "http = fast, no JavaScript. plain = real Chrome.",
          choices=BROWSER_MODES),
    Field("browser", "visible", KIND_BOOL, "show window",
          "Show the Chrome window (plain mode only)."),
    Field("search", "searxng_url", KIND_TEXT, "searxng url",
          "Base URL of the local SearXNG."),
    Field("search", "min_interval_s", KIND_FLOAT, "min interval s",
          "Least seconds between two outbound searches."),
    Field("research", "max_rounds", KIND_INT, "max rounds",
          "Judged research rounds per question."),
    Field("research", "max_steps", KIND_INT, "max steps",
          "Decision steps per research round."),
    Field("files", "root", KIND_TEXT, "root",
          "Directory the file and code agents work in."),
    Field("memory", "enabled", KIND_BOOL, "enabled",
          "Keep (query, answer) pairs between sessions."),
    Field("memory", "path", KIND_TEXT, "path", "Where memory is stored."),
    Field("code", "retrieval", KIND_BOOL, "retrieval",
          "EXPERIMENTAL: fetch and run code from the web."),
    Field("code", "snippet_cache", KIND_TEXT, "snippet cache",
          "File that caches retrieved snippets."),
    Field("voice", "enabled", KIND_BOOL, "enabled",
          "Listen and speak (at start, and now when saved)."),
    Field("voice", "lang", KIND_TEXT, "language",
          "Vosk language code, for example en-us."),
    Field("voice", "stt_model_path", KIND_TEXT, "stt model path",
          "Vosk model folder. Empty = auto-download."),
    Field("voice", "tts_model_path", KIND_TEXT, "tts model path",
          "Piper .onnx voice. Needs its .onnx.json beside it."),
    Field("voice", "tts_speaker", KIND_INT, "tts speaker",
          "Speaker id. -1 = the voice default."),
    Field("voice", "post_speak_delay", KIND_FLOAT, "post speak delay",
          "Seconds before the microphone unmutes."),
    Field("camera", "index", KIND_INT, "device index",
          "Camera device the look agent captures from."),
    Field("relay", "pin", KIND_INT, "gpio pin",
          "BCM pin of the active-low relay board."),
)


def fields_of(section: str) -> tuple:
    """Return the fields of one section, in display order.

    Used by the screen and the CLI. Args: section (str): INI section name.
    Returns: tuple of Field.
    """
    return tuple(item for item in FIELDS if item.section == section)


def find_field(dotted: str) -> Field | None:
    """Find a field by its ``section.key`` name, or None.

    Used by ``threetoks config get/set``. Args: dotted (str): the name,
    for example ``llm.model``. Returns: the Field, or None if unknown.
    """
    for item in FIELDS:
        if item.dotted == dotted.strip().lower():
            return item
    return None


def get_value(config: ThreetoksConfig, field: Field):
    """Read the current value of ``field`` from ``config``.

    Args: config (ThreetoksConfig): the config; field (Field): the knob.
    Returns: the value (str, int, float or bool).
    """
    return getattr(getattr(config, field.section), field.key)


def set_value(config: ThreetoksConfig, field: Field, value) -> ThreetoksConfig:
    """Return a copy of ``config`` with ``field`` set to ``value``.

    Args: config (ThreetoksConfig): the base; field (Field): the knob;
    value: the new value, already parsed. Returns: a new ThreetoksConfig.
    """
    section = replace(getattr(config, field.section), **{field.key: value})
    return replace(config, **{field.section: section})


def default_value(field: Field):
    """Return the built-in default of ``field``.

    Args: field (Field): the knob. Returns: the default value.
    """
    return get_value(ThreetoksConfig(), field)


def format_value(field: Field, value) -> str:
    """Show ``value`` the way the screen and ``config show`` print it.

    Args: field (Field): the knob; value: its current value.
    Returns: str, for example ``on`` or ``(empty)``.
    """
    if field.kind == KIND_BOOL:
        return _ON if value else _OFF
    return str(value) if str(value) != "" else _EMPTY_SHOWN


def serialize_value(field: Field, value) -> str:
    """Write ``value`` the way it is stored in the INI file.

    Args: field (Field): the knob; value: its current value.
    Returns: str, for example ``true`` or ``5.0``.
    """
    if field.kind == KIND_BOOL:
        return "true" if value else "false"
    return str(value)


def _parse_number(field: Field, text: str):
    """Parse ``text`` as the field's number kind, or raise ValueError."""
    return int(text) if field.kind == KIND_INT else float(text)


def _parse_text(field: Field, text: str) -> tuple:
    """Check a text value: one line, and not blank unless allowed."""
    if "\n" in text or "\r" in text:
        return None, "one line only"
    if not text and default_value(field):
        return None, "cannot be empty"
    return text, None


def parse_value(field: Field, text: str) -> tuple:
    """Turn typed ``text`` into a value for ``field``.

    Used by the edit box and ``config set``. Args: field (Field): the knob;
    text (str): what the user typed. Returns: (value, None) when it is
    good, or (None, message) when it is not.
    """
    text = text.strip()
    if field.kind == KIND_TEXT:
        return _parse_text(field, text)
    if field.kind == KIND_BOOL:
        return _parse_bool(text)
    if field.kind == KIND_CHOICE:
        ok = text.lower() in field.choices
        return (text.lower(), None) if ok else (None, "pick one of the list")
    return _parse_bounded(field, text)


def _parse_bool(text: str) -> tuple:
    """Parse on/off words; a bad word returns a message."""
    word = text.lower()
    if word in ("1", "true", "yes", "on"):
        return True, None
    if word in ("0", "false", "no", "off"):
        return False, None
    return None, "use on or off"


def _parse_bounded(field: Field, text: str) -> tuple:
    """Parse a number and check it is finite and not below the minimum."""
    try:
        value = _parse_number(field, text)
    except ValueError:
        return None, f"not a {field.kind}"
    if in_range(field.section, field.key, value):
        return value, None
    if field.minimum is not None and value < field.minimum:
        return None, f"at least {field.minimum:g}"
    return None, "not a finite number"


def cycle_value(field: Field, value, step: int):
    """Move ``value`` one step along the field's range (left or right key).

    Choice cycles round the list, bool flips, numbers move by 1 (0.5 for
    floats) and stop at the minimum. Text does not change.
    Args: field (Field): the knob; value: current value; step (int): -1/+1.
    Returns: the new value.
    """
    if field.kind == KIND_BOOL:
        return not value
    if field.kind == KIND_CHOICE:
        index = field.choices.index(value) if value in field.choices else 0
        return field.choices[(index + step) % len(field.choices)]
    if field.kind in (KIND_INT, KIND_FLOAT):
        return _step_number(field, value, step)
    return value


def _step_number(field: Field, value, step: int):
    """Add one step to a number and clamp to the minimum."""
    size = 1 if field.kind == KIND_INT else _FLOAT_STEP
    moved = round(value + step * size, _FLOAT_DIGITS)
    if field.minimum is not None:
        moved = max(field.minimum, moved)
    return int(moved) if field.kind == KIND_INT else float(moved)


def changed_fields(old: ThreetoksConfig, new: ThreetoksConfig) -> tuple:
    """List the fields whose value differs between two configs.

    Used by save and by the live-apply step. Args: old, new
    (ThreetoksConfig). Returns: tuple of Field, in table order.
    """
    return tuple(item for item in FIELDS
                 if get_value(old, item) != get_value(new, item))


def invalid_fields(parser) -> list:
    """List the keys whose text the loader ignores for its default.

    The warnings below and the INI saver call this, so a bad value is
    both reported and repaired by the next save.

    Args:
        parser (ConfigParser): the file as read by ``config.read_parser``.

    Returns:
        list of (Field, str, str): the field, its raw text, and why the
        text is not used.
    """
    found = []
    for item in FIELDS:
        raw = parser.get(item.section, item.key, fallback=None)
        problem = None if raw is None else parse_value(item, raw)[1]
        if problem:
            found.append((item, raw.strip(), problem))
    return found


def unknown_names(parser) -> list:
    """List the sections and keys in the file that no setting reads.

    :func:`config_warnings` calls this, so a typo such as ``[LLM]`` or
    ``votek`` is named instead of silently ignored.

    Args:
        parser (ConfigParser): the file as read by ``config.read_parser``.

    Returns:
        list of str: ``[name]`` for a section, ``section.key`` for a key.
    """
    known = covered_names()
    sections = {section.name for section in SECTIONS}
    found = [f"[{name}]" for name in parser.sections() if name not in sections]
    for name in parser.sections():
        if name in sections:
            found += [f"{name}.{key}" for key in parser[name]
                      if f"{name}.{key}" not in known]
    return found


def config_warnings(path: str = None) -> list:
    """Say what in the config file is not used, one line per problem.

    The TUI start, the live reload, ``config show/get/set``, ``doctor``
    and the config screen show these lines.

    Args:
        path (str or None): file to check; default the resolved file.

    Returns:
        list of str: the whole-file problem alone, or one line per
        ignored value or unknown name; empty when everything is used.
    """
    problem = config_problem(path)
    if problem:
        return [problem]
    parser = read_parser(path)[0]
    ignored = [f"⚠ {item.dotted} = {raw!r} is ignored ({why}); using "
               f"{format_value(item, default_value(item))}"
               for item, raw, why in invalid_fields(parser)]
    return ignored + [f"⚠ {name} is not a setting; it is ignored"
                      for name in unknown_names(parser)]


def covered_names() -> set:
    """Return every ``section.key`` in the registry (used by tests).

    Returns: set of str.
    """
    return {item.dotted for item in FIELDS}


def dataclass_names() -> set:
    """Return every ``section.key`` the config dataclasses define.

    A test compares this with :func:`covered_names`. Returns: set of str.
    """
    config = ThreetoksConfig()
    return {f"{section.name}.{item.name}" for section in SECTIONS
            for item in dataclass_fields(getattr(config, section.name))}


if __name__ == "__main__":
    assert covered_names() == dataclass_names(), covered_names() ^ \
        dataclass_names()
    port = find_field("browser.mode")
    assert parse_value(port, "PLAIN") == ("plain", None)
    assert parse_value(find_field("llm.vote_k"), "0")[0] is None
    assert parse_value(find_field("search.min_interval_s"), "nan")[0] is None
    assert cycle_value(find_field("voice.post_speak_delay"), 0.8, -1) == 0.3
    assert cycle_value(port, "http", 1) == "plain"
    cfg = set_value(ThreetoksConfig(), find_field("llm.model"), "x")
    assert changed_fields(ThreetoksConfig(), cfg) == (find_field("llm.model"),)
    print("smoke OK")
