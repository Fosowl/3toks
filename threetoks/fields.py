"""Field registry: one table that describes every config knob.

The config screen, the ``threetoks config`` command, and the doctor read
this table. It lists each knob once: its section, kind, label, help text,
and when a change takes effect. A test checks that the table covers every
field of every config dataclass, so a new knob cannot be forgotten.
"""
from dataclasses import dataclass, fields as dataclass_fields, replace

from threetoks.config import (BROWSER_MODES, PROVIDER_CHOICES,
                              ThreetoksConfig)

KIND_CHOICE = "choice"
KIND_BOOL = "bool"
KIND_INT = "int"
KIND_FLOAT = "float"
KIND_TEXT = "text"

APPLY_LIVE = "live"          # a running session takes the change now
APPLY_RESTART = "restart"    # read once at start
APPLY_VOICE = "/voice on"    # read when voice mode starts

_FLOAT_STEP = 0.5
_ON, _OFF = "on", "off"
_EMPTY_SHOWN = "(empty)"


@dataclass(frozen=True)
class Section:
    """One ``[section]`` of the INI file with its harbor-themed title."""

    name: str
    title: str
    tagline: str


@dataclass(frozen=True)
class Field:
    """One config knob: where it lives, how to edit it, when it applies."""

    section: str
    key: str
    kind: str
    label: str
    help: str
    choices: tuple = ()
    apply: str = APPLY_RESTART
    minimum: float | None = None

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
          "Model that makes every decision node.", apply=APPLY_LIVE),
    Field("llm", "host", KIND_TEXT, "ollama host",
          "Where Ollama listens (ollama provider only)."),
    Field("llm", "api_base", KIND_TEXT, "api base",
          "Overrides the provider URL. Required for custom."),
    Field("llm", "vote_k", KIND_INT, "vote k",
          "Samples per decision. 1 is the measured default.", minimum=1),
    Field("llm", "send_seed", KIND_BOOL, "send seed",
          "Off for servers that reject the sampling seed."),
    Field("llm", "disable_think", KIND_BOOL, "disable think",
          "Pre-close the think block. Thinking Qwen needs it on."),
    Field("browser", "mode", KIND_CHOICE, "fetch mode",
          "http = fast, no JavaScript. plain = real Chrome.",
          choices=BROWSER_MODES, apply=APPLY_LIVE),
    Field("browser", "visible", KIND_BOOL, "show window",
          "Show the Chrome window (plain mode only).", apply=APPLY_LIVE),
    Field("search", "searxng_url", KIND_TEXT, "searxng url",
          "Base URL of the local SearXNG."),
    Field("search", "min_interval_s", KIND_FLOAT, "min interval s",
          "Least seconds between two outbound searches.", minimum=0),
    Field("research", "max_rounds", KIND_INT, "max rounds",
          "Judged research rounds per question.", apply=APPLY_LIVE,
          minimum=1),
    Field("research", "max_steps", KIND_INT, "max steps",
          "Decision steps allowed per episode.", minimum=1),
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
          "Start already listening and speaking.", apply=APPLY_VOICE),
    Field("voice", "lang", KIND_TEXT, "language",
          "Vosk language code, for example en-us.", apply=APPLY_VOICE),
    Field("voice", "stt_model_path", KIND_TEXT, "stt model path",
          "Vosk model folder. Empty = auto-download.", apply=APPLY_VOICE),
    Field("voice", "tts_model_path", KIND_TEXT, "tts model path",
          "Piper .onnx voice. Needs its .onnx.json beside it.",
          apply=APPLY_VOICE),
    Field("voice", "tts_speaker", KIND_INT, "tts speaker",
          "Speaker id. -1 = the voice default.", apply=APPLY_VOICE,
          minimum=-1),
    Field("voice", "post_speak_delay", KIND_FLOAT, "post speak delay",
          "Seconds before the microphone unmutes.", apply=APPLY_VOICE,
          minimum=0),
    Field("camera", "index", KIND_INT, "device index",
          "Camera device the look agent captures from.", minimum=0),
    Field("relay", "pin", KIND_INT, "gpio pin",
          "BCM pin of the active-low relay board.", minimum=0),
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
    """Parse a number and check the field's minimum."""
    try:
        value = _parse_number(field, text)
    except ValueError:
        return None, f"not a {field.kind}"
    if field.minimum is not None and value < field.minimum:
        return None, f"at least {field.minimum:g}"
    return value, None


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
    moved = value + step * size
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
    assert cycle_value(port, "http", 1) == "plain"
    cfg = set_value(ThreetoksConfig(), find_field("llm.model"), "x")
    assert changed_fields(ThreetoksConfig(), cfg) == (find_field("llm.model"),)
    print("smoke OK")
