"""Five end-to-end bug-fix scenarios, entirely inside this spike's sandbox.

Every fixture module here is invented for this spike -- none of it is real
repo code. Each scenario plants one bug plus a trusted test that fails
against the buggy source and passes against the intended fix, and exercises
a different corner of the edit vertical:

  1. direct lookup (unique symbol name) + replace-span
  2. direct lookup + insert-after-span (a helper the code calls but never
     defines -- the harness finds the missing name/arity for free, see
     edit_ops.find_missing_helper)
  3. direct lookup + delete-span, and ZERO generation calls at all
  4. no symbol named in the request -> folder -> file -> def navigation,
     then replace-span
  5. an ambiguous symbol name (defined in two files) -> one disambiguation
     menu, then replace-span

`request` is deliberately phrased like a human bug report, not a symbol
lookup key -- scenarios 1/2/3/5 happen to name the function; scenario 4
does not, on purpose, to force navigation.
"""
from dataclasses import dataclass, field
from pathlib import Path

OP_REPLACE = "replace"
OP_INSERT_AFTER = "insert_after"
OP_DELETE = "delete"


@dataclass
class Scenario:
    key: str
    title: str
    request: str
    files: dict[str, str]          # relative posix path -> buggy source
    test_code: str                  # trusted oracle test (fails before, passes after)
    expected_operation: str          # replace | insert_after | delete
    target_name: str                  # function name the fix targets
    # offline-only scripting aids (never seen by the vertical itself):
    nav_keywords: list[str] = field(default_factory=list)   # folder/file/def
    disambiguate_keyword: str | None = None
    good_completion: str | None = None      # canned correct model output
    delete_keyword: str | None = None        # substring of the buggy line


def materialize(scenario: Scenario, dest_root: Path) -> None:
    """Write a scenario's fixture files fresh into dest_root."""
    for rel, source in scenario.files.items():
        path = dest_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


# ------------------------------------------------------------ scenario 1

_S1_PAGINATOR = '''"""Pagination helpers."""


def slice_page(items, page, size):
    """Return the 1-based `page`-th slice of `items`, each of length `size`."""
    start = (page - 1) * size
    end = start + size + 1
    return items[start:end]


def page_count(items, size):
    """Number of pages needed to cover all items."""
    if size <= 0:
        return 0
    return (len(items) + size - 1) // size
'''

_S1_TEST = '''from pagination.paginator import slice_page

items = list(range(10))
result = slice_page(items, 2, 3)
assert result == [3, 4, 5], result
assert len(result) == 3, result
'''

SCENARIO_1 = Scenario(
    key="1_direct_replace",
    title="off-by-one in slice_page (direct lookup, replace-span)",
    request="fix the off-by-one bug in slice_page -- it's returning one "
            "extra item at the end of the page",
    files={"pagination/paginator.py": _S1_PAGINATOR},
    test_code=_S1_TEST,
    expected_operation=OP_REPLACE,
    target_name="slice_page",
    good_completion=("start = (page - 1) * size\n    end = start + size\n"
                     "    return items[start:end]"),
)


# ------------------------------------------------------------ scenario 2

_S2_STATS = '''"""Small numeric helpers."""


def clamp(value, low, high):
    """Clamp value into [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def average(numbers):
    """Arithmetic mean of a non-empty sequence."""
    total = 0
    for n in numbers:
        total = _safe_add(total, n)
    return total / len(numbers)
'''

_S2_TEST = '''from mathutils.stats import average

assert average([1, 2, 3]) == 2, average([1, 2, 3])
assert average([4, 4, 4]) == 4, average([4, 4, 4])
'''

SCENARIO_2 = Scenario(
    key="2_direct_insert_after",
    title="average() calls an undefined helper (direct lookup, insert-after-span)",
    request="average() crashes with a NameError -- it calls a helper "
            "function that was never written in this file. Add the "
            "missing helper.",
    files={"mathutils/stats.py": _S2_STATS},
    test_code=_S2_TEST,
    expected_operation=OP_INSERT_AFTER,
    target_name="average",
    good_completion="return a + b",
)


# ------------------------------------------------------------ scenario 3

_S3_CASING = '''"""Case-conversion helpers."""

import re


def to_snake_case(name):
    """Convert CamelCase or spaced text to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\\1_\\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s1)
    result = s2.replace(" ", "_").lower()
    result = result.upper()
    return result
'''

_S3_TEST = '''from textutils.casing import to_snake_case

assert to_snake_case("CamelCase") == "camel_case", to_snake_case("CamelCase")
assert to_snake_case("already snake") == "already_snake", \\
    to_snake_case("already snake")
'''

SCENARIO_3 = Scenario(
    key="3_direct_delete",
    title="stray line undoes casing in to_snake_case (direct lookup, "
          "delete-span, zero generation calls)",
    request="to_snake_case is returning UPPERCASE text instead of "
            "snake_case -- there's a stray line undoing the case "
            "conversion right before it returns. Remove it.",
    files={"textutils/casing.py": _S3_CASING},
    test_code=_S3_TEST,
    expected_operation=OP_DELETE,
    target_name="to_snake_case",
    delete_keyword="result.upper()",
)


# ------------------------------------------------------------ scenario 4

_S4_STRINGS = '''"""Whitespace helpers."""


def normalize_whitespace(text):
    """Collapse runs of whitespace into single spaces and strip the ends."""
    result = text.replace("\\t", " ")
    return result.strip()


def count_words(text):
    """Count whitespace-separated words."""
    return len(text.split())
'''

_S4_WORDCOUNT_DECOY = '''"""Word counting decoy (unrelated to the bug)."""


def unique_words(text):
    """Count distinct words, case-insensitive."""
    return len(set(text.lower().split()))
'''

_S4_ROUNDING_DECOY = '''"""Rounding helpers -- decoy folder, unrelated to the bug."""

import math


def round_half_up(value):
    """Round a float to the nearest int, .5 rounds up."""
    return math.floor(value + 0.5)
'''

_S4_TEST = '''from textutils.strings import normalize_whitespace

assert normalize_whitespace("a   b\\tc") == "a b c", \\
    normalize_whitespace("a   b\\tc")
assert normalize_whitespace("  x  y  ") == "x y", \\
    normalize_whitespace("  x  y  ")
'''

SCENARIO_4 = Scenario(
    key="4_navigation_replace",
    title="double spaces survive normalize_whitespace (no symbol named -> "
          "folder/file/def navigation, replace-span)",
    request="the whitespace cleanup helper in the text utilities keeps "
            "double spaces in the middle of sentences instead of "
            "collapsing them down to one",
    files={
        "textutils/strings.py": _S4_STRINGS,
        "textutils/wordcount.py": _S4_WORDCOUNT_DECOY,
        "mathutils_extra/rounding.py": _S4_ROUNDING_DECOY,
    },
    test_code=_S4_TEST,
    expected_operation=OP_REPLACE,
    target_name="normalize_whitespace",
    nav_keywords=["textutils", "strings.py", "normalize_whitespace"],
    good_completion=('import re\n    result = re.sub(r"\\s+", " ", text)\n'
                     "    return result.strip()"),
)


# ------------------------------------------------------------ scenario 5

_S5_USERS = '''"""User record validation."""


def validate(record):
    """Return True when the user record has a plausible email."""
    email = record.get("email", "")
    return "@" in email and len(email) > 3
'''

_S5_ORDERS = '''"""Order record validation."""


def validate(record):
    """Return True when the order's total is a positive amount."""
    total = record.get("total", 0)
    return total >= 0
'''

_S5_TEST = '''from duplicates import orders

assert orders.validate({"total": 5}) is True, orders.validate({"total": 5})
assert orders.validate({"total": 0}) is False, orders.validate({"total": 0})
'''

SCENARIO_5 = Scenario(
    key="5_ambiguous_replace",
    title="validate() is ambiguous (two files define it) -> disambiguation "
          "menu, replace-span",
    request="fix the bug in validate -- it's letting zero-total orders "
            "through when it shouldn't",
    files={
        "duplicates/users.py": _S5_USERS,
        "duplicates/orders.py": _S5_ORDERS,
    },
    test_code=_S5_TEST,
    expected_operation=OP_REPLACE,
    target_name="validate",
    disambiguate_keyword="orders.py",
    good_completion=('total = record.get("total", 0)\n'
                     "    return total > 0"),
)


ALL_SCENARIOS: list[Scenario] = [SCENARIO_1, SCENARIO_2, SCENARIO_3,
                                  SCENARIO_4, SCENARIO_5]


if __name__ == "__main__":
    import ast
    import tempfile

    assert len({s.key for s in ALL_SCENARIOS}) == 5
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in ALL_SCENARIOS:
            root = Path(tmp) / scenario.key
            materialize(scenario, root)
            for rel in scenario.files:
                ast.parse((root / rel).read_text())    # fixtures are valid
    print("smoke OK")
