"""Planted-bug scenarios for the edit-vertical live eval.

Scenarios 1-5 are ported verbatim from spike E8 so the numbers stay
comparable with the spike's report; 6-12 are new and target exactly the
v2 mechanisms: method spans, free operation inference, ranked
disambiguation (both when it helps and when it misleads), delete menus
with several plausible statements, deeper navigation, and one
E6-difficulty logic body.

Each scenario is a self-contained corpus (fixture files + a trusted test
that fails on the buggy source and passes on a correct fix).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OP_REPLACE, OP_INSERT_AFTER, OP_DELETE = "replace", "insert_after", "delete"


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    request: str
    files: dict           # relative posix path -> buggy source
    test_code: str        # fails before the fix, passes after
    expected_operation: str


def materialize(scenario: Scenario, dest_root: Path) -> None:
    """Write a scenario's fixture files fresh into dest_root."""
    for rel, source in scenario.files.items():
        path = dest_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


# ------------------------------------------------- 1-5: ported from E8

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

SCENARIO_1 = Scenario(
    key="1_direct_replace",
    title="off-by-one in slice_page (direct lookup, replace)",
    request="fix the off-by-one bug in slice_page -- it's returning one "
            "extra item at the end of the page",
    files={"pagination/paginator.py": _S1_PAGINATOR},
    test_code=("from pagination.paginator import slice_page\n"
               "result = slice_page(list(range(10)), 2, 3)\n"
               "assert result == [3, 4, 5], result\n"),
    expected_operation=OP_REPLACE,
)

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

SCENARIO_2 = Scenario(
    key="2_direct_insert_after",
    title="average() calls an undefined helper (insert-after)",
    request="average() crashes with a NameError -- it calls a helper "
            "function that was never written in this file. Add the "
            "missing helper.",
    files={"mathutils/stats.py": _S2_STATS},
    test_code=("from mathutils.stats import average\n"
               "assert average([1, 2, 3]) == 2\n"
               "assert average([4, 4, 4]) == 4\n"),
    expected_operation=OP_INSERT_AFTER,
)

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

SCENARIO_3 = Scenario(
    key="3_direct_delete",
    title="stray line undoes casing in to_snake_case (delete)",
    request="to_snake_case is returning UPPERCASE text instead of "
            "snake_case -- there's a stray line undoing the case "
            "conversion right before it returns. Remove it.",
    files={"textutils/casing.py": _S3_CASING},
    test_code=("from textutils.casing import to_snake_case\n"
               'assert to_snake_case("CamelCase") == "camel_case"\n'
               'assert to_snake_case("already snake") == "already_snake"\n'),
    expected_operation=OP_DELETE,
)

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

_S4_ROUNDING_DECOY = '''"""Rounding helpers -- decoy folder, unrelated."""

import math


def round_half_up(value):
    """Round a float to the nearest int, .5 rounds up."""
    return math.floor(value + 0.5)
'''

SCENARIO_4 = Scenario(
    key="4_navigation_replace",
    title="double spaces survive normalize_whitespace (navigate)",
    request="the whitespace cleanup helper in the text utilities keeps "
            "double spaces in the middle of sentences instead of "
            "collapsing them down to one",
    files={"textutils/strings.py": _S4_STRINGS,
           "textutils/wordcount.py": _S4_WORDCOUNT_DECOY,
           "mathutils_extra/rounding.py": _S4_ROUNDING_DECOY},
    test_code=("from textutils.strings import normalize_whitespace\n"
               'assert normalize_whitespace("a   b\\tc") == "a b c"\n'
               'assert normalize_whitespace("  x  y  ") == "x y"\n'),
    expected_operation=OP_REPLACE,
)

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

SCENARIO_5 = Scenario(
    key="5_ambiguous_replace",
    title="validate() defined twice -> disambiguation, replace",
    request="fix the bug in validate -- it's letting zero-total orders "
            "through when it shouldn't",
    files={"duplicates/users.py": _S5_USERS,
           "duplicates/orders.py": _S5_ORDERS},
    test_code=("from duplicates import orders\n"
               'assert orders.validate({"total": 5}) is True\n'
               'assert orders.validate({"total": 0}) is False\n'),
    expected_operation=OP_REPLACE,
)


# ------------------------------------------------- 6-12: new for v2

_S6_CART = '''"""A tiny shopping cart."""


class Cart:
    def __init__(self, prices):
        self.prices = list(prices)

    def total(self):
        """Sum of all prices in the cart."""
        result = 0
        for price in self.prices:
            result = result - price
        return result
'''

SCENARIO_6 = Scenario(
    key="6_method_replace",
    title="Cart.total subtracts instead of adds (method span, replace)",
    request="Cart.total returns negative numbers -- fix total so it "
            "actually sums the prices",
    files={"shop/cart.py": _S6_CART},
    test_code=("from shop.cart import Cart\n"
               "assert Cart([1, 2, 3]).total() == 6\n"
               "assert Cart([]).total() == 0\n"),
    expected_operation=OP_REPLACE,
)

_S7_NUMERICS = '''"""Sequence numerics."""


def spread(numbers):
    """Max minus min of a non-empty sequence."""
    return max(numbers) - min(numbers)


def average(numbers):
    """Arithmetic mean of a non-empty sequence."""
    total = 0
    for n in numbers:
        total = _accumulate(total, n)
    return total / len(numbers)
'''

SCENARIO_7 = Scenario(
    key="7_inferred_insert",
    title="undefined helper, request gives no hint (free inference)",
    request="average in the numerics module is broken and blows up on "
            "any input, make it work",
    files={"numerics/seq.py": _S7_NUMERICS},
    test_code=("from numerics.seq import average\n"
               "assert average([2, 4, 6]) == 4\n"),
    expected_operation=OP_INSERT_AFTER,
)

_S8_INVOICE = '''"""Invoice rendering."""


def render(data):
    """One-line invoice summary."""
    total = sum(data.get("amounts", []))
    return "Total: " + total
'''

_S8_EMAIL = '''"""Email rendering."""


def render(data):
    """Subject line for a notification email."""
    return "[notice] " + data.get("subject", "")
'''

_S8_SUMMARY = '''"""Daily summary rendering."""


def render(data):
    """Counts per day."""
    return ", ".join(f"{k}={v}" for k, v in sorted(data.items()))
'''

SCENARIO_8 = Scenario(
    key="8_three_way_disambiguate",
    title="render() defined three times, keyword picks invoice free",
    request="the invoice render crashes when it prints the totals",
    files={"reports/invoice.py": _S8_INVOICE,
           "reports/email.py": _S8_EMAIL,
           "reports/summary.py": _S8_SUMMARY},
    test_code=("from reports.invoice import render\n"
               'assert render({"amounts": [2, 3]}) == "Total: 5"\n'),
    expected_operation=OP_REPLACE,
)

_S9_COUNTING = '''"""Counting helpers."""


def count_evens(nums):
    """How many numbers are even."""
    count = 0
    evens = [n for n in nums if n % 2 == 0]
    count = len(evens)
    count = count + len(nums)
    return count
'''

SCENARIO_9 = Scenario(
    key="9_delete_multi_statement",
    title="stray double-count line among several statements (delete)",
    request="count_evens double counts -- there's a stray line adding "
            "the length of the whole list on top of the answer. Remove "
            "that line.",
    files={"counting/evens.py": _S9_COUNTING},
    test_code=("from counting.evens import count_evens\n"
               "assert count_evens([1, 2, 3, 4]) == 2\n"
               "assert count_evens([]) == 0\n"),
    expected_operation=OP_DELETE,
)

_S10_CIPHER = '''"""Letter shifting."""


def shift_char(c, k):
    """Shift a lowercase letter k places, wrapping past z back to a."""
    return chr(ord(c) + k)
'''

SCENARIO_10 = Scenario(
    key="10_hard_logic_replace",
    title="shift_char never wraps past z (E6-difficulty logic)",
    request="shift_char is supposed to wrap around the alphabet but "
            "shifting z by 1 gives garbage instead of a -- fix the wrap",
    files={"cipher/shift.py": _S10_CIPHER},
    test_code=("from cipher.shift import shift_char\n"
               "assert shift_char('z', 1) == 'a'\n"
               "assert shift_char('a', 2) == 'c'\n"
               "assert shift_char('y', 3) == 'b'\n"),
    expected_operation=OP_REPLACE,
)

_S11_ORDERS_OK = '''"""Order price display (correct)."""


def format_price(cents):
    """Cents -> dollars string."""
    return f"${cents / 100:.2f}"
'''

_S11_RECEIPTS_BUG = '''"""Receipt price display."""


def format_price(cents):
    """Cents -> dollars string."""
    return f"${cents}.00"
'''

SCENARIO_11 = Scenario(
    key="11_misleading_keyword_relocate",
    title="request keyword points at the WRONG file (relocate recovery)",
    request="format_price shows wrong prices on orders",
    files={"billing/orders.py": _S11_ORDERS_OK,
           "billing/receipts.py": _S11_RECEIPTS_BUG},
    test_code=("from billing.receipts import format_price\n"
               'assert format_price(150) == "$1.50"\n'),
    expected_operation=OP_REPLACE,
)

_S12_DURATION = '''"""Duration parsing."""


def parse_duration(text):
    """'MM:SS' -> total seconds."""
    minutes, seconds = text.split(":")
    return int(minutes) + int(seconds)


def format_duration(total):
    """Total seconds -> 'MM:SS'."""
    return f"{total // 60}:{total % 60:02d}"
'''

_S12_CALENDAR_DECOY = '''"""Date helpers -- decoy file."""


def is_weekend(day_index):
    """0=Monday ... 6=Sunday."""
    return day_index >= 5
'''

_S12_GEOMETRY_DECOY = '''"""Geometry -- decoy folder."""


def area(w, h):
    """Rectangle area."""
    return w * h
'''

SCENARIO_12 = Scenario(
    key="12_deep_navigation",
    title="no symbol named, two folders, decoys (navigate deep)",
    request="the time parsing helper adds minutes and seconds together "
            "without converting the minutes first",
    files={"timeutils/duration.py": _S12_DURATION,
           "timeutils/calendar.py": _S12_CALENDAR_DECOY,
           "geometry/shapes.py": _S12_GEOMETRY_DECOY},
    test_code=("from timeutils.duration import parse_duration\n"
               "assert parse_duration('2:30') == 150\n"
               "assert parse_duration('0:45') == 45\n"),
    expected_operation=OP_REPLACE,
)


ALL_SCENARIOS = [SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5,
                 SCENARIO_6, SCENARIO_7, SCENARIO_8, SCENARIO_9, SCENARIO_10,
                 SCENARIO_11, SCENARIO_12]


if __name__ == "__main__":
    import ast
    import tempfile

    from threetoks.code.edit import oracle

    assert len({s.key for s in ALL_SCENARIOS}) == 12
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in ALL_SCENARIOS:
            root = Path(tmp) / scenario.key
            materialize(scenario, root)
            for rel in scenario.files:
                ast.parse((root / rel).read_text())      # fixtures are valid
            ok, _ = oracle.run_test(root, scenario.test_code)
            assert not ok, f"{scenario.key}: test must fail on buggy source"
    print("smoke OK")
