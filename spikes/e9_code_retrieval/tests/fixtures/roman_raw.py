"""Simulated raw.githubusercontent.com fetch for a roman-numeral repo."""
import re

ROMANS = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}


def _is_valid(s):
    return bool(re.fullmatch(r"[IVXLCDM]+", s))


def romanToInt(s):
    if not _is_valid(s):
        raise ValueError("not a roman numeral")
    total = 0
    for i, ch in enumerate(s):
        value = ROMANS[ch]
        if i + 1 < len(s) and ROMANS[s[i + 1]] > value:
            total -= value
        else:
            total += value
    return total


def unrelated_helper(x):
    """A sibling function the matcher should ignore for this spec."""
    return x * 2
