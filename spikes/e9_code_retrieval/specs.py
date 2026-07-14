"""Benchmark specs for E9 retrieval-as-repair.

A ``Spec`` is what the greenfield coding vertical already produces per
method: a canonical name, a signature, and an anchor assert derived from
one example input/output pair (docs/DESIGN-coding-agent.md §4). Retrieval
never sees the anchor as a hint to *write* code — it is purely the free
execution judge (task instructions §c): a fetched candidate is accepted
only if `python3 -c "<candidate>; <anchor>"` exits zero.

``aliases`` are additional names the fuzzy/substring matcher will accept
when scanning fetched source for a matching top-level def — real-world
code rarely uses the exact name we planned it under.

8 classic (textbook, E6's known failure tail includes roman_to_int and a
caesar cipher) + 4 invented (retrieval SHOULD miss these; they exist to
measure honest miss behaviour and its wall-clock cost).
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Spec:
    name: str                      # canonical name the anchor assert calls
    aliases: tuple                 # other plausible names in the wild
    anchor: str                    # a ready `assert name(...) == ...` line
    queries: tuple                 # search queries to try, in order
    classic: bool                  # True = expect a hit, False = expect a miss


CLASSIC_SPECS = (
    Spec(
        name="roman_to_int",
        aliases=("roman_to_integer", "romanToInt", "roman_numeral_to_int",
                  "romantoint", "convert_roman_to_int", "roman_to_decimal",
                  "romanToDecimal"),
        anchor='assert roman_to_int("MCMXCIV") == 1994',
        queries=('"def roman_to_int" python',
                  'roman to integer python function github'),
        classic=True,
    ),
    Spec(
        name="caesar_encode",
        aliases=("caesar_cipher", "caesar_encrypt", "encrypt_caesar",
                  "caesarCipher", "caesar_cipher_encrypt", "encrypt",
                  "caesar", "encrypt_message"),
        anchor='assert caesar_encode("xyz", 3) == "abc"',
        queries=('caesar cipher python function github',
                  'python caesar cipher encrypt function def'),
        classic=True,
    ),
    Spec(
        name="levenshtein",
        aliases=("levenshtein_distance", "edit_distance",
                  "levenshteinDistance", "levenshtein_distance_dp"),
        anchor='assert levenshtein("kitten", "sitting") == 3',
        queries=('"def levenshtein" python',
                  'levenshtein distance python function github'),
        classic=True,
    ),
    Spec(
        name="gcd",
        aliases=("greatest_common_divisor", "find_gcd", "compute_gcd",
                  "gcd_recursive", "gcd_iterative"),
        anchor='assert gcd(48, 18) == 6',
        queries=('"def gcd" python function github',
                  'greatest common divisor python function implementation'),
        classic=True,
    ),
    Spec(
        name="is_prime",
        aliases=("isprime", "check_prime", "prime_check", "is_prime_number"),
        anchor='assert is_prime(17) == True',
        queries=('"def is_prime" python',
                  'python function check if number is prime github'),
        classic=True,
    ),
    Spec(
        name="binary_search",
        aliases=("binarysearch", "binary_search_iterative",
                  "binary_search_recursive", "bsearch"),
        anchor='assert binary_search([1, 3, 5, 7, 9, 11], 7) == 3',
        queries=('"def binary_search" python',
                  'binary search python function implementation github'),
        classic=True,
    ),
    Spec(
        name="fibonacci",
        aliases=("fib", "nth_fibonacci", "fibonacci_number", "fibo"),
        anchor='assert fibonacci(10) == 55',
        queries=('"def fibonacci" python',
                  'fibonacci sequence python function github'),
        classic=True,
    ),
    Spec(
        name="is_palindrome",
        aliases=("ispalindrome", "check_palindrome", "palindrome_check",
                  "is_a_palindrome"),
        anchor='assert is_palindrome("racecar") == True',
        queries=('"def is_palindrome" python',
                  'python check palindrome string function github'),
        classic=True,
    ),
)

NONCLASSIC_SPECS = (
    Spec(
        name="snake_to_camel_but_keep_first",
        aliases=("snake_to_camel_keep_first", "to_camel_keep_first_word"),
        anchor=('assert snake_to_camel_but_keep_first("make_http_request") '
                '== "makeHttpRequest"'),
        queries=('snake_to_camel_but_keep_first python function',
                  'python convert snake case to camel case keep first word'),
        classic=False,
    ),
    Spec(
        name="zigzag_join",
        aliases=("zigzag_merge", "interleave_join"),
        anchor='assert zigzag_join([1, 2, 3], [10, 20]) == "1-10-2-20-3"',
        queries=('zigzag_join python function',
                  'python interleave two lists join with dash leftover'),
        classic=False,
    ),
    Spec(
        name="weighted_vowel_score",
        aliases=("vowel_weighted_score", "score_vowels_weighted"),
        anchor='assert weighted_vowel_score("banana") == 3',
        queries=('weighted_vowel_score python function',
                  'python weighted vowel scoring function a=1 e=2 i=3'),
        classic=False,
    ),
    Spec(
        name="title_case_odd_words",
        aliases=("title_case_every_other_word", "capitalize_odd_words"),
        anchor=('assert title_case_odd_words("the quick brown fox jumps") '
                '== "the Quick brown Fox jumps"'),
        queries=('title_case_odd_words python function',
                  'python title case every other word in a sentence'),
        classic=False,
    ),
)

ALL_SPECS = CLASSIC_SPECS + NONCLASSIC_SPECS


if __name__ == "__main__":
    assert len(CLASSIC_SPECS) == 8
    assert len(NONCLASSIC_SPECS) == 4
    assert {s.name for s in CLASSIC_SPECS} >= {"roman_to_int", "caesar_encode"}
    assert all(s.anchor.startswith("assert ") for s in ALL_SPECS)
    # every anchor must itself be valid Python once paired with a stub body
    import ast
    for s in ALL_SPECS:
        ast.parse(s.anchor)
    print("smoke OK")
