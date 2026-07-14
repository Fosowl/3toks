"""Offline test of the extraction+gate funnel: saved fixtures only, no
network. Run with:
    python3 -m unittest spikes.e9_code_retrieval.tests.test_extract_gate
or:
    python3 -m pytest spikes/e9_code_retrieval/tests/test_extract_gate.py
"""
import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract  # noqa: E402
import fetchers  # noqa: E402
import judge  # noqa: E402
from specs import Spec  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"

ROMAN_SPEC = Spec(
    name="roman_to_int", aliases=("romanToInt",),
    anchor='assert roman_to_int("MCMXCIV") == 1994',
    queries=(), classic=True)
CAESAR_SPEC = Spec(
    name="caesar_encode", aliases=(),
    anchor='assert caesar_encode("xyz", 3) == "abc"',
    queries=(), classic=True)
GCD_SPEC = Spec(
    name="gcd", aliases=(),
    anchor='assert gcd(48, 18) == 6',
    queries=(), classic=True)


class ExtractGateFunnelTest(unittest.TestCase):
    """A fetched raw .py file with a fuzzy-named def + a helper it calls +
    a whitelisted import: should survive every gate and pass the anchor."""

    def test_full_funnel_accepts_good_candidate(self):
        text = (FIXTURES / "roman_raw.py").read_text()
        candidates, parsed_ok = extract.find_matches(
            text, ROMAN_SPEC, "fixture://roman_raw.py")
        self.assertTrue(parsed_ok)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tier, "exact")  # 'romanToInt' alias

        tree = ast.parse(text)
        extract.assemble(candidates[0], ROMAN_SPEC.name, tree)
        self.assertIsNone(candidates[0].rejected)
        self.assertIn("def roman_to_int(", candidates[0].snippet)
        self.assertIn("_is_valid", candidates[0].snippet)  # helper pulled in
        self.assertIn("import re", candidates[0].snippet)  # its import too

        ok, err, elapsed = judge.judge(candidates[0].snippet,
                                       ROMAN_SPEC.anchor)
        self.assertTrue(ok, err)
        self.assertLess(elapsed, 5.0)

    def test_unsafe_import_rejected_before_execution(self):
        """The unsafe fixture must never reach judge.judge at all — the
        assemble() gate rejects it on the disallowed `os` import."""
        text = (FIXTURES / "unsafe_raw.py").read_text()
        candidates, parsed_ok = extract.find_matches(
            text, CAESAR_SPEC, "fixture://unsafe_raw.py")
        self.assertTrue(parsed_ok)
        self.assertEqual(len(candidates), 1)

        tree = ast.parse(text)
        extract.assemble(candidates[0], CAESAR_SPEC.name, tree)
        self.assertIsNotNone(candidates[0].rejected)
        self.assertTrue(candidates[0].rejected.startswith("unsafe_import"),
                        candidates[0].rejected)
        self.assertIsNone(candidates[0].snippet)

    def test_html_pre_block_extraction_and_full_round_trip(self):
        """A non-GitHub HTML page (rosettacode/tutorial-style): pull code
        out of <pre> blocks, name-match, gate, and pass the anchor."""
        html = (FIXTURES / "gcd_page.html").read_text()
        blocks = fetchers.extract_code_blocks(html)
        self.assertEqual(len(blocks), 1)  # the "hello world" pre is skipped
        self.assertIn("def gcd(", blocks[0])

        candidates, parsed_ok = extract.find_matches(
            blocks[0], GCD_SPEC, "fixture://gcd_page.html#pre1")
        self.assertTrue(parsed_ok)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tier, "exact")

        tree = ast.parse(blocks[0])
        extract.assemble(candidates[0], GCD_SPEC.name, tree)
        self.assertIsNone(candidates[0].rejected)
        ok, err, _ = judge.judge(candidates[0].snippet, GCD_SPEC.anchor)
        self.assertTrue(ok, err)

    def test_bad_syntax_file_yields_zero_candidates(self):
        candidates, parsed_ok = extract.find_matches(
            "def f(:\n    pass\n", ROMAN_SPEC, "fixture://broken.py")
        self.assertEqual(candidates, [])
        self.assertFalse(parsed_ok)


if __name__ == "__main__":
    unittest.main()
