"""Offline tests for retrieval-as-repair. No network: providers and page
fetches are faked; execution judging runs real subprocesses on fixtures."""
import unittest

from threetoks.backend.base import FAMILY_CHATML, GenResult, ModelSpec
from threetoks.code import retrieve
from threetoks.code.vertical import CodeVertical
from threetoks.engine import run_episode
from threetoks.policy import Policy, PolicyConfig

ROMAN_PAGE = """<html><body><p>Here is a classic solution:</p>
<pre>
ROMANS = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

def romanToInt(s):
    total = 0
    for i in range(len(s)):
        value = ROMANS[s[i]]
        if i + 1 < len(s) and ROMANS[s[i + 1]] > value:
            total -= value
        else:
            total += value
    return total
</pre></body></html>"""


class ExtractionTest(unittest.TestCase):
    def test_fuzzy_named_def_is_renamed_and_self_contained(self):
        blocks = retrieve.html_code_blocks(ROMAN_PAGE)
        self.assertEqual(len(blocks), 1)
        [snippet] = retrieve.extract_candidates(blocks[0], "roman_to_int")
        self.assertIn("def roman_to_int(", snippet)
        self.assertIn("ROMANS", snippet)               # constant pulled in

    def test_unsafe_import_and_forbidden_builtin_reject_before_execution(self):
        unsafe = "import os\n\ndef target(s):\n    return os.getpid()\n"
        self.assertEqual(retrieve.extract_candidates(unsafe, "target"), [])
        golfed = "def target(s):\n    return eval('1')\n"
        self.assertEqual(retrieve.extract_candidates(golfed, "target"), [])
        dunder = ("def target(s):\n"
                  "    return ().__class__.__mro__\n")
        self.assertEqual(retrieve.extract_candidates(dunder, "target"), [])

    def test_star_import_files_contribute_nothing(self):
        starred = "from os import *\n\ndef target(s):\n    return s\n"
        self.assertEqual(retrieve.extract_candidates(starred, "target"), [])

    def test_github_blob_urls_rewrite_to_raw(self):
        self.assertEqual(
            retrieve.github_raw_url("https://github.com/o/r/blob/main/f.py"),
            "https://raw.githubusercontent.com/o/r/main/f.py")
        self.assertIsNone(retrieve.github_raw_url("https://github.com/o/r"))
        self.assertIsNone(
            retrieve.github_raw_url("https://evil.com/github.com/blob/x"))


class _Result:
    def __init__(self, url):
        self.url = url
        self.title = self.snippet = ""


class _FakeProvider:
    def __init__(self, urls):
        self.urls = urls
        self.queries = []

    def search(self, query, max_results=8):
        self.queries.append(query)
        return [_Result(u) for u in self.urls]


class RetrieverTest(unittest.TestCase):
    def _retriever(self, pages, urls):
        retriever = retrieve.Retriever(_FakeProvider(urls))
        retriever._fetch = lambda url: pages.get(url)
        return retriever

    def test_anchor_passing_candidate_wins_with_provenance(self):
        retriever = self._retriever({"http://x/roman": ROMAN_PAGE},
                                    ["http://x/roman"])
        snippet = retriever("roman_to_int", "s", "roman numerals to int",
                            ['roman_to_int("MCMXCIV") == 1994',
                             'roman_to_int("X") == 10'])
        self.assertIsNotNone(snippet)
        self.assertIn("retrieved from http://x/roman", snippet)

    def test_anchor_failing_candidates_are_honest_misses(self):
        wrong = "<pre>def roman_to_int(s):\n    return 0\n</pre>"
        retriever = self._retriever({"http://x/wrong": wrong},
                                    ["http://x/wrong"])
        self.assertIsNone(retriever("roman_to_int", "s", "roman numerals",
                                    ['roman_to_int("X") == 10']))

    def test_no_examples_means_no_retrieval_at_all(self):
        provider = _FakeProvider(["http://x/a"])
        retriever = retrieve.Retriever(provider)
        self.assertIsNone(retriever("f", "x", "contract", []))
        self.assertEqual(provider.queries, [])         # zero network calls


class VerticalIntegrationTest(unittest.TestCase):
    def test_exhausted_anchor_method_is_repaired_by_retrieval(self):
        # The scripted model produces the same wrong body until attempts
        # exhaust; the retriever then supplies a passing implementation
        # instead of a stub, marked retrieved + verified.
        def fake_retriever(name, args, contract, examples):
            return ("def roman_to_int(s):\n"
                    "    romans = {'I': 1, 'V': 5, 'X': 10}\n"
                    "    total = 0\n"
                    "    for i, ch in enumerate(s):\n"
                    "        value = romans[ch]\n"
                    "        if i + 1 < len(s) and romans[s[i + 1]] > value:\n"
                    "            total -= value\n"
                    "        else:\n"
                    "            total += value\n"
                    "    return total\n")

        class WrongBodyBackend:
            def __init__(self):
                self.texts = ["roman_to_int(s): convert numerals to int",
                              "return 0"]

            def complete(self, model, raw_prompt, opts):
                text = self.texts.pop(0) if len(self.texts) > 1 \
                    else self.texts[0]
                return GenResult(text, 8, 4, 0.0, "stop")

        vertical = CodeVertical(
            "convert roman numerals with roman_to_int(s)", gen_tests=False,
            trusted_examples={"roman_to_int": ['roman_to_int("IX") == 9',
                                               'roman_to_int("X") == 10']},
            retriever=fake_retriever)
        policy = Policy(WrongBodyBackend(),
                        PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = run_episode(vertical, policy, max_steps=25)
        row = next(r for r in outcome["methods"]
                   if r["name"] == "roman_to_int")
        self.assertEqual(row["status"], "tested")
        self.assertTrue(row["verified"])
        self.assertTrue(row["retrieved"])
        self.assertIn("romans[ch]", outcome["answer"])

    def test_retriever_failure_still_stubs_honestly(self):
        def broken_retriever(name, args, contract, examples):
            raise RuntimeError("network down")

        class WrongBodyBackend:
            def __init__(self):
                self.texts = ["roman_to_int(s): convert numerals",
                              "return 0"]

            def complete(self, model, raw_prompt, opts):
                text = self.texts.pop(0) if len(self.texts) > 1 \
                    else self.texts[0]
                return GenResult(text, 8, 4, 0.0, "stop")

        vertical = CodeVertical(
            "convert roman numerals with roman_to_int(s)", gen_tests=False,
            trusted_examples={"roman_to_int": 'roman_to_int("X") == 10'},
            retriever=broken_retriever)
        policy = Policy(WrongBodyBackend(),
                        PolicyConfig(ModelSpec("m", FAMILY_CHATML)))
        outcome = run_episode(vertical, policy, max_steps=25)
        row = next(r for r in outcome["methods"]
                   if r["name"] == "roman_to_int")
        self.assertEqual(row["status"], "stubbed")
        self.assertFalse(row["retrieved"])


if __name__ == "__main__":
    unittest.main()
