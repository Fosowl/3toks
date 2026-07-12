"""Generates benchmark.jsonl: hand-authored (request, label) pairs.

Labels are assigned by genuine user-intent judgment, written independently
of whatever the pre-router or model would guess — see REPORT.md for the
measured (honest) precision/coverage once pre_router.py is run against
this fixed set. A handful of items are deliberately ambiguous (marked in
comments below); they still get one ground-truth label each, chosen by
the most natural reading of the request.
"""
import json
from pathlib import Path

from pre_router import AUTHOR, COMPUTE, EDIT, NAVIGATE

OUT_PATH = Path(__file__).parent / "benchmark.jsonl"

NAVIGATE_ITEMS = [
    "Where is the slugify function defined?",
    "What does parse_line do in parser.py?",
    "Where is the Cache class implemented in pkg/utils.py?",
    "How does the top_ip function pick the most frequent IP?",
    "What's the purpose of the Cache eviction logic in pkg/utils.py?",
    "Explain how notes.md ties into the parser bug.",
    "Which function in parser.py has the off-by-one bug?",
    "Where does the code read the sales.csv file from?",
    "What is Cache.capacity used for?",
    "Can you point me to where slugify is defined?",
    "Which method evicts the oldest cache entry?",
    "I'm trying to understand what parse_line returns — where is it defined?",
    "What does the utils.py module export?",
    "How is the log line parsed in parser.py?",
    "Locate the function that slugifies text.",
    "Walk me through what happens when Cache.put is called twice past capacity.",
    "Is there a class in this codebase that handles caching?",
    # ambiguous: diagnostic question, no explicit fix requested
    "Can you check why parser.py gives the wrong method field?",
    # ambiguous: "how is X computed" reads as explain-the-logic, not compute-fresh
    "How is the top IP computed in parser.py?",
]

EDIT_ITEMS = [
    "Fix the off-by-one bug in parser.py.",
    "Rename slugify to to_slug everywhere in pkg/utils.py.",
    "Add a docstring to the Cache class in utils.py.",
    "Refactor parse_line in parser.py to use regex instead of split.",
    "Debug why top_ip returns the wrong IP sometimes.",
    "Remove the unused capacity check from Cache.put.",
    "Update parser.py so parts[5] is used instead of parts[2] for the method field.",
    "Can you clean up the Cache class so eviction is O(1)?",
    "Change Cache to evict the least-recently-used item instead of the oldest inserted.",
    "There's a typo in notes.md, please correct it.",
    "Patch parser.py so it doesn't crash on malformed log lines.",
    "Add type hints to every function in pkg/utils.py.",
    "This function has a bug: it always returns None instead of the count.",
    "Simplify the eviction logic in Cache.put.",
    "Optimize parse_line so it doesn't call .strip() twice.",
    "The parser.py script throws a KeyError on empty lines — please fix that.",
    # ambiguous: phrased as a computation complaint, but the ask is a fix
    "This script is supposed to sum a column but it just prints zero every "
    "time — what's going on?",
]

COMPUTE_ITEMS = [
    "What's the sum of the revenue column in sales.csv?",
    "Parse access.log and tell me which IP hit the server the most.",
    "How many 404s are in access.log?",
    "What's the average units sold per day in sales.csv?",
    "Parse this log and tell me the top IP.",
    "Count how many lines in access.log come from 10.0.0.1.",
    "What's the total revenue across all regions in sales.csv?",
    "Tell me the most common status code in access.log.",
    "Sum column 2 of data.csv.",
    "How many rows in sales.csv have units greater than 10?",
    "Calculate the median revenue value across the rows in sales.csv.",
    "What's the total number of requests logged in access.log?",
    "Group the sales by region and tell me which region made the most revenue.",
    "I have a CSV of expenses — what's the biggest single expense?",
    "Filter access.log down to just the POST requests and count them.",
    "What percentage of requests in access.log returned a non-200 status?",
]

AUTHOR_ITEMS = [
    "Write a script that reverses a string.",
    "Create a function that checks if a number is prime.",
    "Build me a tool that dedupes a list of email addresses.",
    "Write a script that reads a CSV and prints the average of column 3.",
    "Generate a Python module for basic unit conversions (miles to km, etc).",
    "Make me a small CLI that renames files in a folder to lowercase.",
    "Write a program that plays tic-tac-toe against the user.",
    "Create a class that represents a simple bank account with deposit/withdraw.",
    "Write a script that parses log files like access.log and reports the "
    "top 5 IPs.",
    "Author a quick tool that validates email addresses with a regex.",
    "I need a program that converts Celsius to Fahrenheit — can you make one?",
    "Build a small library for retrying flaky HTTP calls with backoff.",
    "Write me a quick script to shuffle the lines of a text file.",
    "Create a simple script for converting Markdown to plain text.",
    "Can you put together a program that solves Sudoku puzzles?",
    "Generate a tool that batch-renames image files by date taken.",
    # ambiguous: "add a script ... save it as report.py" -> new file, not edit
    "Add a script to compute the average revenue and save it as report.py.",
]

ALL_ITEMS = ([(r, NAVIGATE) for r in NAVIGATE_ITEMS]
            + [(r, EDIT) for r in EDIT_ITEMS]
            + [(r, COMPUTE) for r in COMPUTE_ITEMS]
            + [(r, AUTHOR) for r in AUTHOR_ITEMS])


def build() -> None:
    """Write ALL_ITEMS to benchmark.jsonl, one {request, label} per line."""
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for request, label in ALL_ITEMS:
            handle.write(json.dumps({"request": request, "label": label},
                                    ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build()
    lines = OUT_PATH.read_text().strip().splitlines()
    assert len(lines) == len(ALL_ITEMS) >= 60
    counts: dict[str, int] = {}
    for line in lines:
        label = json.loads(line)["label"]
        counts[label] = counts.get(label, 0) + 1
    assert set(counts) == {NAVIGATE, EDIT, COMPUTE, AUTHOR}
    print(f"wrote {len(lines)} items: {counts}")
    print("smoke OK")
