# E9 — Retrieval-as-repair: fetch, gate, execute

Spike question: given a spec (function name, signature, one anchor assert
derived from an example input/output pair — exactly the contract
`threetoks/code/` already plans), can the harness find a **working**
implementation on the public internet instead of generating one, using
only free deterministic checks plus a subprocess as the judge? No model
call anywhere in this pipeline — search, name matching, safety gates, and
correctness are all harness code or `ast`/`subprocess`.

This is a REPAIR fallback, not a replacement for generation: the target
use case is CodeVertical's own known failure tail (spike E6: `roman_to_int`
and a caesar cipher both failed on a live 1.5b) — classic, widely-implemented
functions where the internet almost certainly has a correct answer sitting
somewhere, and a wrong retrieval is caught for free by the anchor.

## Headline numbers

| | |
|---|---|
| Classic hit rate | **7/8** (87.5%) — includes both E6 failures, `roman_to_int` and `caesar_encode` |
| Non-classic (invented) hit rate | **0/4** — no false accepts |
| Avg wall-clock per classic hit | 9.7s |
| Avg wall-clock per classic miss | 11.7s (the one real miss, `binary_search`, cost about as much as a hit) |
| Avg wall-clock per non-classic task | 8.8s |
| Total benchmark wall-clock | ~203s across two passes (see "How this run actually went" below) — **zero model/Ollama calls** |

The one classic miss (`binary_search`) was an honest miss: two real
candidates were found, gated through, and executed — both failed the
anchor on contract mismatch / a broken variable reference, not on a
search failure. All four non-classic misses were also honest: either zero
search hits, zero name matches among real fetches, or (`zigzag_join`) a
fuzzy-matched real function that correctly failed the anchor. See the
funnel table below for the full trace.

## How this run actually went (report this honestly, it's part of the finding)

The local SearXNG instance at `localhost:8080` worked at the start of this
session. Over the course of probing it manually (about 20 ad-hoc curl
calls while scoping the task) and then running the automated benchmark
(24 queries in ~2 minutes), its upstream engines tripped rate limits one
by one — visible on the instance's own `/stats` page as `Suspended: too
many requests` / `CAPTCHA` — first brave, duckduckgo, qwant, startpage,
wikipedia, and eventually even `google`. **Mojeek** (`www.mojeek.com/search`,
discovered mid-task as an unblocked plain-HTML hop after Bing/DDG/Startpage
all turned out to be bot-walled) also returned a 403 after carrying several
queries in a row. Both recovered on their own within about 45 seconds of
reduced load.

Concretely: the first full 12-spec pass scored 3/8 classic (the first 3
specs, before the search layer went quiet); the remaining 9 specs got zero
search results in that pass. A second pass — after the ~45s cooldown, with
extra pacing (12s between specs on top of the built-in 5s inter-query
pace) — recovered search for all 9 remaining specs and produced the
7/8 + 0/4 numbers above. The merged `results.json` is the final dataset;
nothing here is cherry-picked, both passes are reflected in the funnel
table's search-hit counts.

**Reading this honestly**: this is a real operational finding, not a
fluke. A single local SearXNG instance plus two unauthenticated HTML
scrapers is a shallow, rate-limit-prone resource; a burst of ~15 queries
per minute (well within what an automated repair loop would want) exhausts
it. Retrieval-as-repair, if it ships, needs either much longer pacing than
the 5s baked into `threetoks/web/search.py` today, a rotation across more
independent search backends, or to be reserved for genuinely rare
repair events rather than routine use.

## Providers — what actually worked, from this machine

| Hop | Result |
|---|---|
| `localhost:8080` SearXNG (general web search) | **WORKED**, until this session's own query volume rate-limited/CAPTCHA'd every one of its 6 upstream engines in turn (see above). A fresh instance would not start in this state. |
| Mojeek (`www.mojeek.com/search`) | **WORKED** — plain HTML, no JS challenge, until it too returned 403 after carrying several queries; recovered within ~45s. Discovered specifically because Bing/DDG/Startpage were all dead. |
| Bing HTML (`bing.com/search`) | **BLOCKED** — every query, including plain queries with no `site:` filter, returned a captcha/"verify" challenge page. Matches `threetoks/web/search.py`'s own documented "Bing challenge shell" failure signature. |
| DuckDuckGo HTML — both `html.duckduckgo.com` and `lite.duckduckgo.com` | **BLOCKED** — both return the "anomaly" challenge page. Network-wide block on duckduckgo.com, not one endpoint. |
| Startpage HTML | **BLOCKED** — redirects into a captcha interstitial. |
| grep.app API (`grep.app/api/search`) | **BLOCKED** as predicted — returns a Vercel "Security Checkpoint" bot-check page even to a plain curl with a browser UA. Confirmed once, abandoned per the task's own guidance. |
| searchcode.com `/api/codesearch_I/` | **DEAD** — 404, endpoint no longer exists. |
| GitHub code search — REST `/search/code` | **UNUSABLE unauthenticated** — 401 `Requires authentication`. `GITHUB_TOKEN` is not set in this environment and no `gh` CLI is installed, exactly as flagged going in. |
| GitHub code search — web UI (`github.com/search?type=code`) | **UNUSABLE unauthenticated** — returns HTTP 200 but the embedded JSON payload is `"results":[],"result_count":0,"logged_in":false`. GitHub silently empties code-search results for logged-out requests; it is not merely slow or paginated. |
| `api.github.com` repo metadata / git trees (unauthenticated) | **WORKED** — 60 req/hour. Used only to turn a bare `github.com/<owner>/<repo>` search hit into a shortlist of `.py` files (2 API calls per repo, cached), never for search itself. Never hit the rate limit in this run (54-55/60 remaining throughout). |
| `raw.githubusercontent.com` | **WORKED** — not bot-walled, used for every GitHub-sourced fetch. |

**Notable outcome**: despite building GitHub-specific plumbing (blob-URL
rewrite, repo-root → tree-listing → raw-fetch), **none of the 7 accepted
snippets in this run actually came from GitHub.** Two GitHub-sourced
`caesar_cipher`/`caesar_encode` candidates *were* found, fetched, gated,
and executed for the `caesar_encode` task — both correctly rejected (wrong
arity: they took a `mode` argument the spec's anchor didn't supply) — but
the winning candidate for every one of the 7 hits came from a non-GitHub
page (Stack Overflow, Code Review Stack Exchange, a tutorial site, a
blog, or an ActiveState recipe) via the generic `<pre>`/`<code>`
block extraction path. For this benchmark, the "workable fallback" (a
general search engine plus HTML code-block scraping) carried the whole
result; the GitHub-specific hop added coverage (two more real, correctly-
rejected candidates for `caesar_encode`) without ever being the accept
path.

A bug caught live during the first dry run: `gist.github.com` and
`github.com/topics/...` URLs matched a naive `"github.com" in url`
substring check and got misrouted into the repo-tree resolver (which then
wasted an API call on a guaranteed-404 "repo"). Fixed by checking
`urlparse(url).netloc == "github.com"` exactly — see `fetchers.py`'s
`_is_plain_github_com` and the regression asserts in its own smoke block.

## The funnel, per task

Funnel columns: **search** = total search results across this spec's
queries (after the FallbackProvider chain) · **fetched** = "files"
pulled out of those URLs (a GitHub repo-root can yield several; an HTML
page can yield several `<pre>` blocks; each counted independently) ·
**matched** = top-level defs whose name matched the spec (exact/substring/
fuzzy) across every fetched file · **gated** = matched defs that survived
`assemble()` (imports/helpers resolved, safe, standalone-parseable) and
were actually executed against the anchor · **result** = first anchor pass
wins.

| # | spec | classic? | search | fetched | matched | gated | result | wall-clock | accepted source |
|---|---|---|--:|--:|--:|--:|---|--:|---|
| 1 | roman_to_int | yes | 16 | 19 | 10 | 2 | **HIT** | 6.8s | code.activestate.com/recipes/81611-roman-numerals |
| 2 | caesar_encode | yes | 8 | 14 | 5 | 3 | **HIT** | 15.6s | caesarcipher.org/learn/python-caesar-cipher-... |
| 3 | levenshtein | yes | 8 | 10 | 10 | 1 | **HIT** | 9.4s | askpython.com/python/examples/levenshtein-python-... |
| 4 | gcd | yes | 8 | 27 | 18 | 1 | **HIT** | 9.8s | stackoverflow.com/questions/11175131 |
| 5 | is_prime | yes | 8 | 23 | 20 | 3 | **HIT** | 9.7s | stackoverflow.com/questions/15285534 |
| 6 | binary_search | yes | 8 | 4 | 2 | 2 | miss | 11.7s | — (both candidates failed the anchor) |
| 7 | fibonacci | yes | 8 | 9 | 9 | 8 | **HIT** | 8.5s | waynelambert.dev/blog/post/fibonacci-sequence-algorithm-python |
| 8 | is_palindrome | yes | 8 | 37 | 27 | 2 | **HIT** | 8.0s | codereview.stackexchange.com/questions/236360 |
| 9 | snake_to_camel_but_keep_first | no | 0 | 0 | 0 | 0 | miss (honest) | 6.2s | — (search returned nothing at all for this invented name) |
| 10 | zigzag_join | no | 8 | 8 | 3 | 1 | miss (honest) | 9.9s | — (fuzzy `zig_zag_line` gated, then failed the anchor: wrong arity) |
| 11 | weighted_vowel_score | no | 8 | 14 | 0 | 0 | miss (honest) | 8.3s | — (real pages fetched, nothing name-matched) |
| 12 | title_case_odd_words | no | 8 | 21 | 0 | 0 | miss (honest) | 10.9s | — (real pages fetched, nothing name-matched) |

Two illustrative rejection-before-execution traces from `zigzag_join`
(both from real fetched pages, both correctly stopped by the safety
gates before any subprocess ran):
- a `zig_zag_line` candidate (fuzzy-matched, score 0.80) from a
  turtle-graphics Stack Overflow answer was rejected with
  `unsafe_import:turtle` (not on the whitelist) — a second def of the
  same name on the same page *did* pass the safety gate and ran, then
  correctly failed the anchor on arity.
- a code-golf answer's one-letter function `z` was rejected with
  `forbidden_builtin:exec` before ever reaching a subprocess.

And the `binary_search` near-miss in full, since it's the one classic
task that came closest without landing:
```
candidate binary_search (exact) from stackoverflow.com/.../binary-search-using-a-recursive-function
  -> fail: TypeError: binary_search() missing 2 required positional arguments: 'lower' and 'upper'
candidate binary_search (exact) from stackoverflow.com/.../binary-search-using-a-recursive-function
  -> fail: NameError: name 'recursions' is not defined
```
Same page, two different `def binary_search` blocks (the asker's buggy
original and a respondent's rewrite with its own bug) — the anchor
correctly rejected both rather than accepting a subtly wrong recursive
variant with a different signature.

## Provenance of every accepted snippet

All seven live in `accepted_snippets/*.py`, saved verbatim (only the
matched `def` was renamed to the spec's canonical name; any sibling
helper/import it needed was pulled in unmodified) with a header recording
the source URL. **License detection was not performed** — this is
explicitly future work, not a gap papered over; see Safety below.

| spec | source | what it is |
|---|---|---|
| roman_to_int | code.activestate.com/recipes/81611-roman-numerals | Community ActiveState Python recipe (owner: recipe contributed to activestate's public recipe archive, not a github repo) |
| caesar_encode | caesarcipher.org/learn/... | A tutorial site's worked example, full docstring included |
| levenshtein | askpython.com/python/examples/... | Tutorial-site worked example |
| gcd | stackoverflow.com/questions/11175131 | An answer on a Stack Overflow question (classic Euclidean algorithm) |
| is_prime | stackoverflow.com/questions/15285534 | An answer on a Stack Overflow question |
| fibonacci | waynelambert.dev/blog/post/fibonacci-sequence-algorithm-python | A blog post's worked example (recursive definition) |
| is_palindrome | codereview.stackexchange.com/questions/236360 | A Code Review Stack Exchange answer, with doctest-style examples in its own docstring |

None of the seven came from a github.com repository in this run (see
"Providers" above for why: two GitHub candidates for `caesar_encode` were
found and correctly rejected, but no GitHub candidate ever won).
"Owner/repo" as literally requested by the task doesn't apply to any of
these — they're StackOverflow/CodeReview posts, a tutorial site, a blog,
and a community recipe archive. Recording provenance for these took the
form of the source URL, which is what's saved in each file's header.

## False-accept analysis: is one anchor example enough?

**Zero outright false accepts** in this run — no non-classic (invented)
task's anchor was passed by a wrong candidate, and no classic candidate's
anchor pass was later found to be a wrong implementation by casual
inspection... except one, and it's instructive:

**`is_prime` is a real single-anchor blind spot.** The accepted candidate
is:
```python
def is_prime(n):
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```
The anchor was `is_prime(17) == True`, which this passes. But this
implementation is wrong for `n <= 1`: `is_prime(1)` and `is_prime(0)` both
return `True` (the loop range is empty, so it falls through to the
default `return True`) — it never special-cases "not prime by
definition." A second anchor example, e.g. `is_prime(1) == False` (or
`is_prime(4) == False` to also catch a `%`-typo class of bug), would have
caught this immediately and forced this candidate to be rejected in favor
of a correct one (several other `is_prime` candidates were fetched in the
same run — 20 name-matched, only 3 gated and tried before this one won).

This directly answers the task's question: **one anchor example is not
always a strong enough filter.** It is strong evidence against a
*wildly* wrong implementation (wrong algorithm, wrong output shape, wrong
arity — which is what rejected every other candidate in this benchmark),
but it says nothing about behavior outside the one example's input, and
"edge cases near the domain boundary" (0, 1, negative numbers, empty
input, off-by-one indices) are exactly where classic-function
implementations most often diverge. Two anchors spanning a "normal" case
and a boundary case would close most of this gap for the same reason the
greenfield vertical's design already leans on trusted examples rather
than model-graded ones (docs/DESIGN-coding-agent.md §4) — it's just that
*retrieved* code, unlike harness-generated code, was never conditioned on
the anchor at all, so its coverage of the anchor's specific edge cases is
pure luck.

The **safety gates**, by contrast, worked exactly as designed and did
produce real rejections in this run without ever needing execution:
`unsafe_import:turtle` and `forbidden_builtin:exec` both fired on real
fetched candidates (see the funnel section above) — proof the tight
whitelist matters in practice, not just in the fixture tests.

## Safety — read this before reusing any of this code for anything real

This spike executes code fetched from the open internet. The measures
taken:
- A **much tighter stdlib whitelist** than the greenfield vertical's
  (`safety_gates.SAFE_IMPORT_WHITELIST`): pure computation modules only
  (`math`, `re`, `string`, `collections`, `itertools`, `functools`,
  `operator`, `textwrap`, `unicodedata`, `statistics`, `fractions`,
  `decimal`, `bisect`, `heapq`, `array`, `cmath`). `os`, `sys`,
  `subprocess`, `socket`, `shutil`, `ctypes`, `importlib` are all excluded,
  and any import outside the whitelist rejects the candidate before it is
  ever assembled into runnable source.
- A **forbidden-builtins scan** (`eval`, `exec`, `open`, `__import__`,
  `getattr`, `setattr`, `delattr`, `compile`, `globals`, `locals`, `vars`,
  `input`, and a couple more) that fires on a bare `ast.Name` reference
  anywhere in the candidate or any helper it pulls in — this is what
  caught the code-golf `exec` candidate above.
- A best-effort dunder-attribute trip wire (any `.__something__` access
  other than `__name__`) aimed at classic sandbox-escape gadgets
  (`().__class__.__mro__...`) — **this is not a hardened sandbox check**,
  just a speed bump; a determined adversarial payload could construct
  such an escape without tripping this specific pattern-match.
- Every execution happens in a **fresh `python3 -` subprocess** (reusing
  `threetoks/code/gates.execute_asserts` unmodified) with a **5-second
  timeout** and **no interactive stdin** (the module source is piped in
  via `input=`, which closes stdin immediately after — a stray `input()`
  call gets `EOFError`, not a hang). Nothing fetched is ever `exec`'d or
  `import`'d in this process's own Python interpreter.

**What this does NOT provide, and must not be assumed for any real use:**
this is a subprocess sandbox with an import whitelist and a builtin
denylist, not an OS-level sandbox. It has no seccomp/container/VM
isolation, no network-namespace restriction (a whitelisted module could
still, in principle, be coerced into doing something unintended within
its own semantics — e.g. `re` with a pathological pattern is a ReDoS
risk this spike does not defend against), no memory/CPU limit beyond the
5s wall-clock timeout, and no protection against a Python interpreter
vulnerability. **License detection was not performed at all** — every
accepted snippet's provenance is recorded as a source URL only; shipping
retrieved code into any real system would need an actual license check
(most of what got fetched here is StackOverflow/CC-BY-SA-licensed content,
a tutorial site with no stated license, and one community code recipe —
none of that is unambiguously "free to embed verbatim in another
project," and this spike does not resolve that question). Anyone
reusing this pipeline for more than a research spike needs to add real
sandboxing (a container or VM, not a subprocess) and a real license
pipeline before executing or shipping anything it retrieves.

## Open problems / what's left

- **Search fragility dominates the story.** As documented above, the
  search layer (a single local SearXNG instance plus opportunistically-
  discovered Mojeek) could not sustain the query volume of even a
  12-task benchmark without tripping rate limits partway through. Any
  real deployment of retrieval-as-repair needs either much more
  conservative pacing than `threetoks/web/search.py`'s 5s baseline, a
  wider roster of independent backends, or to treat search calls as a
  scarce resource gated behind "generation genuinely failed," not a
  routine first move.
- **GitHub-specific plumbing never won a single accept in this run**
  despite working correctly (it did fetch, gate, and correctly reject
  two real GitHub candidates for `caesar_encode`). Whether that's sample
  size (12 specs) or a real pattern (StackOverflow/tutorial prose is
  simply more thoroughly indexed by general web search than raw GitHub
  file contents are) is not resolved by this spike.
- **One anchor is not always enough** (see `is_prime` above) — the
  natural next experiment is re-running with 2 anchors per spec (a
  normal case + a boundary case) and measuring whether that would have
  changed the accept for every task, not just caught `is_prime` in
  hindsight.
- **Helper-pulling is one-hop-deep and name-only.** `assemble()` chases a
  sibling top-level function or constant the target calls, up to
  `MAX_HELPER_HOPS = 2`, but has no real scope analysis — a name bound in
  a comprehension, a class body, or via multiple assignment targets
  (`X, Y = 1, 2`) is invisible to `top_level_assigns` and will correctly
  fail the standalone-parse gate rather than silently doing the wrong
  thing, but it does mean some genuinely-correct candidates are rejected
  as unresolvable when a smarter picker would keep them. This fails
  *safe* (a real correct candidate becomes a false miss, never a false
  accept), matching the same philosophy `threetoks/code/gates.py`'s
  conservative `undefined_names` already uses.
- **A recursive self-reference bug was caught and fixed live**: the
  first `fibonacci` accept came back with its own `def` duplicated,
  because the dependency-puller treated the function's own name (called
  recursively) as an unresolved sibling and re-pulled itself. Harmless to
  correctness (Python happily accepts two identical defs) but untidy;
  fixed in `extract.assemble` by seeding the "already resolved" set with
  the candidate's own pre-rename name. Re-verified with a clean re-run
  (`accepted_snippets/fibonacci.py` now has exactly one `def`).
- **No fuzzy-match false accepts occurred, but the risk is visible.**
  `zigzag_join`'s fuzzy match (`zig_zag_line`, score 0.80) against an
  unrelated turtle-graphics function is a case where name similarity
  alone would have been a bad reason to accept — the anchor caught it
  (wrong arity) for free. This is direct evidence for the task's design
  premise: relevance judgment doesn't need a model, because the
  execution judge doesn't care how the candidate was found.

## Files

- `specs.py` — the 12 benchmark specs (8 classic incl. `roman_to_int` /
  `caesar_encode`, 4 invented) with canonical name, aliases, anchor
  assert, and search queries.
- `safety_gates.py` — the tight foreign-code whitelist/denylist (ast
  only, no subprocess).
- `extract.py` — name matching (exact/substring/fuzzy via `difflib`) and
  candidate assembly (imports/helpers/consts pulled in, renamed to the
  canonical name, re-parsed standalone).
- `judge.py` — thin wrapper around `threetoks/code/gates.run_asserts`
  for per-candidate timing.
- `fetchers.py` — the provider chain (SearXNG -> Mojeek -> Bing -> DDG)
  plus GitHub blob/repo-tree resolution and generic `<pre>`/`<code>`
  extraction for non-GitHub pages.
- `pipeline.py` — per-spec funnel orchestration and logging
  (`run_task`), capped at `MAX_URLS_PER_SPEC = 6` /
  `MAX_CANDIDATES_TRIED = 8` to bound wall-clock.
- `run_benchmark.py` — the real entry point: `python3
  spikes/e9_code_retrieval/run_benchmark.py` from the repo root, writes
  `results.json` and `accepted_snippets/*.py`.
- `results.json` — full per-task funnel + attempt log for this run
  (merged from two passes; see "How this run actually went").
- `accepted_snippets/*.py` — the seven verbatim accepted candidates with
  provenance headers.
- `tests/test_extract_gate.py` + `tests/fixtures/*` — the offline
  unittest (no network) covering: a good candidate surviving the full
  funnel and passing its anchor, an unsafe import rejected before
  execution ever runs, HTML `<pre>`-block extraction feeding the same
  funnel, and a syntactically broken file yielding zero candidates.
  Run with `python3 -m pytest spikes/e9_code_retrieval/tests/test_extract_gate.py -q`.

Every module (`specs.py`, `safety_gates.py`, `extract.py`, `judge.py`,
`fetchers.py`, `pipeline.py`) ends with an offline `if __name__ ==
"__main__":` smoke block per repo convention; only `fetchers.py`'s search
classes themselves are untested offline (network-dependent by nature) —
its parsing helpers (`_parse_mojeek_results`, github URL resolution,
`<pre>`-block extraction) are.
