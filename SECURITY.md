# Security Policy

ThreeToks is a local, single-user agent framework: it runs on your machine,
talks to a local Ollama instance, and (with the `web`/`browser` extras)
fetches pages from the open internet and can execute model-written Python.
There is no hosted service and no multi-tenant deployment to worry about,
but the local trust boundary still matters — see "Known risk areas" below
before running it against untrusted input.

## Supported versions

ThreeToks is pre-1.0 and moves fast; only the `main` branch / latest
released version is supported with security fixes. There is no long-term
support branch.

## Reporting a vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

Email **mlg.fcu@gmail.com** with:

- A description of the issue and its impact.
- Steps to reproduce (a minimal `python3 -m threetoks.cli` invocation or
  `--trace` capture is ideal).
- The version/commit you tested against.

You should get an acknowledgment within a few days. There's no bug bounty
— this is a small open-source project — but you'll be credited in the fix
commit/changelog unless you'd rather stay anonymous. Once a fix is ready,
we'll coordinate on a disclosure timeline with you before any public
writeup.

## Known risk areas

These are inherent to what the project does, not bugs by themselves —
be deliberate about exposure when integrating or extending them:

- **The `code` agent executes model-generated Python.**
  `threetoks/code/gates.py` and `runner.py` run model-written modules via
  `subprocess` (import + trusted-example asserts + zero-arg smoke calls).
  This is arbitrary code execution *by design*, gated only by deterministic
  static checks (AST parse, signature/undefined-name checks), not a
  sandbox. Do not point the `code` agent at a task whose generated output
  you wouldn't run yourself, and do not run ThreeToks as a privileged user.
- **The `web` agent fetches arbitrary URLs and (in `stealth`/`plain`
  browser mode) drives a real Chrome instance.** Fetched page content
  becomes part of the model's context and can influence downstream
  decisions (search queries, note selection, follow-up fetches) — treat
  it as untrusted input. It cannot directly execute code, but a page
  designed to manipulate a small model's menu choices is a realistic
  prompt-injection vector; report cases where fetched content changes
  agent *behavior* beyond what a menu decision should allow (e.g.
  escaping the note-taking sandbox, causing an out-of-scope fetch).
- **The `files` agent is read-only and sandboxed** to `services.files_root`
  via `resolve()` + `is_relative_to` with symlink-escape checks
  (`threetoks/agents/files.py`). A bypass of that sandbox (path traversal,
  symlink escape, or reading outside `files_root`) is a valid report.
- **Config and memory files are plain, unencrypted local files**
  (`config.ini`, `threetoks_memory.json`). Don't put secrets in them; they
  are not designed as a credential store.
- **No network service is exposed.** ThreeToks only makes outbound
  requests (to Ollama, to search providers, to fetched pages) — it does
  not listen on a port. Reports assuming an exposed listener don't apply
  unless you've changed that yourself.

## Dependencies

The core package is stdlib-only. The `web` extra depends on `requests`,
`beautifulsoup4`, and `markdownify`; `browser` depends on `selenium`. Keep
these current — `uv pip install -e ".[web,browser]" -U` — since page
fetching/parsing is the most direct exposure to untrusted content.
