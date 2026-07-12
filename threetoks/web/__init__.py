"""Web IO layer for the research vertical.

Deterministic plumbing the harness owns (see docs/DESIGN.md §8): search
providers, HTTP fetch, HTML-to-numbered-sentences, and a provenance-tracked
note store. The model never sees raw HTML — only rendered sentences — and
notes point at verbatim content rather than rewriting it.
"""
