"""Coding vertical: the harness owns the file, the model writes one method.

See docs/DESIGN-coding-agent.md. The model is consulted only through
bounded generations (a plan, then one method body at a time); every
verification is deterministic (ast checks, a subprocess), so the file on
disk always parses by construction.
"""
