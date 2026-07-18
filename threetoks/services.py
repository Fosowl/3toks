"""Shared services injected into agents (search, fetch, file roots).

One Services object is built at REPL startup and handed to every agent
run; agents take what they need and ignore the rest.
"""
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RESEARCH_ROUNDS = 3


@dataclass
class Services:
    """Capabilities available to agents."""
    provider: object = None            # SearchProvider (web)
    fetch_page: object = None          # callable url -> PageText (web)
    files_root: Path = field(default_factory=Path.cwd)
    max_research_rounds: int = DEFAULT_RESEARCH_ROUNDS
    recalled: list = field(default_factory=list)  # memories the selector chose
    retriever: object = None           # code retrieval-as-repair (opt-in)
    relay: object = None               # GPIO light Relay (Pi relay extra)
    capture_frame: object = None       # callable -> base64 JPEG (camera extra)


if __name__ == "__main__":
    services = Services()
    assert services.files_root.exists()
    assert services.max_research_rounds == DEFAULT_RESEARCH_ROUNDS
    print("smoke OK")
