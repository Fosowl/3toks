"""Append-only episode prompt assembly.

Prompt layout (KV-cache-first, see docs/DESIGN.md §4):

    [history]  static prefix + task + compacted log lines   (append-only)
    [observation]  current page / search results            (replaced at
                   page boundaries only)
    [session]  one-line actions taken on the current observation (append-only
               while the observation is open; folded into history when it
               closes)

Only the very end of the prompt changes between consecutive steps on the
same observation, so the backend's prompt cache covers everything else.
"""

HISTORY_SEPARATOR = "\n"


class Episode:
    """Grow-only prompt state for one agent episode."""

    def __init__(self, prefix: str, task: str):
        self._history: list[str] = [prefix, f"TASK: {task}"]
        self._observation = ""
        self._session: list[str] = []
        self.step_count = 0

    def log_session(self, line: str) -> None:
        """Append a one-line action record for the current observation."""
        self._session.append(line)

    def open_observation(self, text: str, closing_summary: str = "") -> None:
        """Replace the observation, folding the closed session into history.

        closing_summary is one line describing what the closed observation
        yielded (e.g. "> read 'France' page, noted 2 sentences"). This is
        the only point where earlier prompt content changes — one accepted
        cache re-ingest per page boundary.
        """
        if self._session or closing_summary:
            if closing_summary:
                self._history.append(closing_summary)
            self._history.extend(self._session)
            self._session = []
        self._observation = text

    def render_base(self) -> str:
        """Assemble the prompt body (no node menu / prefill)."""
        parts = [HISTORY_SEPARATOR.join(self._history)]
        if self._observation:
            parts.append(self._observation)
        if self._session:
            parts.append(HISTORY_SEPARATOR.join(self._session))
        return "\n\n".join(parts)


if __name__ == "__main__":
    episode = Episode("SYSTEM: choose actions by number.", "find X")
    episode.open_observation("[1] first page sentence")
    first = episode.render_base()
    episode.log_session("> noted s1")
    second = episode.render_base()
    assert second.startswith(first), "session append must preserve prefix"
    episode.open_observation("[1] second page", "> left page one")
    third = episode.render_base()
    assert "> noted s1" in third and "[1] first page sentence" not in third
    print("smoke OK")
