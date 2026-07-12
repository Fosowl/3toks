"""Command-line entry point: run one web-research episode.

    python3 -m threetoks.cli "When was the Eiffel Tower completed?" \
        --trace trace.jsonl
"""
import argparse
import time

from threetoks.backend.base import FAMILY_CHATML, FAMILY_R1, ModelSpec
from threetoks.backend.ollama import OllamaBackend
from threetoks.config import load_config
from threetoks.engine import run_episode  # noqa: F401 (eval harness imports)
from threetoks.policy import Policy, PolicyConfig
from threetoks.research import deep_research
from threetoks.services import Services
from threetoks.trace import Tracer
from threetoks.web.notes import NoteStore
from threetoks.web.vertical import WebResearchVertical

# The web extras (requests, bs4, markdownify) are OPTIONAL: everything
# that touches them imports lazily so the core runs on a bare install.

DEFAULT_MODEL = "gemma3:4b"
DEFAULT_MAX_STEPS = 25

# E1 winner ("menu_picker") extended to cover all three reply shapes.
SYSTEM_CHATML = (
    "You are a menu selector controlling a research agent. When shown a "
    "numbered list of actions, reply with exactly one digit: the number of "
    "the best action. When asked for sentence numbers, reply with only the "
    "fewest numbers separated by commas. Otherwise reply with the shortest "
    "correct answer. Do not explain.")


def model_spec(name: str) -> ModelSpec:
    """Infer the template family from the model name."""
    family = FAMILY_R1 if "r1" in name else FAMILY_CHATML
    return ModelSpec(name, family)


def make_policy_config(name: str) -> PolicyConfig:
    """Policy config with the family-appropriate system prompt."""
    spec = model_spec(name)
    system = SYSTEM_CHATML if spec.family == FAMILY_CHATML else ""
    return PolicyConfig(spec, system=system)


def fetch_page(url: str):
    """Fetch a URL and reduce it to numbered-sentence PageText."""
    from threetoks.web.fetch import fetch_html      # lazy: web extras
    from threetoks.web.textify import html_to_page
    return html_to_page(fetch_html(url), url)


def build_parser(config=None) -> argparse.ArgumentParser:
    """CLI arguments, with defaults drawn from ``config.ini``."""
    config = config or load_config()
    parser = argparse.ArgumentParser(description="ThreeToks web research")
    parser.add_argument("task", help="research question to answer")
    parser.add_argument("--model", default=config.llm.model)
    parser.add_argument("--max-steps", type=int,
                        default=config.research.max_steps)
    parser.add_argument("--trace", default=None, help="JSONL trace path")
    parser.add_argument("--rounds", type=int, default=config.research.max_rounds,
                        help="max judged research rounds")
    return parser


def make_fetch_page(config):
    """Build a ``(fetch_page, fetcher)`` pair for the configured browser mode.

    ``http`` uses the module-level plain-HTTP :func:`fetch_page` and no
    fetcher to close. ``plain`` / ``stealth`` lazily import their Chrome
    backend and return the live fetcher so the caller can close it.
    """
    mode = config.browser.mode
    if mode == "plain":
        from threetoks.web.browser import make_fetcher
        fetcher = make_fetcher(config.browser.visible)
        return fetcher.fetch_page, fetcher
    if mode == "stealth":
        from threetoks.web.stealth_browser import make_stealth_fetcher
        fetcher = make_stealth_fetcher(config.browser.visible)
        return fetcher.fetch_page, fetcher
    return fetch_page, None


def main() -> None:
    """Run one episode and print the outcome with basic accounting."""
    config = load_config()
    import threetoks.web.search as search_mod  # lazy: web extras
    from threetoks.web.search import auto_provider
    search_mod.MIN_SEARCH_INTERVAL_S = config.search.min_interval_s
    args = build_parser(config).parse_args()
    policy = Policy(OllamaBackend(), make_policy_config(args.model),
                    Tracer(args.trace))
    page_fetch, fetcher = make_fetch_page(config)
    services = Services(provider=auto_provider(), fetch_page=page_fetch,
                        max_research_rounds=args.rounds)
    start = time.time()
    try:
        outcome = deep_research(args.task, services, policy)
    finally:
        if fetcher is not None:
            fetcher.close()
    elapsed = time.time() - start
    print(f"\nANSWER: {outcome['answer']}")
    print(f"\nNOTES:\n{outcome['notes'] or '(none)'}")
    print(f"\nstats: {outcome['rounds']} rounds "
          f"(judged {'good' if outcome.get('judged_good') else 'exhausted'}), "
          f"queries: {outcome['queries']}, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
