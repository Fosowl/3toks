"""Quick health probes: Ollama, its models, SearXNG, and API keys.

The config screen and ``threetoks doctor`` call :func:`sweep` once and show
the answers as hints beside each setting. Every probe is short, never
raises, and treats a bad answer as "not reachable".
"""
import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass

from threetoks.config import ThreetoksConfig

PROBE_TIMEOUT_S = 0.8
JOIN_GRACE_S = 0.5     # extra wait for a probe thread past its timeout
_WEB_SCHEMES = ("http://", "https://")
_ANTHROPIC_KEY = "ANTHROPIC_API_KEY"


@dataclass(frozen=True)
class Probes:
    """The answers of one sweep.

    ``models`` is None when Ollama is not asked or does not answer.
    ``key`` is ``(env name, is set)`` or None when no key is needed.
    """

    ollama_up: bool = False
    models: tuple | None = None
    searxng_up: bool = False
    key: tuple | None = None


def http_reachable(url: str, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Check that a web server answers at ``url`` (any status counts).

    Args: url (str): http or https address; timeout (float): seconds.
    Returns: True when a server replied, False otherwise.
    """
    if not url.startswith(_WEB_SCHEMES):
        return False
    try:
        urllib.request.urlopen(url, timeout=timeout).close()
    except urllib.error.HTTPError:
        return True  # the server answered with an error page
    except (OSError, ValueError):
        return False
    return True


def ollama_models(host: str, timeout: float = PROBE_TIMEOUT_S) -> tuple | None:
    """List the models an Ollama server has pulled.

    Args: host (str): Ollama base URL; timeout (float): seconds.
    Returns: tuple of model names, or None if the server does not answer.
    """
    if not host.startswith(_WEB_SCHEMES):
        return None
    url = host.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as reply:
            data = json.load(reply)
    except (OSError, ValueError):
        return None
    names = (item.get("name", "") for item in data.get("models", []))
    return tuple(name for name in names if name)


def key_status(provider: str) -> tuple | None:
    """Tell whether the API key of ``provider`` is in the environment.

    Args: provider (str): the ``[llm] provider`` name. Returns: None for
    ollama, else (environment variable name, True when it is set).
    """
    from threetoks.backend.providers import OPENAI_COMPAT
    if provider == "anthropic":
        env_name = _ANTHROPIC_KEY
    elif provider in OPENAI_COMPAT:
        env_name = OPENAI_COMPAT[provider][1]
    else:
        return None
    return env_name, bool(os.environ.get(env_name))


def _run_parallel(jobs: dict) -> dict:
    """Run each job in a thread and collect results by name.

    Args: jobs (dict): name -> zero-argument callable. Returns: dict
    name -> result, with None for a job that did not finish in time.
    """
    results = dict.fromkeys(jobs)
    def work(name, job):
        """Run one probe and store its answer."""
        results[name] = job()
    threads = [threading.Thread(target=work, args=pair, daemon=True)
               for pair in jobs.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(PROBE_TIMEOUT_S + JOIN_GRACE_S)
    return results


def sweep(config: ThreetoksConfig) -> Probes:
    """Probe the places ``config`` points at, in parallel.

    Args: config (ThreetoksConfig): the config to check. Returns: Probes.
    """
    jobs = {"search": lambda: http_reachable(config.search.searxng_url)}
    if config.llm.provider == "ollama":
        jobs["models"] = lambda: ollama_models(config.llm.host)
    found = _run_parallel(jobs)
    models = found.get("models")
    return Probes(ollama_up=models is not None, models=models,
                  searxng_up=bool(found["search"]),
                  key=key_status(config.llm.provider))


if __name__ == "__main__":
    assert http_reachable("file:///etc/passwd") is False
    assert ollama_models("http://127.0.0.1:9") is None
    assert key_status("ollama") is None
    assert key_status("anthropic")[0] == _ANTHROPIC_KEY
    shown = sweep(ThreetoksConfig())
    print("smoke OK", shown)
