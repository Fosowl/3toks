"""The policy: consult the model at a decision node and parse the answer.

Owns everything that makes tiny-model decisions reliable:
- option permutation per attempt (defeats position bias),
- prefill steering + tight token caps (forces format),
- a retry ladder with temperature bumps,
- an optional bounded think phase for reasoning distills.
"""
import random
from dataclasses import dataclass

from threetoks.backend.base import (FAMILY_R1, ChatPrompt, GenOpts, GenResult,
                                   LLMBackend, ModelSpec, build_raw_prompt,
                                   family_stops)
from threetoks.nodes import Decision
from threetoks.render import Episode
from threetoks.trace import Tracer

# E4: temp-0 failures are confident, so attempt 2 is a pure reshuffle
# (recovers 29% of failures); temperature only enters at attempt 3.
TEMPERATURE_LADDER = (0.0, 0.0, 0.4)
VOTE_TEMPERATURE = 0.0
THINK_STOP = "</think>"
NODE_SEPARATOR = "\n\n"
RESOLVED_CLIP = 80


@dataclass(frozen=True)
class PolicyConfig:
    """Which model decides, and how hard it retries."""
    spec: ModelSpec
    system: str = ""
    seed: int = 0
    max_attempts: int = 3
    num_ctx: int = 2048
    vote_k: int = 1  # >1 = self-consistency voting on menu nodes


class Policy:
    """Turns (episode, node) into a Decision via the backend."""

    def __init__(self, backend: LLMBackend, config: PolicyConfig,
                 tracer: Tracer | None = None):
        self.backend = backend
        self.config = config
        self.tracer = tracer or Tracer(None)
        chat = getattr(backend, "chat", None)  # chat-API vs raw transport
        self._chat = chat if callable(chat) else None

    def decide(self, episode: Episode, node, think_budget: int = 0) -> Decision:
        """Run voting (menus, if configured) or the retry ladder."""
        if self.config.vote_k > 1 and node.kind == "menu":
            return self._decide_by_vote(episode, node)
        return self._decide_by_ladder(episode, node, think_budget)

    def _decide_by_ladder(self, episode: Episode, node,
                          think_budget: int) -> Decision:
        """Retry ladder; returns the last parse if all attempts fail."""
        decision = Decision(node.kind, None, "", valid=False)
        for attempt in range(self.config.max_attempts):
            perm = self._permutation(node, episode.step_count, attempt)
            user = episode.render_base() + NODE_SEPARATOR + node.render(perm)
            result = self._generate(user, node, attempt, think_budget)
            decision = node.parse(result.text, perm)
            self._record(episode, node, perm, attempt, result, decision)
            if decision.valid:
                break
        episode.step_count += 1
        return decision

    def _decide_by_vote(self, episode: Episode, node) -> Decision:
        """Majority vote over vote_k samples with different option orders.

        Permutation decorrelates position-bias errors between samples
        (E1/E4 data); temperature stays 0 so each vote is deterministic.
        Falls back to the retry ladder when no sample parses.
        """
        votes: list[Decision] = []
        for sample in range(self.config.vote_k):
            perm = self._permutation(node, episode.step_count, sample)
            user = episode.render_base() + NODE_SEPARATOR + node.render(perm)
            result = self._generate_plain(user, node)
            decision = node.parse(result.text, perm)
            self._record(episode, node, perm, sample, result, decision)
            if decision.valid:
                votes.append(decision)
        if not votes:  # ladder does the single step_count increment
            return self._decide_by_ladder(episode, node, think_budget=0)
        episode.step_count += 1
        counts: dict[object, int] = {}
        for vote in votes:
            counts[vote.value] = counts.get(vote.value, 0) + 1
        winner = max(counts, key=lambda value: counts[value])
        return next(vote for vote in votes if vote.value == winner)

    def _generate_plain(self, user: str, node) -> GenResult:
        """One completion at vote temperature, no think phase."""
        return self._call(user, self.config.system, node.prefill, node,
                          VOTE_TEMPERATURE)

    def _generate(self, user: str, node, attempt: int,
                  think_budget: int) -> GenResult:
        """One completion, optionally preceded by a bounded think phase.

        A node may pin its own temperature (``node.temperature``); the
        coding vertical uses this to resample a method body with diversity
        on a repair pass rather than climbing the attempt ladder. The
        think phase needs raw-mode prompt surgery, so it never runs over
        a chat transport.
        """
        override = getattr(node, "temperature", None)
        temperature = override if override is not None else \
            TEMPERATURE_LADDER[min(attempt, len(TEMPERATURE_LADDER) - 1)]
        prefix = ""
        if think_budget > 0 and self.config.spec.family == FAMILY_R1 \
                and self._chat is None:
            prefix = self._think(user, think_budget, temperature)
        system = getattr(node, "system", None) or self.config.system
        return self._call(user, system, prefix + node.prefill, node,
                          temperature)

    def _call(self, user: str, system: str, prefill: str, node,
              temperature: float) -> GenResult:
        """One completion via the chat or raw transport.

        Chat backends get un-templated parts and only the node's own
        stops (end-of-turn is the API's job); the raw path renders the
        family template and must let the family stops ride along.
        """
        images = tuple(getattr(node, "images", ()))
        if self._chat is not None:
            opts = GenOpts(max_tokens=node.max_tokens,
                           temperature=temperature,
                           stop=tuple(getattr(node, "stop", ())),
                           num_ctx=self.config.num_ctx, images=images)
            return self._chat(self.config.spec.name,
                              ChatPrompt(system, user, prefill), opts)
        prompt = build_raw_prompt(self.config.spec, user, system=system,
                                  prefill=prefill)
        opts = GenOpts(max_tokens=node.max_tokens, temperature=temperature,
                       stop=self._stops(node), num_ctx=self.config.num_ctx,
                       images=images)
        return self.backend.complete(self.config.spec.name, prompt, opts)

    def _stops(self, node) -> tuple[str, ...]:
        """The node's stop sequences plus the family's end-of-turn token.

        Setting any stop overrides the Modelfile's defaults in raw mode,
        so the family stop must always ride along or a generation node's
        custom stops let the model run past its own end of turn.
        """
        return tuple(getattr(node, "stop", ())) + family_stops(self.config.spec)

    def _think(self, user: str, budget: int, temperature: float) -> str:
        """Bounded reasoning phase; returns text to inject before the prefill.

        Uses an open <think> prompt stopped at budget or the closing tag,
        then the answer call continues from the closed think block — the
        shared prompt prefix keeps the second call cheap (KV reuse).
        """
        prompt = build_raw_prompt(self.config.spec, user,
                                  system=self.config.system, think=True)
        opts = GenOpts(max_tokens=budget, temperature=temperature,
                       stop=(THINK_STOP,), num_ctx=self.config.num_ctx)
        result = self.backend.complete(self.config.spec.name, prompt, opts)
        return f"<think>\n{result.text}\n{THINK_STOP}\n\n"

    def _permutation(self, node, step: int, attempt: int) -> tuple[int, ...]:
        """Deterministic option shuffle; empty for option-less nodes."""
        if node.kind != "menu":
            return ()
        rng = random.Random(self.config.seed * 1_000_003
                            + step * 101 + attempt)
        order = list(range(len(node.options)))
        rng.shuffle(order)
        return tuple(order)

    def _resolve_picks(self, node, decision: Decision) -> list[str] | None:
        """Map picked indices to the texts they point to, when known."""
        items = getattr(node, "items", None)
        if not (items and decision.valid and isinstance(decision.value, list)):
            return None
        return [items[i - 1][:RESOLVED_CLIP] for i in decision.value
                if 1 <= i <= len(items)]

    def _record(self, episode: Episode, node, perm: tuple[int, ...],
                attempt: int, result: GenResult, decision: Decision) -> None:
        """Trace one attempt."""
        resolved = self._resolve_picks(node, decision)
        self.tracer.record({
            **({"resolved": resolved} if resolved else {}),
            **({"tag": node.tag} if getattr(node, "tag", None) else {}),
            **({"mode": "chat"} if self._chat is not None else {}),
            "step": episode.step_count, "node": node.kind,
            "question": node.question[:120], "perm": list(perm),
            "attempt": attempt, "raw_text": result.text,
            "value": decision.value, "valid": decision.valid,
            "prompt_tokens": result.prompt_tokens,
            "out_tokens": result.out_tokens, "wall_s": round(result.wall_s, 3)})


if __name__ == "__main__":
    from threetoks.nodes import MenuNode

    class _FakeBackend:
        def __init__(self, scripted):
            self.scripted = list(scripted)
            self.prompts = []

        def complete(self, model, raw_prompt, opts):
            self.prompts.append(raw_prompt)
            return GenResult(self.scripted.pop(0), 10, 2, 0.01, "stop")

    backend = _FakeBackend(["garbage", " 2"])
    policy = Policy(backend, PolicyConfig(ModelSpec("m", FAMILY_R1)))
    episode = Episode("SYS", "t")
    decision = policy.decide(episode, MenuNode("Pick.", ["a", "b", "c"]))
    assert decision.valid and decision.value in {"a", "b", "c"}
    assert len(backend.prompts) == 2, "retry ladder should have re-asked"
    assert backend.prompts[0].endswith("ANSWER:")
    print("smoke OK")
