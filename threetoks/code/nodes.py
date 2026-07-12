"""Generation nodes for the coding vertical.

Unlike the menu/pick nodes, the model here writes bounded multi-line text:
a plan, a method body, or test asserts. Each node is duck-typed to what
the policy reads (kind, prefill, max_tokens, stop, render, parse, question)
so it flows through the same retry ladder without touching the core nodes.
A body/plan node's `valid` flag encodes only *syntactic* success, so the
ladder resamples on a parse failure; correctness is the vertical's job.
"""
import ast

from threetoks.code import gates, plan
from threetoks.nodes import Decision

PLAN_PREFILL = "FUNCTIONS:\n1. "
PLAN_MAX_TOKENS = 140
IMPLEMENT_MAX_TOKENS = 200
TEST_MAX_TOKENS = 60
IMPLEMENT_STOP = ("\ndef ", "\nclass ", "\nif __name__", "\n@", "\nprint(")

# The E5-winning system prompts, one per generation kind.
PLAN_SYSTEM = "You plan Python programs as a list of small functions."
IMPLEMENT_SYSTEM = ("You are a Python programmer. Write exactly one small "
                    "Python function with a complete, working body. Output "
                    "only code, no explanations, no markdown. Never leave it "
                    "unfinished: no pass, no ..., no raise NotImplementedError.")
TEST_SYSTEM = "You write Python assert statements to test a function."


class PlanNode:
    """Ask for 3-4 method signatures; parse leniently into ParsedMethods."""

    kind = "plan"
    tag = "plan"
    system = PLAN_SYSTEM
    prefill = PLAN_PREFILL
    max_tokens = PLAN_MAX_TOKENS
    stop = ("\n\n",)

    def __init__(self, goal: str):
        self.question = f"plan the functions for: {goal}"

    def render(self, perm: tuple[int, ...]) -> str:
        """One instruction line; the goal is already in the episode task."""
        return (f"List the {plan.MAX_METHODS} or fewer small functions to "
                "write, one per line, as name(args): purpose. "
                "Helpers before the functions that call them.")

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Recover methods from the prefill-prefixed completion."""
        methods = plan.parse_plan(self.prefill + text)
        return Decision(self.kind, methods, text, valid=bool(methods))


class ImplementNode:
    """Ask for one method body; valid iff it parses as the named function."""

    kind = "implement"
    tag = "impl"
    system = IMPLEMENT_SYSTEM
    stop = IMPLEMENT_STOP
    max_tokens = IMPLEMENT_MAX_TOKENS

    def __init__(self, name: str, args: str, contract: str,
                 examples: list[str] | None = None,
                 temperature: float | None = None):
        self.name = name
        self.args = args
        self.contract = contract
        self.examples = list(examples or [])
        self.temperature = temperature
        self.prefill = f"def {name}({args}):\n    "
        self.question = f"write {name}({args})"

    def render(self, perm: tuple[int, ...]) -> str:
        """The one method's spec; goal and siblings live in the episode."""
        lines = ["Write this function with a real, working body.",
                 f"{self.name}({self.args}): {self.contract}"]
        for example in self.examples:
            lines.append(f"Example: {example}")
        lines.append("Do not use pass or raise NotImplementedError — "
                     "actually implement it.")
        return "\n".join(lines)

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Reconstruct and trim to the named function; valid iff it parses."""
        source = gates.function_source(self.name, self.args, text)
        return Decision(self.kind, source, text, valid=source is not None)


class TestNode:
    """Ask for a couple of asserts; always valid (asserts are advisory)."""

    kind = "test"
    tag = "test"
    system = TEST_SYSTEM
    prefill = "assert "
    max_tokens = TEST_MAX_TOKENS
    stop = ("\n\n",)

    def __init__(self, name: str, args: str):
        self.name = name
        self.question = f"test {name}"

    def render(self, perm: tuple[int, ...]) -> str:
        """Request two asserts, one per line."""
        return (f"Write 2 assert statements that test {self.name}. "
                "One per line.")

    def parse(self, text: str, perm: tuple[int, ...]) -> Decision:
        """Keep well-formed single-assert lines; never fail the ladder."""
        asserts = _valid_asserts(self.prefill + text, self.name)
        return Decision(self.kind, asserts, text, valid=True)


def _valid_asserts(text: str, name: str) -> list[str]:
    """Lines that parse to one assert and mention the function by name."""
    kept: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("assert") or f"{name}(" not in line:
            continue
        try:
            tree = ast.parse(line)
        except (SyntaxError, ValueError):
            continue
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert):
            kept.append(line)
    return kept


if __name__ == "__main__":
    plan_decision = PlanNode("count words").parse(
        "count_vowels(text): counts vowels\n2. tally(words): count each", ())
    assert [m.name for m in plan_decision.value] == ["count_vowels", "tally"]
    impl = ImplementNode("clamp", "value, low, high", "limit to range")
    good = impl.parse("return max(low, min(high, value))", ())
    assert good.valid and "def clamp" in good.value
    assert not impl.parse("return (((", ()).valid          # syntax error
    tests = TestNode("add", "a, b").parse("add(1, 2) == 3\nassert add(0,0)==0", ())
    assert tests.valid and tests.value == [
        "assert add(1, 2) == 3", "assert add(0,0)==0"], tests.value
    assert TestNode("add", "a, b").parse("1 == 1\nassert True", ()).value == []
    print("smoke OK")
