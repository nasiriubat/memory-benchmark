"""Consistency evaluator with a per-task fact ledger.

For DFx and MemoryArena, we build a fact ledger per task containing:
constraints, accepted decisions, user feedback, prior recommendations,
corrected mistakes, and expected facts. We score whether later outputs
contradict the ledger. For deterministic scenarios we use stored facts; for
open text we fall back to an LLM judge with a strict JSON rubric.

Returns a score in [0, 1] (1.0 = fully consistent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.llms.base import LLMClient
from benchmark.logging import get_logger
from benchmark.metrics import consistency_from_contradictions
from benchmark.utils import extract_json_object, token_set

_log = get_logger("benchmark.eval.consistency")


@dataclass
class FactLedger:
    """Accumulated task facts that later outputs must not contradict."""

    constraints: list[str] = field(default_factory=list)
    accepted_decisions: list[str] = field(default_factory=list)
    user_feedback: list[str] = field(default_factory=list)
    prior_recommendations: list[str] = field(default_factory=list)
    corrected_mistakes: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)

    def all_facts(self) -> list[str]:
        return (
            self.constraints
            + self.accepted_decisions
            + self.user_feedback
            + self.prior_recommendations
            + self.corrected_mistakes
            + self.expected_facts
        )

    def add_from_subtask(self, *, question: str, expected: str | None, metadata: dict[str, Any]) -> None:
        """Update the ledger from a subtask's known facts/metadata."""
        for key, target in (
            ("constraints", self.constraints),
            ("accepted_decisions", self.accepted_decisions),
            ("user_feedback", self.user_feedback),
            ("prior_recommendations", self.prior_recommendations),
            ("corrected_mistakes", self.corrected_mistakes),
            ("expected_facts", self.expected_facts),
        ):
            value = metadata.get(key)
            if isinstance(value, str):
                target.append(value)
            elif isinstance(value, list):
                target.extend(str(v) for v in value)
        if expected:
            self.expected_facts.append(expected)


_CONSISTENCY_SYSTEM = (
    "You are a strict consistency judge. Given a list of established FACTS "
    "(constraints, accepted decisions, feedback, prior recommendations) and a "
    "new AGENT OUTPUT, determine whether the output contradicts any fact. "
    'Respond ONLY with JSON: {"score": <float 0.0-1.0 where 1.0 means fully '
    'consistent>, "reason": "<short>"}.'
)


def deterministic_consistency(output: str, ledger: FactLedger) -> tuple[float, int, int]:
    """Heuristic contradiction check using negation + fact-token overlap.

    Returns (score, total_checks, contradictions). This is a conservative
    deterministic signal; the LLM judge is preferred for open text.
    """
    facts = [f for f in ledger.all_facts() if f.strip()]
    if not facts:
        return 1.0, 0, 0
    out_tokens = token_set(output)
    negations = {"not", "no", "never", "cannot", "wont", "dont", "avoid", "without"}
    contradictions = 0
    for fact in facts:
        fact_tokens = token_set(fact)
        if not fact_tokens:
            continue
        overlap = len(fact_tokens & out_tokens) / max(len(fact_tokens), 1)
        # If the output strongly references the fact's subject but introduces a
        # negation absent from the fact, flag a potential contradiction.
        if overlap >= 0.5:
            out_has_neg = bool(out_tokens & negations)
            fact_has_neg = bool(fact_tokens & negations)
            if out_has_neg != fact_has_neg:
                contradictions += 1
    score = consistency_from_contradictions(len(facts), contradictions)
    return score, len(facts), contradictions


def score_consistency(
    *,
    output: str,
    ledger: FactLedger,
    model: LLMClient | None = None,
    deterministic: bool = True,
) -> float:
    """Score consistency in [0, 1], preferring deterministic facts then judge."""
    if not ledger.all_facts():
        return 1.0
    if deterministic:
        score, total, contradictions = deterministic_consistency(output, ledger)
        # If deterministic check found a clear signal, trust it.
        if total > 0 and contradictions > 0:
            return score
    if model is not None:
        facts = "\n".join(f"- {f}" for f in ledger.all_facts())
        messages = [
            {"role": "system", "content": _CONSISTENCY_SYSTEM},
            {"role": "user", "content": f"FACTS:\n{facts}\n\nAGENT OUTPUT:\n{output}"},
        ]
        response = model.generate(messages, tools=None, metadata={"json_output": True})
        if not response.model_unavailable_reason:
            parsed = extract_json_object(response.text)
            if parsed and "score" in parsed:
                try:
                    return max(0.0, min(1.0, float(parsed["score"])))
                except (TypeError, ValueError):
                    _log.warning("Consistency judge returned non-numeric score.")
    # Fall back to the deterministic score (defaults to 1.0 when no contradictions).
    score, _, _ = deterministic_consistency(output, ledger)
    return score
