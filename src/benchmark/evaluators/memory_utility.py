"""Memory utility evaluators (paper RQ4).

Computes:
  * retrieval_relevance_score - deterministic gold-fact overlap, else LLM judge.
  * retrieval_correctness_score - whether retrieved memory conflicts with known
    task facts / prior accepted constraints.
  * memory_usage_score - whether the final answer actually used retrieved memory
    (prefers ``used_memory_ids``, falls back to LLM judge / overlap).

``memory_utility_gain_vs_stateless`` is computed in :mod:`benchmark.metrics`
since it requires the matched stateless baseline score.
"""

from __future__ import annotations

from typing import Any

from benchmark.evaluators.consistency import FactLedger, deterministic_consistency
from benchmark.llms.base import LLMClient
from benchmark.logging import get_logger
from benchmark.schemas import MemoryRecord
from benchmark.utils import extract_json_object, token_set

_log = get_logger("benchmark.eval.memory_utility")


def retrieval_relevance_score(
    *,
    query: str,
    retrieved: list[MemoryRecord],
    gold_memory_facts: list[str] | None = None,
    model: LLMClient | None = None,
) -> float:
    """Relevance of retrieved memory to the query / gold facts."""
    if not retrieved:
        return 0.0
    if gold_memory_facts:
        gold_tokens: set[str] = set()
        for fact in gold_memory_facts:
            gold_tokens |= token_set(fact)
        if gold_tokens:
            scores = []
            for rec in retrieved:
                rec_tokens = token_set(rec.text)
                if not rec_tokens:
                    scores.append(0.0)
                    continue
                scores.append(len(rec_tokens & gold_tokens) / len(rec_tokens | gold_tokens or {1}))
            return sum(scores) / len(scores)
    # Deterministic fallback: overlap with the query.
    query_tokens = token_set(query)
    if query_tokens:
        scores = []
        for rec in retrieved:
            rec_tokens = token_set(rec.text)
            union = len(query_tokens | rec_tokens) or 1
            scores.append(len(query_tokens & rec_tokens) / union)
        det = sum(scores) / len(scores)
        if det > 0 or model is None:
            return det
    if model is not None:
        return _judge_relevance(model, query, retrieved)
    return 0.0


def retrieval_correctness_score(
    *,
    retrieved: list[MemoryRecord],
    ledger: FactLedger | None = None,
) -> float:
    """1.0 if retrieved memory does not conflict with known facts/constraints."""
    if not retrieved:
        return 0.0
    if ledger is None or not ledger.all_facts():
        # No known facts to conflict with; treat retrieval as non-conflicting.
        return 1.0
    conflicts = 0
    for rec in retrieved:
        score, total, contradictions = deterministic_consistency(rec.text, ledger)
        if total > 0 and contradictions > 0:
            conflicts += 1
    return max(0.0, 1.0 - conflicts / len(retrieved))


def memory_usage_score(
    *,
    final_answer: str,
    used_memory_ids: list[str],
    retrieved: list[MemoryRecord],
    model: LLMClient | None = None,
) -> float:
    """Whether the final answer actually used retrieved memory."""
    if not retrieved:
        return 0.0
    # Prefer the explicit machine-readable signal.
    if used_memory_ids:
        retrieved_ids = {r.id for r in retrieved}
        valid_used = [mid for mid in used_memory_ids if mid in retrieved_ids]
        return len(valid_used) / len(retrieved) if retrieved else 0.0
    # Fallback: content overlap between answer and retrieved memory.
    answer_tokens = token_set(final_answer)
    if answer_tokens:
        used = 0
        for rec in retrieved:
            rec_tokens = token_set(rec.text)
            if rec_tokens and len(rec_tokens & answer_tokens) / len(rec_tokens) >= 0.3:
                used += 1
        overlap_score = used / len(retrieved)
        if overlap_score > 0 or model is None:
            return overlap_score
    if model is not None:
        return _judge_usage(model, final_answer, retrieved)
    return 0.0


# ------------------------------------------------------------------- judges
def _judge_relevance(model: LLMClient, query: str, retrieved: list[MemoryRecord]) -> float:
    mem = "\n".join(f"- {r.text}" for r in retrieved)
    messages = [
        {
            "role": "system",
            "content": 'Rate how relevant the MEMORY is to the QUERY. Respond ONLY '
            'with JSON: {"score": 0.0-1.0, "reason": "short"}.',
        },
        {"role": "user", "content": f"QUERY:\n{query}\n\nMEMORY:\n{mem}"},
    ]
    return _judge_score(model, messages)


def _judge_usage(model: LLMClient, final_answer: str, retrieved: list[MemoryRecord]) -> float:
    mem = "\n".join(f"- {r.text}" for r in retrieved)
    messages = [
        {
            "role": "system",
            "content": 'Did the ANSWER actually use the MEMORY? Respond ONLY with '
            'JSON: {"score": 0.0-1.0, "reason": "short"}.',
        },
        {"role": "user", "content": f"MEMORY:\n{mem}\n\nANSWER:\n{final_answer}"},
    ]
    return _judge_score(model, messages)


def _judge_score(model: LLMClient, messages: list[dict[str, Any]]) -> float:
    response = model.generate(messages, tools=None, metadata={"json_output": True})
    if response.model_unavailable_reason:
        return 0.0
    parsed = extract_json_object(response.text)
    if not parsed or "score" not in parsed:
        return 0.0
    try:
        return max(0.0, min(1.0, float(parsed["score"])))
    except (TypeError, ValueError):
        return 0.0
