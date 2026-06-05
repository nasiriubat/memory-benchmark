"""LLM-judge evaluator with a strict JSON rubric.

Used only as a fallback when official scores and deterministic checks are not
available/sufficient. The judge is instructed to return strict JSON
``{"score": 0.0-1.0, "reason": "short"}``. If the judge LLM is unavailable
(offline mock), the judge returns a neutral score and flags it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmark.llms.base import LLMClient
from benchmark.logging import get_logger
from benchmark.utils import extract_json_object

_log = get_logger("benchmark.eval.judge")


@dataclass
class JudgeResult:
    score: float
    reason: str
    available: bool = True


_JUDGE_SYSTEM = (
    "You are a strict evaluation judge. Compare the agent answer to the "
    "expected answer and rubric. Respond ONLY with JSON: "
    '{"score": <float 0.0-1.0>, "reason": "<short>"}.'
)


def judge(
    *,
    model: LLMClient,
    question: str,
    prediction: str,
    expected: str | None = None,
    rubric: str = "",
) -> JudgeResult:
    """Run the LLM judge. Returns a neutral result if the model is offline."""
    user = (
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED:\n{expected or '(none provided)'}\n\n"
        f"RUBRIC:\n{rubric or 'Score correctness and helpfulness.'}\n\n"
        f"AGENT ANSWER:\n{prediction}"
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
    response = model.generate(messages, tools=None, metadata={"json_output": True})
    if response.model_unavailable_reason:
        # Offline mock or unavailable model: do not fabricate a confident score.
        return JudgeResult(score=0.0, reason="judge_model_unavailable", available=False)

    parsed: dict[str, Any] | None = extract_json_object(response.text)
    if not parsed or "score" not in parsed:
        _log.warning("LLM judge returned non-JSON or missing score; defaulting to 0.0.")
        return JudgeResult(score=0.0, reason="unparseable_judge_output", available=True)
    try:
        score = float(parsed["score"])
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    return JudgeResult(score=score, reason=str(parsed.get("reason", ""))[:300], available=True)
