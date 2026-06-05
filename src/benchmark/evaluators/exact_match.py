"""Deterministic exact / fuzzy match evaluator."""

from __future__ import annotations

from benchmark.utils import normalize_text, token_set


def exact_match(prediction: str, expected: str) -> bool:
    """Normalized exact match."""
    return normalize_text(prediction) == normalize_text(expected)


def contains_match(prediction: str, expected: str) -> bool:
    """True if the normalized expected answer appears in the prediction."""
    exp = normalize_text(expected)
    return bool(exp) and exp in normalize_text(prediction)


def token_f1(prediction: str, expected: str) -> float:
    """Token-level F1 between prediction and expected."""
    pred = token_set(prediction)
    gold = token_set(expected)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    overlap = len(pred & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, expected: str | None, *, f1_threshold: float = 0.6) -> float:
    """Best-effort deterministic score in [0, 1].

    Returns 1.0 on exact/contains match, otherwise token-F1 (counted as correct
    by callers when it exceeds ``f1_threshold``).
    """
    if expected is None:
        return 0.0
    if exact_match(prediction, expected) or contains_match(prediction, expected):
        return 1.0
    return token_f1(prediction, expected)
