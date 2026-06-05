"""Core metric calculations.

Implements the metrics from the paper (Section III-H) plus STATE-Bench and
MemoryArena headline metrics. Functions here are pure and deterministic so they
can be unit-tested directly. Custom metrics are kept separate from official
benchmark metrics (which are namespaced ``official_*``) by the callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Task success rate
# ---------------------------------------------------------------------------
def task_success_rate(successful_tasks: int, total_tasks: int) -> float:
    """successful_tasks / total_tasks (0.0 if no tasks)."""
    if total_tasks <= 0:
        return 0.0
    return successful_tasks / total_tasks


# ---------------------------------------------------------------------------
# 2. Rule compliance
# ---------------------------------------------------------------------------
def rule_compliance(rules_satisfied: int, total_rules: int) -> float:
    """rules_satisfied / total_rules (1.0 if there are no rules)."""
    if total_rules <= 0:
        return 1.0
    return rules_satisfied / total_rules


# ---------------------------------------------------------------------------
# 3. Progress score
# ---------------------------------------------------------------------------
def progress_score(correct_subtasks: int, total_subtasks: int) -> float:
    """correct_subtasks / total_subtasks (0.0 if no subtasks)."""
    if total_subtasks <= 0:
        return 0.0
    return correct_subtasks / total_subtasks


# ---------------------------------------------------------------------------
# 4. Improvement over time
# ---------------------------------------------------------------------------
@dataclass
class ImprovementResult:
    slope: float
    relative_improvement: float


def ols_slope(scores: list[float]) -> float:
    """Ordinary-least-squares slope of score vs. iteration index."""
    n = len(scores)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(scores) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, scores))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def improvement_over_time(scores: list[float], early_frac: float = 0.5) -> ImprovementResult:
    """Compute OLS slope and relative improvement (late vs early means)."""
    if not scores:
        return ImprovementResult(slope=0.0, relative_improvement=0.0)
    slope = ols_slope(scores)
    n = len(scores)
    split = max(1, int(n * early_frac))
    early = scores[:split]
    late = scores[split:] or scores[-1:]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    rel = (late_mean - early_mean) / max(abs(early_mean), 1e-9)
    return ImprovementResult(slope=slope, relative_improvement=rel)


# ---------------------------------------------------------------------------
# 8. Reliability (repeated runs)
# ---------------------------------------------------------------------------
@dataclass
class ReliabilityResult:
    pass_at_1: float
    pass_at_5: float
    mean_score: float
    std_score: float
    coefficient_of_variation: float


def pass_at_k(successes: list[bool], k: int) -> float:
    """Fraction of k-sized prefixes (here: whether any of first k runs passed).

    For repeated runs we report pass@k as: 1.0 if at least one of the first k
    runs succeeded, else 0.0 (the standard "any success in k attempts" notion).
    """
    if not successes:
        return 0.0
    window = successes[:k] if k > 0 else successes
    return 1.0 if any(window) else 0.0


def reliability(scores: list[float], successes: list[bool]) -> ReliabilityResult:
    """Compute pass@1, pass@5, mean, std, and coefficient of variation."""
    n = len(scores)
    if n == 0:
        return ReliabilityResult(0.0, 0.0, 0.0, 0.0, 0.0)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(var)
    cov = std / mean if mean not in (0.0, -0.0) else 0.0
    return ReliabilityResult(
        pass_at_1=pass_at_k(successes, 1),
        pass_at_5=pass_at_k(successes, 5),
        mean_score=mean,
        std_score=std,
        coefficient_of_variation=cov,
    )


# ---------------------------------------------------------------------------
# 7. Memory utility gain vs stateless
# ---------------------------------------------------------------------------
def memory_utility_gain(memory_config_score: float, stateless_score: float) -> float:
    """memory_config_score - stateless_score_for_same_model_benchmark_domain_task."""
    return memory_config_score - stateless_score


# ---------------------------------------------------------------------------
# Consistency helper (deterministic ledger overlap)
# ---------------------------------------------------------------------------
def consistency_from_contradictions(total_checks: int, contradictions: int) -> float:
    """1 - contradictions/total_checks, clamped to [0, 1]."""
    if total_checks <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - contradictions / total_checks))
