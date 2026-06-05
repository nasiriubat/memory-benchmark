"""Metric calculation tests."""

from __future__ import annotations

import math

from benchmark.metrics import (
    improvement_over_time,
    memory_utility_gain,
    ols_slope,
    progress_score,
    reliability,
    task_success_rate,
)


def test_task_success_rate():
    assert task_success_rate(3, 4) == 0.75
    assert task_success_rate(0, 0) == 0.0


def test_progress_score():
    assert progress_score(2, 5) == 0.4
    assert progress_score(0, 0) == 0.0
    assert progress_score(5, 5) == 1.0


def test_ols_slope_perfect_increase():
    # y = x -> slope 1.0
    assert math.isclose(ols_slope([0, 1, 2, 3, 4]), 1.0, rel_tol=1e-9)


def test_ols_slope_flat():
    assert ols_slope([0.5, 0.5, 0.5]) == 0.0


def test_improvement_over_time():
    res = improvement_over_time([0.0, 0.0, 1.0, 1.0])
    assert res.slope > 0
    # early mean = 0.0, late mean = 1.0 -> relative improvement large/positive.
    assert res.relative_improvement > 0


def test_memory_utility_gain():
    assert math.isclose(memory_utility_gain(0.8, 0.5), 0.3, rel_tol=1e-9)
    assert math.isclose(memory_utility_gain(0.4, 0.6), -0.2, rel_tol=1e-9)


def test_reliability_pass_at_k_and_cov():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]
    successes = [True, False, True, True, False]
    res = reliability(scores, successes)
    assert res.pass_at_1 == 1.0  # first run succeeded
    assert res.pass_at_5 == 1.0  # at least one of five succeeded
    assert math.isclose(res.mean_score, 0.6, rel_tol=1e-9)
    assert res.std_score > 0
    assert res.coefficient_of_variation > 0


def test_pass_at_1_fail_first():
    res = reliability([0.0, 1.0], [False, True])
    assert res.pass_at_1 == 0.0
    assert res.pass_at_5 == 1.0
