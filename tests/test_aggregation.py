"""Aggregation tests using fake sample data, incl. gain-vs-stateless."""

from __future__ import annotations

import pandas as pd

from benchmark import analysis


def _fake_df() -> pd.DataFrame:
    rows = []
    # Matched cells across two memory systems for one model/benchmark/domain/task.
    for seed in (1, 2):
        rows.append(dict(
            run_id="r", benchmark="memoryarena", domain="bundled_shopping",
            memory_system="stateless", model="m1", task_id="t1", seed=seed,
            success=False, score=0.4, progress_score=0.5, consistency_score=1.0,
            estimated_cost_usd=0.0, latency_ms=10.0, total_tokens=100,
            failure_category="failed_retrieval",
        ))
        rows.append(dict(
            run_id="r", benchmark="memoryarena", domain="bundled_shopping",
            memory_system="mem0", model="m1", task_id="t1", seed=seed,
            success=True, score=0.9, progress_score=1.0, consistency_score=1.0,
            estimated_cost_usd=0.01, latency_ms=20.0, total_tokens=200,
            failure_category="",
        ))
    return pd.DataFrame(rows)


def test_compute_memory_utility_gain_vs_stateless():
    df = analysis.compute_memory_utility_gain(_fake_df())
    mem0 = df[df["memory_system"] == "mem0"]
    # gain = 0.9 - 0.4 = 0.5 for each matched cell.
    assert all(abs(v - 0.5) < 1e-9 for v in mem0["memory_utility_gain_vs_stateless"])
    stateless = df[df["memory_system"] == "stateless"]
    assert all(v == 0.0 for v in stateless["memory_utility_gain_vs_stateless"])


def test_leaderboard_ranks_better_system_first():
    df = analysis.compute_memory_utility_gain(_fake_df())
    board = analysis.leaderboard(df)
    assert board.iloc[0]["memory_system"] == "mem0"
    assert board.iloc[0]["rank"] == 1


def test_summary_and_views_build():
    df = analysis.compute_memory_utility_gain(_fake_df())
    assert not analysis.summary_by_memory_model(df).empty
    assert not analysis.summary_by_benchmark(df).empty
    views = analysis.aggregation_views(df)
    for key in [
        "memory_system_x_model", "memory_system_x_benchmark", "iteration_trend",
        "cost_vs_score", "memory_utility_vs_success", "consistency_vs_success",
    ]:
        assert key in views


def test_failure_categories_counts():
    df = _fake_df()
    fails = analysis.failure_categories(df)
    assert (fails["failure_category"] == "failed_retrieval").any()


def test_pairwise_vs_stateless_runs():
    df = analysis.compute_memory_utility_gain(_fake_df())
    pw = analysis.pairwise_vs_stateless(df, "score")
    assert not pw.empty
    assert (pw["memory_system"] == "mem0").any()
    row = pw[pw["memory_system"] == "mem0"].iloc[0]
    assert row["mean_diff"] > 0
