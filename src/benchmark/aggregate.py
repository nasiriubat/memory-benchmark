"""Aggregation pipeline: read all results and emit aggregated outputs.

Produces:
    results/aggregated/all_results.parquet
    results/aggregated/summary_by_memory_model.csv
    results/aggregated/summary_by_benchmark.csv
    results/aggregated/leaderboard.csv
    results/aggregated/failure_categories.csv
plus the eight aggregation views and the statistical-analysis CSVs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark import analysis
from benchmark.config import load_config
from benchmark.logging import get_logger
from benchmark.utils import ensure_dir

_log = get_logger("benchmark.aggregate")


def aggregate(results_root: Path) -> pd.DataFrame:
    """Run the full aggregation pipeline. Returns the combined DataFrame."""
    out_dir = ensure_dir(results_root / "aggregated")
    df = analysis.load_all_results(results_root)
    if df.empty:
        _log.warning(
            "No results found under %s/by_memory_system. Run a memory system first.",
            results_root,
        )
        return df

    df = analysis.compute_memory_utility_gain(df)

    # Core aggregated outputs.
    df.to_parquet(out_dir / "all_results.parquet", index=False)
    analysis.summary_by_memory_model(df).to_csv(
        out_dir / "summary_by_memory_model.csv", index=False
    )
    analysis.summary_by_benchmark(df).to_csv(
        out_dir / "summary_by_benchmark.csv", index=False
    )
    analysis.leaderboard(df).to_csv(out_dir / "leaderboard.csv", index=False)
    analysis.failure_categories(df).to_csv(out_dir / "failure_categories.csv", index=False)

    # Paper-ready per-system summary (Section III-H metrics) + RQ2 learning curve.
    analysis.paper_summary(df).to_csv(out_dir / "paper_table.csv", index=False)
    analysis.subtask_trend(results_root).to_csv(out_dir / "subtask_trend.csv", index=False)

    # The eight aggregation views.
    views_dir = ensure_dir(out_dir / "views")
    for name, view in analysis.aggregation_views(df).items():
        view.to_csv(views_dir / f"{name}.csv", index=False)

    # Statistical analysis (paper Section III-J).
    _write_statistics(df, out_dir)

    _log.info("Aggregation complete -> %s", out_dir)
    return df


def _write_statistics(df: pd.DataFrame, out_dir: Path) -> None:
    cfg = load_config()
    stats_cfg = cfg.benchmark.get("statistics", {}) or {}
    if not stats_cfg.get("enabled", True):
        return
    factors = stats_cfg.get("anova_factors", ["memory_system", "model"])
    metrics = stats_cfg.get("anova_metrics", ["success", "score", "progress_score"])

    desc = analysis.descriptive_stats(df, metrics)
    if not desc.empty:
        desc.to_csv(out_dir / "descriptive_stats.csv", index=False)

    anova_frames = [analysis.two_way_anova(df, m, factors) for m in metrics]
    anova_frames = [f for f in anova_frames if not f.empty]
    if anova_frames:
        pd.concat(anova_frames, ignore_index=True).to_csv(
            out_dir / "stats_anova.csv", index=False
        )

    tukey_frames = [analysis.tukey_hsd(df, m) for m in metrics]
    tukey_frames = [f for f in tukey_frames if not f.empty]
    if tukey_frames:
        pd.concat(tukey_frames, ignore_index=True).to_csv(
            out_dir / "stats_posthoc.csv", index=False
        )

    pairwise_frames = [analysis.pairwise_vs_stateless(df, m) for m in metrics]
    pairwise_frames = [f for f in pairwise_frames if not f.empty]
    if pairwise_frames:
        combined = pd.concat(pairwise_frames, ignore_index=True)
        combined.to_csv(out_dir / "pairwise_vs_stateless.csv", index=False)
        # Effect sizes are embedded (cohens_d) in the pairwise comparison.
        combined[["memory_system", "metric", "cohens_d"]].to_csv(
            out_dir / "effect_sizes.csv", index=False
        )
