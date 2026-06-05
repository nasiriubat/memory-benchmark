#!/usr/bin/env python
"""Run the statistical analyses (paper Section III-J) on aggregated results.

Reads results/aggregated/all_results.parquet (or rebuilds from per-system
folders) and writes ANOVA, Tukey HSD, effect-size, and pairwise-vs-stateless
CSVs into results/aggregated/.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd

from benchmark import analysis
from benchmark.aggregate import _write_statistics
from benchmark.logging import get_logger
from benchmark.utils import ensure_dir, results_root

_log = get_logger("scripts.statistical_analysis")


def main() -> None:
    root = results_root()
    out_dir = ensure_dir(root / "aggregated")
    parquet = out_dir / "all_results.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        df = analysis.compute_memory_utility_gain(analysis.load_all_results(root))
    if df.empty:
        _log.warning("No results to analyze. Run benchmarks and aggregate first.")
        return
    _write_statistics(df, out_dir)
    _log.info("Statistical analysis written to %s", out_dir)


if __name__ == "__main__":
    main()
