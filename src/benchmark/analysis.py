"""Aggregation + statistical analysis.

Reads all per-memory-system task results, computes the gain-vs-stateless metric,
and produces the aggregated tables, the eight aggregation views, and the
paper's statistical analyses (Section III-J): descriptive stats, two-way ANOVA
(memory_system x model, main + interaction), Tukey HSD post-hoc, effect sizes,
and pairwise comparisons vs the stateless baseline.

All functions operate on a pandas DataFrame of flattened :class:`TaskResult`
rows and never mutate global state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.logging import get_logger
from benchmark.metrics import reliability

_log = get_logger("benchmark.analysis")

# Key axes that define a "matched" comparison cell for gain-vs-stateless.
_MATCH_KEYS = ["model", "benchmark", "domain", "task_id", "seed"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all_results(results_root: Path) -> pd.DataFrame:
    """Load every task result row from the per-memory-system folders."""
    rows: list[dict[str, Any]] = []
    by_system = results_root / "by_memory_system"
    if not by_system.exists():
        return pd.DataFrame()
    for jsonl in by_system.glob("*/*/results.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                _log.warning("Skipping malformed result line in %s", jsonl)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([_flatten_official(r) for r in rows])
    return _enrich_model_columns(df)


def _enrich_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep model slot for matching; add human-readable OpenRouter model ids."""
    if df.empty or "model" not in df.columns:
        return df
    df = df.copy()
    df["model_slot"] = df["model"]
    resolved = df["resolved_model"] if "resolved_model" in df.columns else pd.Series(dtype=str)
    df["model_display"] = resolved.fillna(df["model_slot"])
    if "requested_model" in df.columns:
        df["model_alias"] = df["requested_model"]
    return df


def _flatten_official(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    official = row.pop("official_metrics", {}) or {}
    for key, value in official.items():
        row[f"official_{key}"] = value
    # Drop nested structures that don't belong in a flat table.
    for nested in ("openrouter_model_pricing", "openrouter_provider_policy",
                   "openrouter_supported_parameters"):
        if isinstance(row.get(nested), (dict, list)):
            row[nested] = json.dumps(row[nested])
    return row


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------
def compute_memory_utility_gain(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ``memory_utility_gain_vs_stateless`` by matching the stateless cell."""
    if df.empty or "memory_system" not in df.columns:
        return df
    df = df.copy()
    stateless = df[df["memory_system"] == "stateless"]
    if stateless.empty:
        _log.warning(
            "No stateless baseline found; memory_utility_gain_vs_stateless left as-is. "
            "Run the stateless system first."
        )
        return df
    baseline = (
        stateless.groupby(_MATCH_KEYS)["score"].mean().rename("_stateless_score").reset_index()
    )
    df = df.merge(baseline, on=_MATCH_KEYS, how="left")
    mask = df["memory_system"] != "stateless"
    df.loc[mask, "memory_utility_gain_vs_stateless"] = (
        df.loc[mask, "score"] - df.loc[mask, "_stateless_score"]
    )
    df.loc[df["memory_system"] == "stateless", "memory_utility_gain_vs_stateless"] = 0.0
    df = df.drop(columns=["_stateless_score"])
    return df


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def summary_by_memory_model(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["memory_system", "model_display"] if "model_display" in df.columns else ["memory_system", "model"]
    grp = df.groupby(keys)
    out = _summarize(grp).reset_index()
    if "model_display" in out.columns:
        out = out.rename(columns={"model_display": "model"})
    return out


def summary_by_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grp = df.groupby(["memory_system", "benchmark"])
    return _summarize(grp).reset_index()


def _pass_at_5_for_group(g: pd.DataFrame) -> float:
    """Mean pass@5 across tasks (multiple seeds per task_id)."""
    if g.empty or "task_id" not in g.columns:
        return float("nan")
    passes: list[float] = []
    for _, tg in g.groupby("task_id"):
        succ = tg["success"].astype(bool).tolist()
        scores = pd.to_numeric(tg["score"], errors="coerce").fillna(0.0).tolist()
        if succ:
            passes.append(reliability(scores, succ).pass_at_5)
    return float(np.mean(passes)) if passes else float("nan")


def _summarize(grp: Any) -> pd.DataFrame:
    def _one(g: pd.DataFrame) -> pd.Series:
        out: dict[str, Any] = {
            "n": len(g),
            "task_success_rate": _safe_mean(g["success"].astype(float)),
            "pass_at_5": _pass_at_5_for_group(g),
            "mean_score": _safe_mean(g["score"]),
            "progress_score": _safe_mean(g["progress_score"]),
            "consistency_score": _safe_mean(g["consistency_score"]),
            "memory_utility_gain_vs_stateless": _safe_mean(
                g.get("memory_utility_gain_vs_stateless", pd.Series(dtype=float))
            ),
            "mean_cost_usd": _safe_mean(g["estimated_cost_usd"]),
            "mean_latency_ms": _safe_mean(g["latency_ms"]),
            "total_tokens": _safe_mean(g["total_tokens"]),
        }
        if "model_display" in g.columns:
            out["models_used"] = ", ".join(sorted(g["model_display"].dropna().unique()))
        return pd.Series(out)

    return grp.apply(_one, include_groups=False)


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Rank memory systems by overall task success then mean score."""
    if df.empty:
        return df
    grp = df.groupby("memory_system")
    board = _summarize(grp).reset_index()
    board = board.sort_values(
        by=["task_success_rate", "mean_score"], ascending=False
    ).reset_index(drop=True)
    board.insert(0, "rank", board.index + 1)
    return board


def failure_categories(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "failure_category" not in df.columns:
        return pd.DataFrame(columns=["memory_system", "failure_category", "count"])
    fails = df[df["failure_category"].notna() & (df["failure_category"] != "")]
    if fails.empty:
        return pd.DataFrame(columns=["memory_system", "failure_category", "count"])
    return (
        fails.groupby(["memory_system", "failure_category"]).size().rename("count").reset_index()
    )


# ---------------------------------------------------------------------------
# The eight aggregation views
# ---------------------------------------------------------------------------
def aggregation_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    views: dict[str, pd.DataFrame] = {}
    views["memory_system_x_model"] = summary_by_memory_model(df)
    views["memory_system_x_benchmark"] = summary_by_benchmark(df)
    views["memory_system_x_domain"] = _summarize(
        df.groupby(["memory_system", "domain"])
    ).reset_index()
    model_key = "model_display" if "model_display" in df.columns else "model"
    views["model_x_benchmark"] = _summarize(
        df.groupby([model_key, "benchmark"])
    ).reset_index()
    if model_key == "model_display":
        views["model_x_benchmark"] = views["model_x_benchmark"].rename(
            columns={"model_display": "model"}
        )
    # Iteration trend: score by seed (iteration proxy).
    views["iteration_trend"] = (
        df.groupby(["memory_system", "seed"])["score"].mean().reset_index()
    )
    # Cost vs score (per memory_system x model cell).
    cvs = df.groupby(["memory_system", "model"]).agg(
        mean_cost_usd=("estimated_cost_usd", "mean"),
        mean_score=("score", "mean"),
    ).reset_index()
    views["cost_vs_score"] = cvs
    # Memory utility vs task success.
    muts = df.groupby("memory_system").agg(
        memory_utility_gain_vs_stateless=("memory_utility_gain_vs_stateless", "mean"),
        task_success_rate=("success", "mean"),
    ).reset_index()
    views["memory_utility_vs_success"] = muts
    # Consistency vs task success.
    cvss = df.groupby("memory_system").agg(
        consistency_score=("consistency_score", "mean"),
        task_success_rate=("success", "mean"),
    ).reset_index()
    views["consistency_vs_success"] = cvss
    return views


# ---------------------------------------------------------------------------
# Statistical analysis (paper Section III-J)
# ---------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    present = [m for m in metrics if m in df.columns]
    if not present:
        return pd.DataFrame()
    grp = df.groupby(["memory_system", "model"])[present]
    return grp.agg(["mean", "std", "count"]).reset_index()


def two_way_anova(
    df: pd.DataFrame, metric: str, factors: list[str]
) -> pd.DataFrame:
    """Two-way ANOVA for ``metric`` across two factors (main + interaction)."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        _log.warning("statsmodels not installed; skipping ANOVA.")
        return pd.DataFrame()

    work = df[[*factors, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce").astype(float)
    work = work.dropna()
    # Need at least two levels per factor and enough rows.
    if work.empty or any(work[f].nunique() < 2 for f in factors):
        _log.warning("ANOVA for %s skipped (insufficient factor levels/data).", metric)
        return pd.DataFrame()
    # Degenerate near-zero variance (e.g. all-equal endog) makes ANOVA undefined.
    if work[metric].std(ddof=0) < 1e-12:
        _log.warning("ANOVA for %s skipped (near-zero variance).", metric)
        return pd.DataFrame()
    f1, f2 = factors[0], factors[1]
    formula = f"Q('{metric}') ~ C(Q('{f1}')) * C(Q('{f2}'))"
    try:
        model = ols(formula, data=work).fit()
        table = sm.stats.anova_lm(model, typ=2)
    except Exception as exc:  # noqa: BLE001
        _log.warning("ANOVA failed for %s: %s", metric, exc)
        return pd.DataFrame()
    table = table.reset_index().rename(columns={"index": "term"})
    table.insert(0, "metric", metric)
    # Effect size: eta-squared = SS_term / SS_total.
    ss_total = table["sum_sq"].sum()
    table["eta_squared"] = table["sum_sq"] / ss_total if ss_total else np.nan
    return table


def tukey_hsd(df: pd.DataFrame, metric: str, group_col: str = "memory_system") -> pd.DataFrame:
    """Tukey HSD post-hoc across ``group_col`` for ``metric``."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError:
        _log.warning("statsmodels not installed; skipping Tukey HSD.")
        return pd.DataFrame()
    work = df[[group_col, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce").astype(float)
    work = work.dropna()
    if work[group_col].nunique() < 2 or len(work) < 3:
        return pd.DataFrame()
    if work[metric].std(ddof=0) < 1e-12:
        _log.warning("Tukey HSD for %s skipped (near-zero variance).", metric)
        return pd.DataFrame()
    try:
        res = pairwise_tukeyhsd(work[metric], work[group_col])
    except Exception as exc:  # noqa: BLE001
        _log.warning("Tukey HSD failed for %s: %s", metric, exc)
        return pd.DataFrame()
    out = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
    out.insert(0, "metric", metric)
    return out


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d effect size between two samples."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def pairwise_vs_stateless(df: pd.DataFrame, metric: str = "score") -> pd.DataFrame:
    """Compare each memory system to the stateless baseline (mean diff, t-test, d)."""
    if df.empty or "memory_system" not in df.columns or metric not in df.columns:
        return pd.DataFrame()
    from scipy import stats as scipy_stats

    base = pd.to_numeric(
        df.loc[df["memory_system"] == "stateless", metric], errors="coerce"
    ).dropna()
    rows: list[dict[str, Any]] = []
    for system in sorted(df["memory_system"].unique()):
        if system == "stateless":
            continue
        sample = pd.to_numeric(
            df.loc[df["memory_system"] == system, metric], errors="coerce"
        ).dropna()
        if len(sample) < 2 or len(base) < 2:
            continue
        # Avoid undefined t-test when both groups have zero variance.
        if sample.std(ddof=1) < 1e-12 and base.std(ddof=1) < 1e-12:
            t_stat, p_val = float("nan"), float("nan")
        else:
            t_stat, p_val = scipy_stats.ttest_ind(sample, base, equal_var=False)
        rows.append(
            {
                "memory_system": system,
                "metric": metric,
                "mean": float(sample.mean()),
                "stateless_mean": float(base.mean()),
                "mean_diff": float(sample.mean() - base.mean()),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": cohens_d(sample, base),
            }
        )
    return pd.DataFrame(rows)
