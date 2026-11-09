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
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.logging import get_logger
from benchmark.metrics import pass_at_k

_log = get_logger("benchmark.analysis")

# Key axes that define a "matched" comparison cell for gain-vs-stateless.
_MATCH_KEYS = ["model", "benchmark", "domain", "task_id", "seed"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_text_resilient(path: Path, attempts: int = 25) -> str:
    """Read a file, retrying on the transient OSError (EAGAIN/EDEADLK, "Resource
    deadlock avoided") that cloud-synced filesystems (OneDrive) raise while their
    sync process holds the file. Returns "" if it never succeeds (logged)."""
    for attempt in range(attempts):
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            if attempt == attempts - 1:
                _log.warning("could not read %s after %d attempts: %s", path, attempts, exc)
                return ""
            time.sleep(0.05 * (attempt + 1))
    return ""


def load_all_results(results_root: Path) -> pd.DataFrame:
    """Load every task result row from the per-memory-system folders."""
    rows: list[dict[str, Any]] = []
    by_system = results_root / "by_memory_system"
    if not by_system.exists():
        return pd.DataFrame()
    for jsonl in by_system.glob("*/*/results.jsonl"):
        for line in _read_text_resilient(jsonl).splitlines():
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
    out = _summarize(grp, _runs_per_task(df)).reset_index()
    if "model_display" in out.columns:
        out = out.rename(columns={"model_display": "model"})
    return out


def summary_by_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grp = df.groupby(["memory_system", "benchmark"])
    return _summarize(grp, _runs_per_task(df)).reset_index()


def _runs_per_task(df: pd.DataFrame) -> int:
    """Repeated runs per task (== num_runs). Drives the pass@k label so it never
    hard-codes a stale k (e.g. reporting pass@5 when only 3 runs were executed)."""
    if df.empty or "task_id" not in df.columns:
        return 0
    keys = [c for c in ("memory_system", "model", "task_id") if c in df.columns]
    if not keys:
        return 0
    counter = df.groupby(keys)["seed"].nunique() if "seed" in df.columns else df.groupby(keys).size()
    return int(counter.max()) if len(counter) else 0


def _pass_at_k_for_group(g: pd.DataFrame, k: int) -> float:
    """Mean pass@k across tasks (any success within the k repeated runs)."""
    if g.empty or "task_id" not in g.columns:
        return float("nan")
    passes: list[float] = []
    for _, tg in g.groupby("task_id"):
        succ = tg["success"].astype(bool).tolist()
        if succ:
            passes.append(pass_at_k(succ, k if k > 0 else len(succ)))
    return float(np.mean(passes)) if passes else float("nan")


def _summarize(grp: Any, k: int = 0) -> pd.DataFrame:
    # Name the reliability column after the actual number of repeated runs.
    pass_col = f"pass_at_{k}" if k > 0 else "pass_at_k"

    def _one(g: pd.DataFrame) -> pd.Series:
        out: dict[str, Any] = {
            "n": len(g),
            "task_success_rate": _safe_mean(g["success"].astype(float)),
            pass_col: _pass_at_k_for_group(g, k),
            "mean_score": _safe_mean(g["score"]),
            "progress_score": _safe_mean(g["progress_score"]),
            "consistency_score": _safe_mean(g["consistency_score"]),
            "memory_relevance_score": _safe_mean(
                g.get("memory_relevance_score", pd.Series(dtype=float))
            ),
            "memory_correctness_score": _safe_mean(
                g.get("memory_correctness_score", pd.Series(dtype=float))
            ),
            "memory_usage_score": _safe_mean(
                g.get("memory_usage_score", pd.Series(dtype=float))
            ),
            "memory_utility_gain_vs_stateless": _safe_mean(
                g.get("memory_utility_gain_vs_stateless", pd.Series(dtype=float))
            ),
            "mean_cost_usd": _safe_mean(
                g.get("total_cost_usd", g["estimated_cost_usd"])  # agent + judge
            ),
            "mean_latency_ms": _safe_mean(g["latency_ms"]),
            "total_tokens": _safe_mean(g["total_tokens"]),
        }
        if "model_display" in g.columns:
            out["models_used"] = ", ".join(sorted(g["model_display"].dropna().unique()))
        return pd.Series(out)

    return grp.apply(_one, include_groups=False)


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Rank memory systems by progress score then memory-utility gain.

    (Full-task success is all-or-nothing and ~0 on hard tasks, so it is a poor
    sort key; progress_score / gain are the paper's primary performance metrics.)
    """
    if df.empty:
        return df
    grp = df.groupby("memory_system")
    board = _summarize(grp, _runs_per_task(df)).reset_index()
    board = board.sort_values(
        by=["progress_score", "memory_utility_gain_vs_stateless"], ascending=False
    ).reset_index(drop=True)
    board.insert(0, "rank", board.index + 1)
    return board


def paper_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-memory-system summary aligned to the paper (Section III-H).

    One row per system with the primary performance metrics (progress score,
    gain vs stateless), memory-use quality (relevance/correctness/usage),
    learning (improvement slope), memory behaviour, and efficiency.
    """
    if df.empty:
        return df

    def _one(g: pd.DataFrame) -> pd.Series:
        def m(col: str) -> float:
            return _safe_mean(g[col]) if col in g.columns else float("nan")

        return pd.Series(
            {
                "n": len(g),
                "progress_score": m("progress_score"),
                "mean_score": m("score"),
                "gain_vs_stateless": m("memory_utility_gain_vs_stateless"),
                "consistency": m("consistency_score"),
                "relevance": m("memory_relevance_score"),
                "correctness": m("memory_correctness_score"),
                "usage": m("memory_usage_score"),
                "improvement_slope": m("official_memoryarena_improvement_slope"),
                "retrieved": m("retrieved_memory_count"),
                "written": m("written_memory_count"),
                "total_cost_usd": m("total_cost_usd")
                if "total_cost_usd" in g.columns
                else m("estimated_cost_usd"),
                "latency_ms": m("latency_ms"),
                "full_task_success": m("success"),
            }
        )

    out = df.groupby("memory_system").apply(_one, include_groups=False).reset_index()
    return out.sort_values("progress_score", ascending=False).reset_index(drop=True)


def subtask_trend(results_root: "Path") -> pd.DataFrame:
    """Mean subtask score by subtask position, per memory system (RQ2 learning
    curve). Reads ``evaluator_score`` events from the raw event logs the runner
    writes under ``results/raw/<run>/<system>/**/events.jsonl``."""
    import json
    import re

    raw = results_root / "raw"
    rows: list[tuple[str, int, float]] = []
    if raw.exists():
        for ev in raw.glob("**/events.jsonl"):
            parts = ev.relative_to(raw).parts  # (run_id, system, model, benchmark, domain, ...)
            if len(parts) < 2:
                continue
            system = parts[1]
            model = parts[2] if len(parts) > 2 else "unknown"  # model slot
            for line in _read_text_resilient(ev).splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if e.get("event_type") != "evaluator_score":
                    continue
                match = re.search(r"(\d+)", str(e.get("subtask_id") or ""))
                score = (e.get("payload") or {}).get("score")
                if not match or score is None:
                    continue
                rows.append((system, model, int(match.group(1)), float(score)))
    if not rows:
        return pd.DataFrame(columns=["memory_system", "model", "subtask_index", "mean_score", "n"])
    d = pd.DataFrame(rows, columns=["memory_system", "model", "subtask_index", "score"])
    return (
        d.groupby(["memory_system", "model", "subtask_index"])["score"]
        .agg(mean_score="mean", n="size")
        .reset_index()
    )


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
    k = _runs_per_task(df)
    views["memory_system_x_domain"] = _summarize(
        df.groupby(["memory_system", "domain"]), k
    ).reset_index()
    model_key = "model_display" if "model_display" in df.columns else "model"
    views["model_x_benchmark"] = _summarize(
        df.groupby([model_key, "benchmark"]), k
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
