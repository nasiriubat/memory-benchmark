"""Export the benchmark results to a single, reviewer-friendly Excel workbook.

Reads the aggregated CSVs in results/aggregated/ (produced by `make aggregate`)
plus the raw per-task rows, and writes one .xlsx with:

    Overview            - what the study is, sample, models, scoring
    Definitions         - plain-English meaning of every metric column
    Leaderboard         - systems ranked by progress score
    Per-system summary  - the paper's per-system metric table
    Learning curve      - mean score by subtask position (within task)
    Failure taxonomy    - failure reasons with human-readable labels
    Per-model summary   - metrics broken down by model
    All results (raw)   - every per-task row (curated columns)

Run inside the project image so pandas/openpyxl are available:

    docker compose -p memory-agent-benchmark run --rm --no-deps \
        --entrypoint python stateless scripts/export_excel.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "results" / "aggregated"
OUT = ROOT / "benchmark_results.xlsx"

# Clean display names + stable ordering for the LLM backbones / memory systems.
_MODEL_LABEL = {
    "gpt-5.5": "gpt-5.5",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
}
_MODEL_ORDER = ["gpt-5.5", "Gemini 3.5 Flash", "Claude Opus 4.8"]
_SYS_ORDER = ["stateless", "hindsight", "mem0", "acontext"]

# Columns that are always-null or clutter in the raw rows (mirrors the UI's
# _DROP_COLS) so the "All results" sheet stays readable.
DROP_COLS = {
    "rule_compliance", "ux_score", "error", "model_unavailable_reason",
    "openrouter_model_catalog_created_at", "openrouter_generation_id",
    "openrouter_request_id", "openrouter_supported_parameters",
    "openrouter_provider_policy", "openrouter_model_pricing",
    "practical_correctness", "usefulness", "dfx_constraint_adherence",
    "reasoning_clarity", "engineering_applicability",
    "model_slot", "cached_input_tokens", "memory_store_size_bytes",
}

FAILURE_LABELS = {
    "missing_memory_write": "No memory written",
    "irrelevant_memory_write": "Irrelevant memory written",
    "failed_retrieval": "Nothing retrieved",
    "incorrect_or_outdated_retrieval": "Wrong/outdated memory retrieved",
    "incorrect_use_of_retrieved_memory": "Retrieved but not used",
    "over_reliance_on_prior_feedback": "Over-relied on prior feedback",
    "inconsistency_across_iterations": "Inconsistent across runs",
    "model_reasoning_error": "Model reasoning error",
    "tool_use_error": "Tool-use error",
    "benchmark_environment_error": "Environment/backend error",
    "unknown": "Unknown",
}
FAILURE_EXPLAIN = {
    "missing_memory_write": "Task failed and the system stored nothing.",
    "irrelevant_memory_write": "Memories written were off-topic for the task.",
    "failed_retrieval": "Memories existed but retrieval returned nothing.",
    "incorrect_or_outdated_retrieval": "Retrieved memories were wrong or stale (correctness < 0.5).",
    "incorrect_use_of_retrieved_memory": (
        "Correct memories were retrieved but the agent barely used them (usage < 0.3), "
        "usually because what was retrieved wasn't relevant to the subtask."
    ),
    "over_reliance_on_prior_feedback": "Leaned too hard on earlier feedback.",
    "inconsistency_across_iterations": "Answers varied across repeated runs (consistency < 0.5).",
    "model_reasoning_error": "Memory worked, but the model reasoned to the wrong answer.",
    "tool_use_error": "The agent misused a tool.",
    "benchmark_environment_error": "An error/crash in the backend or harness.",
    "unknown": "Did not match any specific taxonomy branch.",
}

DEFINITIONS = [
    ("memory_system", "The memory mechanism under test; 'stateless' is the no-memory baseline."),
    ("model / models_used", "LLM backbone used by the agent (gpt-5.5)."),
    ("n", "Number of task rows (tasks x repeated runs) behind the aggregate."),
    ("progress_score", "PRIMARY. Fraction of a task's subtasks answered correctly (LLM-judged), averaged."),
    ("memory_utility_gain_vs_stateless / gain_vs_stateless",
     "PRIMARY. This system's score minus the matched stateless row (same task+seed). Positive = memory helped."),
    ("mean_score / score", "Mean of the per-subtask LLM-judge correctness scores (0-1)."),
    ("task_success_rate / full_task_success",
     "Full-task success: 1 only if EVERY subtask is correct (all-or-nothing). Near-zero on hard tasks."),
    ("pass_at_3", "Fraction of tasks solved in at least one of the 3 repeated runs (reliability)."),
    ("consistency / consistency_score", "1 - contradictions vs the task's fact ledger (staying consistent)."),
    ("relevance / memory_relevance_score", "How well retrieved memories match the query."),
    ("correctness / memory_correctness_score", "How factually correct the stored/retrieved memories are."),
    ("usage / memory_usage_score", "How much the agent actually grounded its answer in retrieved memory."),
    ("improvement_slope", "Trend of score across a task's subtasks (positive = improving within task)."),
    ("retrieved / retrieved_memory_count", "Avg memories retrieved per task."),
    ("written / written_memory_count", "Avg memories written per task."),
    ("total_cost_usd / mean_cost_usd", "Avg USD per task = agent LLM cost + judge LLM cost."),
    ("latency_ms / mean_latency_ms", "Avg wall-clock milliseconds per task."),
    ("total_tokens", "Avg agent input+output tokens per task."),
    ("subtask_index", "Position of a subtask within a task (0 = first)."),
    ("failure_category / reason", "Heuristic diagnostic label for a failed task (see Failure taxonomy sheet)."),
    ("F, PR(>F)", "ANOVA F-statistic and its p-value (PR(>F) < 0.05 = the factor significantly affects the metric)."),
    ("eta_squared", "ANOVA effect size: share of variance explained by that factor (0-1)."),
    ("t_stat, p_value", "Paired comparison vs stateless: t-statistic and p-value (p < 0.05 = significant improvement)."),
    ("cohens_d", "Standardised effect size vs stateless (~0.2 small, 0.5 medium, 0.8 large)."),
    ("meandiff, p-adj, reject", "Tukey HSD post-hoc: mean difference, adjusted p, and whether groups differ significantly."),
]


def _overview() -> pd.DataFrame:
    # Pull the models / systems actually present so the sheet never goes stale.
    mm = _csv("summary_by_memory_model.csv")
    models = ", ".join(sorted(mm["model"].astype(str).unique())) if not mm.empty else "gpt-5.5"
    systems = ", ".join(sorted(mm["memory_system"].astype(str).unique())) if not mm.empty else \
        "stateless, hindsight, mem0, acontext"
    n_models = mm["model"].nunique() if not mm.empty else 1
    rows = [
        ("Study", "Memory Matters - Evaluating Learning Behavior in LLM-Based Agents (ICSE TR)"),
        ("Benchmark", "MemoryArena"),
        ("Dataset", "HuggingFace ZexueHe/memoryarena, config: bundled_shopping"),
        ("Task", "Multi-turn shopping: each step depends on remembering earlier decisions/constraints."),
        ("Sample", f"40 tasks (random, seed 42) x 3 repeated runs, ~6 subtasks per task, per model ({n_models} models)."),
        ("Memory systems", f"{systems}  (stateless = no-memory baseline)"),
        ("Agent models (LLM backbones)", models),
        ("Judge / extraction model", "gpt-4o-mini (grades answer correctness + memory-use quality)"),
        ("Primary metrics", "progress_score (correct subtasks / total) and memory_utility_gain_vs_stateless"),
        ("Note on success", "Full-task success is all-or-nothing (every subtask correct) so it is near-zero; read progress_score."),
        ("Note on zep", "zep is EXCLUDED: its self-hosted backend never started (legacy Zep 0.27 config + "
                        "client/server API mismatch) so every run fell back to a local store. Preserved in "
                        "results/EXCLUDED_zep_fallback/."),
        ("Note on Bedrock", "Claude routes via Anthropic-direct on OpenRouter; Amazon Bedrock is ignored (disabled for this account)."),
        ("Generated from", "results/aggregated/ (make aggregate) + raw per-task rows"),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _load_raw() -> pd.DataFrame:
    root = ROOT / "results" / "by_memory_system"
    frames = []
    for jsonl in sorted(root.glob("*/*/results.jsonl")):
        rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[[c for c in df.columns if c not in DROP_COLS]]


def _csv(name: str) -> pd.DataFrame:
    p = AGG / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _failure_sheet() -> pd.DataFrame:
    fc = _csv("failure_categories.csv")
    if fc.empty:
        return fc
    fc = fc.copy()
    fc.insert(1, "reason", fc["failure_category"].map(FAILURE_LABELS).fillna(fc["failure_category"]))
    fc["what_it_means"] = fc["failure_category"].map(FAILURE_EXPLAIN).fillna("")
    return fc.sort_values(["memory_system", "count"], ascending=[True, False])


def _autofit_and_style(writer: pd.ExcelWriter) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(bold=True, color="FFFFFF")
    for ws in writer.book.worksheets:
        ws.freeze_panes = "A2"
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        # Auto column width (capped) from the longest value per column.
        for col in ws.columns:
            letter = col[0].column_letter
            longest = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[letter].width = min(max(longest + 2, 12), 60)


def main() -> None:
    sheets: dict[str, pd.DataFrame] = {}
    sheets["Overview"] = _overview()
    sheets["Definitions"] = pd.DataFrame(DEFINITIONS, columns=["Metric", "Meaning"])

    lb = _csv("leaderboard.csv")
    if not lb.empty:
        sheets["Leaderboard"] = lb.round(4)
    paper = _csv("paper_table.csv")
    if not paper.empty:
        sheets["Per-system summary"] = paper.round(4)
    trend = _csv("subtask_trend.csv")
    if not trend.empty:
        sheets["Learning curve"] = trend.round(4)
    fail = _failure_sheet()
    if not fail.empty:
        sheets["Failure taxonomy"] = fail
    per_model = _csv("summary_by_memory_model.csv")
    if not per_model.empty:
        sheets["Per-model summary"] = per_model.round(4)

    # Statistical analysis (paper §III-J): descriptive stats, two-way ANOVA,
    # Tukey post-hoc, and pairwise comparisons vs stateless (t, p, Cohen's d).
    for csv_name, sheet in [
        ("descriptive_stats.csv", "Descriptive stats"),
        ("stats_anova.csv", "Stats - ANOVA (2-way)"),
        ("pairwise_vs_stateless.csv", "Stats - vs stateless"),
        ("stats_posthoc.csv", "Stats - post-hoc Tukey"),
    ]:
        d = _csv(csv_name)
        if not d.empty:
            num = d.select_dtypes("float").columns
            d[num] = d[num].round(4)
            sheets[sheet] = d

    raw = _load_raw()
    if not raw.empty:
        # Round float columns for readability; keep every row.
        floats = raw.select_dtypes("float").columns
        raw[floats] = raw[floats].round(4)
        sheets["All results (raw)"] = raw

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        _autofit_and_style(writer)

    # Keep a copy under docs/ so the dashboard (which mounts docs/) can serve it.
    docs_copy = ROOT / "docs" / OUT.name
    docs_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT, docs_copy)

    total_rows = len(sheets.get("All results (raw)", []))
    print(f"Wrote {OUT} (+ {docs_copy}) with {len(sheets)} sheets ({total_rows} raw task rows).")
    for name, df in sheets.items():
        print(f"  - {name}: {len(df)} rows x {df.shape[1]} cols")

    _write_by_system_workbook()


# Comprehensive per-(system, model) metrics: (source column, header, decimals, scale).
# Mean over the 120 rows per cell unless noted. Grouped for readability.
_BY_SYSTEM_COLS = [
    ("__label", "LLM backbone", None, 1.0),
    # --- Performance & learning ---
    ("progress_score", "Progress score", 3, 1.0),
    ("progress_score__sd", "Progress score (SD)", 3, 1.0),
    ("memory_utility_gain_vs_stateless", "Gain vs stateless", 3, 1.0),
    ("score", "Mean subtask score", 3, 1.0),
    ("consistency_score", "Consistency", 3, 1.0),
    ("success", "Full-task success", 3, 1.0),
    ("pass_at_3", "pass@3", 3, 1.0),
    ("official_memoryarena_improvement_slope", "Improvement slope (within-task)", 3, 1.0),
    # --- Memory behaviour (RQ4) ---
    ("memory_relevance_score", "Relevance", 3, 1.0),
    ("memory_correctness_score", "Correctness", 3, 1.0),
    ("memory_usage_score", "Usage", 3, 1.0),
    ("retrieved_memory_count", "Retrieved memories (avg)", 2, 1.0),
    ("written_memory_count", "Written memories (avg)", 2, 1.0),
    ("memory_retrieval_latency_ms", "Retrieval latency (ms)", 0, 1.0),
    ("memory_write_latency_ms", "Write latency (ms)", 0, 1.0),
    # --- Efficiency / cost ---
    ("total_cost_usd", "Cost total ($)", 4, 1.0),
    ("estimated_cost_usd", "Agent cost ($)", 4, 1.0),
    ("judge_cost_usd", "Judge cost ($)", 4, 1.0),
    ("latency_ms", "Latency (s)", 1, 1 / 1000),
    ("input_tokens", "Input tokens", 0, 1.0),
    ("output_tokens", "Output tokens", 0, 1.0),
    ("reasoning_tokens", "Reasoning tokens", 0, 1.0),
    ("total_tokens", "Total tokens", 0, 1.0),
    ("turn_count", "Turns", 2, 1.0),
    ("tool_call_count", "Tool calls", 2, 1.0),
    ("n", "n (task rows)", 0, 1.0),
]


def _per_system_model_agg() -> pd.DataFrame:
    """Mean of every paper-relevant metric per (memory_system, model), computed
    from the full aggregated data (has gain + flattened official metrics)."""
    pq = AGG / "all_results.parquet"
    if not pq.exists():
        return pd.DataFrame()
    df = pd.read_parquet(pq)
    mean_cols = [c for c, *_ in _BY_SYSTEM_COLS
                 if c in df.columns and c not in ("__label", "n", "progress_score__sd")]
    g = df.groupby(["memory_system", "model_display"])
    agg = g[mean_cols].mean().reset_index()
    agg["n"] = g.size().values
    agg["progress_score__sd"] = g["progress_score"].std().values
    # pass@3 (a reliability metric, not a row-mean) comes from the summary view.
    sm = _csv("summary_by_memory_model.csv")
    if not sm.empty and "pass_at_3" in sm.columns:
        agg = agg.merge(
            sm[["memory_system", "model", "pass_at_3"]].rename(columns={"model": "model_display"}),
            on=["memory_system", "model_display"], how="left",
        )
    agg["__label"] = agg["model_display"].map(lambda x: _MODEL_LABEL.get(str(x), str(x)))
    return agg


def _write_by_system_workbook() -> None:
    """Second workbook: one sheet per memory system, one row per LLM backbone,
    with the FULL set of paper-relevant metrics as columns (readable, not raw)."""
    agg = _per_system_model_agg()
    if agg.empty:
        return
    order = {m: i for i, m in enumerate(_MODEL_ORDER)}
    systems = [s for s in _SYS_ORDER if s in set(agg["memory_system"])]
    out = ROOT / "benchmark_by_system.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # A self-contained definitions sheet so the workbook needs no external key.
        pd.DataFrame(DEFINITIONS, columns=["Metric", "Meaning"]).to_excel(
            writer, sheet_name="Definitions", index=False
        )
        for s in systems:
            sub = agg[agg["memory_system"] == s].copy()
            sub = sub.sort_values(by="__label", key=lambda c: c.map(lambda m: order.get(m, 99)))
            data: dict[str, list] = {}
            for src, header, dec, scale in _BY_SYSTEM_COLS:
                if src not in sub.columns:
                    continue
                col = sub[src] * scale if scale != 1.0 else sub[src]
                if dec is None:
                    data[header] = col.tolist()
                elif dec == 0:
                    data[header] = [int(round(v)) if pd.notna(v) else None for v in col]
                else:
                    data[header] = [round(float(v), dec) if pd.notna(v) else None for v in col]
            pd.DataFrame(data).to_excel(writer, sheet_name=s[:31], index=False)
        _autofit_and_style(writer)
    docs_copy = ROOT / "docs" / out.name
    docs_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, docs_copy)
    print(f"Wrote {out} (+ {docs_copy}) with {len(systems)} per-system sheets, "
          f"{sum(1 for c in _BY_SYSTEM_COLS) - 1} metric columns each.")


if __name__ == "__main__":
    main()
