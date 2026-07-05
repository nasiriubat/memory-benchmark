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
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "results" / "aggregated"
OUT = ROOT / "benchmark_results.xlsx"

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
]


def _overview() -> pd.DataFrame:
    rows = [
        ("Study", "Memory Matters - Evaluating Learning Behavior in LLM-Based Agents (ICSE TR)"),
        ("Benchmark", "MemoryArena"),
        ("Dataset", "HuggingFace ZexueHe/memoryarena, config: bundled_shopping"),
        ("Task", "Multi-turn shopping: each step depends on remembering earlier decisions/constraints."),
        ("Sample", "40 tasks (random, seed 42) x 3 repeated runs, ~6 subtasks per task."),
        ("Memory systems", "stateless (baseline), hindsight, mem0, zep, acontext"),
        ("Agent model", "gpt-5.5 (reasoning effort: medium)"),
        ("Judge / extraction model", "gpt-4o-mini (grades answer correctness + memory-use quality)"),
        ("Primary metrics", "progress_score (correct subtasks / total) and memory_utility_gain_vs_stateless"),
        ("Note on success", "Full-task success is all-or-nothing (every subtask correct) so it is near-zero; read progress_score."),
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

    total_rows = len(sheets.get("All results (raw)", []))
    print(f"Wrote {OUT} with {len(sheets)} sheets ({total_rows} raw task rows).")
    for name, df in sheets.items():
        print(f"  - {name}: {len(df)} rows x {df.shape[1]} cols")


if __name__ == "__main__":
    main()
