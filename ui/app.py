"""Memory Agent Benchmark Dashboard (Streamlit + Plotly).

Reads the aggregated outputs and visualizes all calculated metrics. If no
results exist yet, it shows a friendly message with the commands to run the
benchmarks. Filtered results are downloadable as JSON and CSV, and each memory
system's result bundle can be downloaded too.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make the benchmark package importable when run via `streamlit run ui/app.py`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

RESULTS = Path(__file__).resolve().parents[1] / "results"
AGG = RESULTS / "aggregated"
DOCS = Path(__file__).resolve().parents[1] / "docs"

# Columns that are always-null or pure clutter on the current OpenAI + MemoryArena
# setup. Hidden from the drill-down and the filtered downloads; the raw
# per-system results.jsonl keeps every field.
_DROP_COLS = {
    "rule_compliance", "ux_score", "error", "model_unavailable_reason",
    "openrouter_model_catalog_created_at", "openrouter_generation_id",
    "openrouter_request_id", "openrouter_supported_parameters",
    "openrouter_provider_policy", "openrouter_model_pricing",
    "practical_correctness", "usefulness", "dfx_constraint_adherence",
    "reasoning_clarity", "engineering_applicability",
    "model_slot", "cached_input_tokens", "memory_store_size_bytes",
}


def _curated(df: pd.DataFrame) -> pd.DataFrame:
    """Drop always-null / clutter columns for display + download."""
    return df[[c for c in df.columns if c not in _DROP_COLS]]


# Plain-language definitions for every metric/column shown, aligned to the paper
# (Section III-H). Surfaced via a "Details" button next to each table/chart.
_GLOSSARY: dict[str, str] = {
    "rank": "Position, ranked by progress score then memory-utility gain.",
    "memory_system": "The memory mechanism under test (stateless = no-memory baseline).",
    "model": "LLM backbone used (this run: gpt-5.5).",
    "model_display": "Concrete resolved model id.",
    "models_used": "LLM backbone(s) present in this group.",
    "benchmark": "Benchmark suite (memoryarena).",
    "domain": "Task config within the benchmark (bundled_shopping).",
    "task_id": "Task identifier.",
    "seed": "Repeat index — each seed is an independent run with isolated memory (measures variance, not learning).",
    "n": "Number of result rows in the group (tasks x seeds).",
    "progress_score": "PRIMARY. Fraction of a task's subtasks judged correct (correct / total). The paper's partial-completion metric (PS).",
    "score": "Mean of the per-subtask LLM-judge correctness scores (0-1).",
    "mean_score": "Mean of the per-subtask LLM-judge correctness scores (0-1).",
    "success": "Full-task success: 1 only if EVERY subtask is correct. Near-zero on hard tasks - read progress score instead.",
    "full_task_success": "1 only if EVERY subtask in the task is correct (all-or-nothing). Near-zero on hard tasks.",
    "task_success_rate": "Mean full-task success (every subtask correct) across the group.",
    "pass_at_k": "Fraction of tasks solved in at least one of the k repeated runs (reliability across seeds); k = num_runs.",
    "memory_utility_gain_vs_stateless": "PRIMARY. This system's score minus the matched stateless row's (same task+seed). Positive = memory helped.",
    "gain_vs_stateless": "This system's score minus the matched stateless score (same task+seed). Positive = memory helped.",
    "consistency": "1 - contradictions against the per-task fact ledger (staying consistent with earlier decisions/constraints).",
    "consistency_score": "1 - contradictions against the per-task fact ledger.",
    "relevance": "Judge: were the retrieved memories relevant to the current question? (blank for stateless - no retrieval).",
    "memory_relevance_score": "Judge: were the retrieved memories relevant to the current question?",
    "correctness": "Judge: were the retrieved memories factually correct vs the task's fact ledger?",
    "memory_correctness_score": "Judge: were the retrieved memories factually correct vs the fact ledger?",
    "usage": "Judge: did the agent actually use the retrieved memory in its answer?",
    "memory_usage_score": "Judge: did the agent actually use the retrieved memory in its answer?",
    "improvement_slope": "Slope of score across a task's sequential subtasks (the paper's 'improvement over time'). ~0/negative as later subtasks get harder.",
    "subtask_index": "Position of the subtask within the task (0 = first). Later positions depend on remembering earlier ones.",
    "retrieved": "Avg number of memory entries retrieved per task.",
    "retrieved_memory_count": "Avg number of memory entries retrieved per task.",
    "written": "Avg number of memory entries written per task.",
    "written_memory_count": "Avg number of memory entries written per task.",
    "total_cost_usd": "Avg USD per task = agent LLM cost + judge LLM cost.",
    "mean_cost_usd": "Avg USD per task (agent + judge).",
    "latency_ms": "Avg wall-clock ms per task.",
    "mean_latency_ms": "Avg wall-clock ms per task.",
    "total_tokens": "Avg agent input+output tokens per task.",
    "failure_category": "Where the memory process failed (e.g. failed retrieval, incorrect use of retrieved memory) - paper's failure taxonomy.",
    "count": "Number of occurrences.",
}


# Human-readable label + plain-English cause for each failure taxonomy value
# (paper Section III-K). The chart shows the short label; the full explanation is
# in the bar hover text and the list below the chart. Covers every FailureCategory
# in src/benchmark/schemas.py so future runs render regardless of which fire.
_FAILURE_LABELS: dict[str, str] = {
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
_FAILURE_EXPLAIN: dict[str, str] = {
    "missing_memory_write": "Task failed and the system stored nothing — no memory to learn from.",
    "irrelevant_memory_write": "The system wrote memories, but they were off-topic for the task.",
    "failed_retrieval": "Memories existed but retrieval returned nothing for the query.",
    "incorrect_or_outdated_retrieval": "Retrieved memories were factually wrong or stale (memory-correctness < 0.5).",
    "incorrect_use_of_retrieved_memory": (
        "Correct memories were retrieved, but the agent barely grounded its answer in them "
        "(memory-usage < 0.3) — typically because what was retrieved wasn't relevant to the "
        "specific subtask (low retrieval relevance), so the agent ignored it and answered wrong."
    ),
    "over_reliance_on_prior_feedback": "The agent leaned too hard on earlier feedback instead of the current task.",
    "inconsistency_across_iterations": "Answers varied across repeated runs of the same task (consistency < 0.5).",
    "model_reasoning_error": "Memory worked, but the model reasoned to the wrong answer.",
    "tool_use_error": "The agent misused a tool while solving the task.",
    "benchmark_environment_error": "An error/crash in the backend or harness, not a memory decision.",
    "unknown": "Failure that didn't match any specific taxonomy branch.",
}


def _details(label: str, keys: Any) -> None:
    """A 'Details' button next to a table/chart explaining each metric shown."""
    def _defn(k: str) -> str | None:
        if k in _GLOSSARY:
            return _GLOSSARY[k]
        # Dynamic pass@k column (e.g. "pass_at_3") maps to the generic entry.
        if k.startswith("pass_at_") and k[len("pass_at_"):].isdigit():
            return _GLOSSARY["pass_at_k"]
        return None

    lines = [f"- **{k}** — {_defn(k)}" for k in dict.fromkeys(keys) if _defn(k)]
    if not lines:
        return
    try:
        box = st.popover(f"ℹ️ Details — {label}")
    except Exception:  # older streamlit without popover
        box = st.expander(f"ℹ️ Details — {label}")
    with box:
        st.markdown("\n".join(lines))


st.set_page_config(page_title="Memory Agent Benchmark Dashboard", layout="wide")


def _load_csv(name: str) -> pd.DataFrame:
    path = AGG / name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return pd.DataFrame()


def _csv_download(df: pd.DataFrame, filename: str, label: str, key: str) -> None:
    """Offer a DataFrame as a downloadable CSV (no-op when empty).

    Used to let a reviewer grab each paper-ready aggregated table directly.
    Streamlit needs a unique key per download button on the page.
    """
    if df is None or df.empty:
        return
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def _load_from_per_system() -> pd.DataFrame:
    """Load task rows from per-memory-system folders (live during a long run)."""
    root = RESULTS / "by_memory_system"
    if not root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for jsonl in sorted(root.glob("*/*/results.jsonl")):
        try:
            rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                frames.append(pd.DataFrame(rows))
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        return pd.DataFrame()
    return _enrich_models(pd.concat(frames, ignore_index=True))


def _enrich_models(df: pd.DataFrame) -> pd.DataFrame:
    """Show concrete OpenRouter model ids (not only config slot names)."""
    if df.empty or "model" not in df.columns:
        return df
    out = df.copy()
    if "model_slot" not in out.columns:
        out["model_slot"] = out["model"]
    resolved = out["resolved_model"] if "resolved_model" in out.columns else pd.Series(dtype=str)
    out["model_display"] = resolved.fillna(out["model_slot"])
    return out


def _load_all() -> pd.DataFrame:
    """Prefer aggregated parquet; merge in per-system rows for in-progress runs."""
    agg = pd.DataFrame()
    path = AGG / "all_results.parquet"
    if path.exists():
        try:
            agg = _enrich_models(pd.read_parquet(path))
        except Exception:  # noqa: BLE001
            agg = pd.DataFrame()

    live = _load_from_per_system()
    if agg.empty:
        return live
    if live.empty:
        return agg

    key_cols = [c for c in ("run_id", "memory_system", "model", "benchmark", "domain", "task_id", "seed")
                if c in agg.columns and c in live.columns]
    if key_cols:
        combined = pd.concat([agg, live], ignore_index=True).drop_duplicates(subset=key_cols, keep="last")
        return _enrich_models(combined)
    return _enrich_models(pd.concat([agg, live], ignore_index=True))


def _empty_state() -> None:
    st.title("Memory Agent Benchmark Dashboard")
    st.warning("No results found yet. Run a memory system, then aggregate.")
    st.markdown(
        """
Run **one memory system at a time** (see command list in the progress panel above):

```bash
make clear && make init-run    # once per campaign
source scripts/init_run.sh     # sets RUN_ID
make ui                        # dashboard (terminal 1)
make stateless                 # then each system, same RUN_ID
make hindsight
# ... mem0, zep, acontext
make aggregate
```

Or locally:

```bash
python scripts/run_memory_system.py --memory-system stateless
python scripts/aggregate_results.py
streamlit run ui/app.py
```
        """
    )


_STATUS_ICON = {
    "done": "✅ done",
    "running": "🔄 running",
    "pending": "⏳ pending",
    "failed": "❌ failed",
}


def _progress_panel() -> None:
    """Show run progress: done / running / remaining memory systems."""
    st.subheader("Run progress")
    try:
        from benchmark.progress import MEMORY_SYSTEMS_ORDER, load_status, summary_for_ui

        table = summary_for_ui(RESULTS)
        status = load_status(RESULTS)
        run_id = status.get("run_id") or "(not started)"
        st.caption(f"**RUN_ID:** `{run_id}` — re-use the same id for every system in one campaign.")

        done_n = int((table["status"] == "done").sum())
        total = len(MEMORY_SYSTEMS_ORDER)
        st.progress(done_n / total if total else 0.0, text=f"{done_n} / {total} memory systems complete")

        display = table.copy()
        display["status"] = display["status"].map(lambda s: _STATUS_ICON.get(s, s))
        st.dataframe(display, use_container_width=True, hide_index=True)

        current = status.get("current_system")
        pending = [r.memory_system for r in table.itertuples() if r.status == "pending"]
        if current:
            st.info(f"**Running now:** `{current}`")
        elif pending:
            nxt = pending[0]
            st.code(
                f"export RUN_ID={run_id}\nmake {nxt}\n# or: bash scripts/run_one.sh {nxt}",
                language="bash",
            )
        elif done_n == total:
            st.success("All memory systems finished. Run `make aggregate` if you have not yet.")
        else:
            st.warning("Set a RUN_ID and run systems in order (stateless first).")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Progress tracker unavailable: {exc}")


def _metric_card(col, label: str, series: pd.Series, fmt: str = "{:.3f}") -> None:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    value = fmt.format(vals.mean()) if len(vals) else "n/a"
    col.metric(label, value)


def _study_header(df: pd.DataFrame) -> None:
    """Brief description of the experiment (dynamic facts + a details expander)."""

    def _uniq(col: str, fallback: str) -> str:
        if df.empty or col not in df.columns:
            return fallback
        vals = sorted({str(x) for x in df[col].dropna().unique()})
        return ", ".join(vals) if vals else fallback

    benchmark = _uniq("benchmark", "memoryarena")
    domain = _uniq("domain", "bundled_shopping")
    systems = _uniq("memory_system", "stateless, hindsight, mem0, zep, acontext")
    model = _uniq("model_display", _uniq("resolved_model", "gpt-5.5"))
    n_tasks = int(df["task_id"].nunique()) if (not df.empty and "task_id" in df) else 0
    n_seeds = int(df["seed"].nunique()) if (not df.empty and "seed" in df) else 0

    st.markdown(
        "**Memory Matters** — evaluating how different **memory mechanisms** affect an "
        "LLM agent's learning behavior on iterative, interdependent tasks. Every memory system "
        "runs the *same* tasks through *one* agent pipeline; a **stateless** agent is the baseline, "
        "and we measure whether memory improves performance on later, memory-dependent steps."
    )
    st.markdown(
        f"- **Benchmark / dataset:** `{benchmark}` — config `{domain}` "
        f"(HuggingFace `ZexueHe/memoryarena`): a multi-turn shopping task where each purchase step "
        f"depends on remembering earlier decisions, budget, and constraints.\n"
        f"- **Sample:** {n_tasks} tasks × {n_seeds} repeated runs (random sample, seed 42).\n"
        f"- **Memory systems:** {systems}. Here **`stateless`** is the no-memory baseline "
        f"(control); the others each add a memory mechanism and are measured against it.\n"
        f"- **Agent model:** {model} (reasoning effort: medium). **LLM judge:** gpt-4o-mini "
        f"(grades answer correctness + memory-use quality; memory-system extraction also gpt-4o-mini).\n"
        f"- **Primary metrics:** **progress score** (correct subtasks / total) and **memory-utility "
        f"gain vs stateless**. Full-task success is all-or-nothing, so it is near-zero on this hard task."
    )
    with st.expander("Full study details & research questions"):
        st.markdown(
            "This study integrates existing memory mechanisms into a single agentic pipeline and "
            "compares them under identical tasks (a factorial design over memory system × LLM "
            "backbone × iteration depth). The goal is not a new memory mechanism, but a controlled "
            "comparison of existing ones.\n\n"
            "**Research questions**\n"
            "- **RQ1** — How do different memory mechanisms influence agent performance and learning "
            "behavior on iterative, interdependent tasks?\n"
            "- **RQ2** — Do memory-augmented agents improve over repeated interactions and transfer "
            "knowledge, compared with a stateless agent?\n"
            "- **RQ3** — How does the choice of LLM backbone affect memory-based learning?\n"
            "- **RQ4** — How effectively do agents *retrieve* and *use* stored memory (relevance, "
            "correctness, usage)?\n\n"
            "**Scoring.** Answers are graded by an LLM judge — deterministic token overlap cannot "
            "score MemoryArena's structured (product-ID) answers. **Memory-utility gain** joins each "
            "memory-system row to the matched stateless row (same task + seed) and is computed at "
            "`make aggregate`. The **learning curve** below (mean score by subtask position) shows the "
            "stateless agent degrading on later steps while memory systems hold up.\n\n"
            "*Note:* the repeated runs use isolated memory per seed (they measure variance/consistency, "
            "not cross-run learning); within-task learning is the subtask-position trend."
        )
    st.divider()


def _reports_section() -> None:
    """Read + download the paper-ready report, figures, and Excel workbook."""
    report = DOCS / "RESULTS_REPORT.md"
    xlsx = DOCS / "benchmark_results.xlsx"
    by_system = DOCS / "benchmark_by_system.xlsx"
    figs = sorted((DOCS / "figures").glob("*.png")) if (DOCS / "figures").exists() else []
    if not (report.exists() or xlsx.exists() or figs):
        return

    _XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    st.subheader("📄 Report & downloads")
    c1, c2, c3 = st.columns(3)
    if xlsx.exists():
        c1.download_button(
            "⬇ Excel — full workbook",
            data=xlsx.read_bytes(),
            file_name="benchmark_results.xlsx",
            mime=_XLSX_MIME,
            key="dl_xlsx_main",
        )
    if by_system.exists():
        c2.download_button(
            "⬇ Excel — per memory system",
            data=by_system.read_bytes(),
            file_name="benchmark_by_system.xlsx",
            mime=_XLSX_MIME,
            key="dl_xlsx_bysystem",
        )
    if report.exists():
        c3.download_button(
            "⬇ Report (.md)",
            data=report.read_text(encoding="utf-8"),
            file_name="RESULTS_REPORT.md",
            mime="text/markdown",
            key="dl_report_md",
        )

    if report.exists():
        with st.expander("Read the full results report", expanded=False):
            text = report.read_text(encoding="utf-8")
            # st.markdown can't load the report's relative figure paths, so render
            # the text and drop the figures in explicitly below.
            st.markdown(text)
            for fp in figs:
                st.image(str(fp), caption=fp.name, use_container_width=True)
    elif figs:
        with st.expander("Figures", expanded=False):
            for fp in figs:
                st.image(str(fp), caption=fp.name, use_container_width=True)
    st.divider()


def main() -> None:
    st.title("Memory Agent Benchmark Dashboard")

    # Auto-refresh is only useful while a campaign is still running. Once every
    # tracked memory system is "done", pause it automatically (still toggleable).
    try:
        from benchmark.progress import load_status

        _systems = (load_status(RESULTS).get("systems") or {}).values()
        all_done = bool(_systems) and all(s.get("status") == "done" for s in _systems)
    except Exception:  # noqa: BLE001
        all_done = False

    if all_done:
        st.sidebar.caption("✅ All systems complete — live auto-refresh paused.")
    if st.sidebar.checkbox("Auto-refresh (30s)", value=not all_done):
        st.markdown(
            '<meta http-equiv="refresh" content="30">',
            unsafe_allow_html=True,
        )

    df = _load_all()
    _study_header(df)
    _reports_section()
    _progress_panel()
    st.divider()

    if df.empty:
        _empty_state()
        return

    # ---- Filters ----
    st.sidebar.header("Filters")

    def _multi(label: str, col: str) -> list:
        if col not in df.columns:
            return []
        options = sorted([x for x in df[col].dropna().unique()])
        return st.sidebar.multiselect(label, options, default=options)

    f_bench = _multi("Benchmark", "benchmark")
    f_domain = _multi("Domain", "domain")
    f_system = _multi("Memory system", "memory_system")
    f_model = _multi("Model (resolved OpenRouter id)", "model_display")
    f_run = _multi("Run id", "run_id")

    view = df.copy()
    for col, selected in (
        ("benchmark", f_bench),
        ("domain", f_domain),
        ("memory_system", f_system),
        ("model_display", f_model),
        ("run_id", f_run),
    ):
        if col in view.columns and selected:
            view = view[view[col].isin(selected)]

    if view.empty:
        st.info("No rows match the current filters.")
        return

    # ---- Metric cards ----
    # Headline the graded per-subtask metrics (progress score + memory-utility
    # gain). "Full-task success" is all-or-nothing across a task's ~6 subtasks, so
    # it is near-zero on hard tasks and kept only as a secondary, labeled card.
    st.subheader("Headline metrics (filtered)")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    _metric_card(c1, "Progress score", view["progress_score"])
    _metric_card(c2, "Memory utility gain",
                 view.get("memory_utility_gain_vs_stateless", pd.Series(dtype=float)))
    _metric_card(c3, "Mean subtask score", view["score"])
    _metric_card(c4, "Consistency", view.get("consistency_score", pd.Series(dtype=float)))
    _metric_card(c5, "Full-task success (all subtasks)", view["success"].astype(float))
    total_cost = view["total_cost_usd"] if "total_cost_usd" in view.columns else view["estimated_cost_usd"]
    _metric_card(c6, "Total cost / task", total_cost, fmt="${:.6f}")
    _metric_card(c7, "Latency (ms)", view["latency_ms"], fmt="{:.0f}")
    st.caption(
        "**Progress score** (correct subtasks / total) and **memory utility gain** (this row's score − "
        "the matched stateless row's) are the primary metrics. **Full-task success** requires *every* "
        "subtask correct, so it is near-zero on hard tasks — read progress/gain instead. Cost is agent + "
        "judge. Memory utility gain is derived at `make aggregate` (needs the stateless baseline) and is "
        "null in the per-system `results.jsonl` by design."
    )
    _details("Headline metrics", [
        "progress_score", "memory_utility_gain_vs_stateless", "score",
        "consistency_score", "full_task_success", "total_cost_usd", "latency_ms",
    ])

    # ---- Leaderboard ----
    st.subheader("Leaderboard")
    st.caption(
        "Ranks memory systems, aggregated across **all** LLM backbones in the dataset. "
        "For a per-backbone breakdown see the Model comparison section below."
    )
    leaderboard = _load_csv("leaderboard.csv")
    if not leaderboard.empty:
        st.dataframe(leaderboard, use_container_width=True)
        _csv_download(leaderboard, "leaderboard.csv", "⬇ Download leaderboard (CSV)", "dl_leaderboard")
        _details("Leaderboard", leaderboard.columns)
    else:
        st.dataframe(
            view.groupby("memory_system")["success"].mean().reset_index(),
            use_container_width=True,
        )

    # ---- Model comparison (RQ3: how the LLM backbone affects each system) ----
    st.subheader("Model comparison — memory system × LLM backbone (RQ3)")
    mm = _load_csv("summary_by_memory_model.csv")
    if not mm.empty and "model" in mm.columns:
        # Respect the sidebar model filter if the user narrowed it.
        if "model_display" in view.columns:
            keep = set(view["model_display"].astype(str))
            if keep:
                mm = mm[mm["model"].astype(str).isin(keep)]
    if not mm.empty and "model" in mm.columns:
        gain_col = "memory_utility_gain_vs_stateless"
        st.caption(
            "Every memory system evaluated under each LLM backbone. progress_score and "
            "gain-vs-stateless are the primary metrics (higher = better). This is the cross-model "
            "view — the per-system tables below reflect whatever models are selected in the sidebar."
        )
        # Per-model scorecard: mean progress + mean gain across memory systems.
        models = sorted(mm["model"].astype(str).unique())
        for col, mdl in zip(st.columns(len(models)), models):
            sub = mm[mm["model"].astype(str) == mdl]
            ps = sub["progress_score"].mean() if "progress_score" in sub else float("nan")
            gv = (
                sub[sub["memory_system"] != "stateless"][gain_col].mean()
                if gain_col in sub else float("nan")
            )
            col.metric(mdl, f"progress {ps:.3f}", f"mean gain {gv:+.3f}")
        show_cols = [c for c in [
            "memory_system", "model", "n", "progress_score", "mean_score", gain_col,
            "consistency_score", "pass_at_3", "mean_cost_usd", "mean_latency_ms",
        ] if c in mm.columns]
        st.dataframe(
            mm[show_cols].round(3).sort_values(["memory_system", "model"]),
            use_container_width=True,
        )
        _csv_download(mm, "summary_by_memory_model.csv",
                      "⬇ Download model comparison (CSV)", "dl_modelcmp")
        if "progress_score" in mm.columns:
            st.plotly_chart(
                px.bar(mm.sort_values("memory_system"), x="memory_system", y="progress_score",
                       color="model", barmode="group",
                       labels={"progress_score": "progress score", "model": "LLM backbone"}),
                use_container_width=True,
            )
        if gain_col in mm.columns:
            g = mm[mm["memory_system"] != "stateless"]
            st.plotly_chart(
                px.bar(g.sort_values("memory_system"), x="memory_system", y=gain_col,
                       color="model", barmode="group",
                       labels={gain_col: "gain vs stateless", "model": "LLM backbone"}),
                use_container_width=True,
            )
        _details("Model comparison", [
            "memory_system", "progress_score", "memory_utility_gain_vs_stateless",
            "consistency_score", "pass_at_3", "mean_cost_usd", "mean_latency_ms",
        ])
    else:
        st.caption("Model comparison appears once results are aggregated (`make aggregate`).")

    # ---- Paper-ready per-system summary (Section III-H) ----
    st.subheader("Per-system summary (paper metrics)")
    st.caption(
        "Paper §III-H metrics per memory system, aggregated across **all** models. "
        "Use the Model comparison section for the per-backbone split."
    )
    paper = _load_csv("paper_table.csv")
    if not paper.empty:
        st.dataframe(paper.round(3), use_container_width=True)
        _csv_download(paper, "paper_table.csv", "⬇ Download paper summary table (CSV)", "dl_paper")
        _details("Per-system summary", paper.columns)
        if "gain_vs_stateless" in paper.columns:
            st.subheader("Memory utility gain vs stateless")
            gdf = paper[paper["memory_system"] != "stateless"].sort_values("gain_vs_stateless")
            st.plotly_chart(
                px.bar(gdf, x="memory_system", y="gain_vs_stateless",
                       labels={"gain_vs_stateless": "score gain vs stateless"}),
                use_container_width=True,
            )
            _details("Gain vs stateless chart", ["memory_system", "gain_vs_stateless"])
    else:
        st.caption("Run `make aggregate` to build the paper summary table.")

    # RQ2 learning curve: mean score across sequential subtask positions.
    st.subheader("Learning curve — mean score by subtask position (within task)")
    trend = _load_csv("subtask_trend.csv")
    if not trend.empty:
        trend = trend.sort_values("subtask_index")
        facet = None
        # Map the model slot (from the raw-event path) to its display name and,
        # when >1 model is present, show one small-multiple per model backbone.
        if "model" in trend.columns and {"model", "model_display"} <= set(view.columns):
            slot2disp = dict(zip(view["model"].astype(str), view["model_display"].astype(str)))
            trend["model_display"] = trend["model"].map(slot2disp).fillna(trend["model"])
            keep = set(view["model_display"].astype(str))
            if keep:
                trend = trend[trend["model_display"].isin(keep)]
            if not trend.empty and trend["model_display"].nunique() > 1:
                facet = "model_display"
        fig = px.line(
            trend, x="subtask_index", y="mean_score", color="memory_system",
            markers=True, facet_col=facet,
            labels={"subtask_index": "subtask position", "mean_score": "mean score"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Within-task performance across sequential subtasks (the paper's 'improvement over "
            "time'). The repeated runs use isolated memory per seed, so they capture variance, not "
            "cross-run learning."
        )
        _csv_download(trend, "subtask_trend.csv", "⬇ Download learning-curve data (CSV)", "dl_trend")
        _details("Learning curve", ["subtask_index", "mean_score", "memory_system"])
    else:
        st.caption("Run `make aggregate` to build the learning curve.")

    # ---- Charts ----
    st.subheader("Score by memory system")
    by_sys = view.groupby("memory_system")["score"].mean().reset_index()
    st.plotly_chart(px.bar(by_sys, x="memory_system", y="score"), use_container_width=True)
    _details("Score by memory system", ["memory_system", "score"])

    st.subheader("Memory system x model success rate (heatmap)")
    model_col = "model_display" if "model_display" in view.columns else "model"
    pivot = (
        view.assign(success=view["success"].astype(float))
        .pivot_table(index="memory_system", columns=model_col, values="success", aggfunc="mean")
    )
    if not pivot.empty:
        st.plotly_chart(
            px.imshow(pivot, text_auto=".2f", aspect="auto", color_continuous_scale="Blues"),
            use_container_width=True,
        )
        _details("Success heatmap", ["memory_system", "model_display", "success"])

    st.subheader("Progress score by iteration (seed)")
    if "seed" in view.columns:
        seed_trend = view.groupby(["memory_system", "seed"])["progress_score"].mean().reset_index()
        st.plotly_chart(
            px.line(seed_trend, x="seed", y="progress_score", color="memory_system", markers=True),
            use_container_width=True,
        )
        _details("Progress by iteration", ["seed", "progress_score", "memory_system"])

    st.subheader("Cost vs success")
    model_col = "model_display" if "model_display" in view.columns else "model"
    cost_col = "total_cost_usd" if "total_cost_usd" in view.columns else "estimated_cost_usd"
    cvs = view.groupby(["memory_system", model_col]).agg(
        cost=(cost_col, "mean"), progress_score=("progress_score", "mean")
    ).reset_index()
    st.plotly_chart(
        px.scatter(cvs, x="cost", y="progress_score", color="memory_system", symbol=model_col,
                   labels={"cost": "total cost / task ($)"},
                   hover_data=["memory_system", model_col]),
        use_container_width=True,
    )
    _details("Cost vs progress", ["total_cost_usd", "progress_score", "memory_system"])

    # ---- Failure taxonomy (paper Section III-K) ----
    st.subheader("Failure taxonomy (paper §III-K)")
    st.caption(
        "Heuristic classification: each **failed** task is assigned a single reason, *derived* from "
        "the measured sub-metrics (retrieval count, memory-correctness, memory-usage, consistency) — "
        "it is a diagnostic label, not an independently measured metric. See the assignment rule below."
    )
    fails = _load_csv("failure_categories.csv")
    if fails.empty and "failure_category" in view.columns:
        tmp = view[view["failure_category"].notna() & (view["failure_category"] != "")]
        if not tmp.empty:
            fails = tmp.groupby(["memory_system", "failure_category"]).size().rename(
                "count").reset_index()
    if not fails.empty:
        fails = fails.copy()
        fails["reason"] = fails["failure_category"].map(_FAILURE_LABELS).fillna(
            fails["failure_category"]
        )
        fails["why"] = fails["failure_category"].map(_FAILURE_EXPLAIN).fillna("")
        # Wrap the hover explanation so the tooltip stays readable.
        fails["why_wrapped"] = fails["why"].apply(
            lambda t: "<br>".join(textwrap.wrap(t, 60)) if t else ""
        )
        fig = px.bar(
            fails,
            x="reason",
            y="count",
            color="memory_system",
            barmode="group",
            custom_data=["why_wrapped"],
            labels={"reason": "Failure reason", "count": "Failed tasks"},
        )
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y} failed tasks"
            "<br><br>%{customdata[0]}<extra></extra>"
        )
        st.plotly_chart(fig, use_container_width=True)
        # Spell out every failure reason present in the current view.
        present = [c for c in _FAILURE_LABELS if c in set(fails["failure_category"])]
        st.markdown(
            "**What each failure reason means**\n\n"
            + "\n".join(
                f"- **{_FAILURE_LABELS[c]}** — {_FAILURE_EXPLAIN[c]}" for c in present
            )
        )
        with st.expander("How the failure reason is assigned (rule)"):
            st.markdown(
                "Each failed task is labelled by the first matching rule "
                "(`_classify_failure`, runner.py) — measured signals in **bold**:\n\n"
                "1. task raised an **error** → *Environment/backend error*\n"
                "2. **retrieved = 0** and **written = 0** → *No memory written*\n"
                "3. **retrieved = 0** → *Nothing retrieved*\n"
                "4. **memory-correctness < 0.5** → *Wrong/outdated memory retrieved*\n"
                "5. **memory-usage < 0.3** → *Retrieved but not used*\n"
                "6. **consistency < 0.5** → *Inconsistent across runs*\n"
                "7. otherwise → *Model reasoning error*"
            )
        _csv_download(
            fails[["memory_system", "failure_category", "count"]],
            "failure_categories.csv",
            "⬇ Download failure taxonomy (CSV)",
            "dl_failures",
        )
        _details("Failure categories", ["failure_category", "count", "memory_system"])
    else:
        st.caption("No failures recorded for the current selection.")

    # ---- Drill-down ----
    st.subheader("Drill-down")
    drill_cols = [
        c
        for c in [
            "task_id", "memory_system", "model_display", "resolved_model",
            "benchmark", "domain", "seed",
            "progress_score", "score", "memory_utility_gain_vs_stateless",
            "retrieved_memory_count", "written_memory_count", "memory_usage_score",
            "total_cost_usd", "failure_category",
        ]
        if c in view.columns
    ]
    st.dataframe(view[drill_cols], use_container_width=True, height=350)
    _details("Drill-down", drill_cols)

    # ---- Downloads (curated: clutter/always-null columns dropped) ----
    st.subheader("Download (filtered view)")
    curated = _curated(view)
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download CSV",
        curated.to_csv(index=False).encode("utf-8"),
        file_name="filtered_results.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Download JSON",
        json.dumps(curated.to_dict(orient="records"), default=str).encode("utf-8"),
        file_name="filtered_results.json",
        mime="application/json",
    )

    # Per-memory-system bundle downloads.
    st.subheader("Download per-memory-system bundles")
    by_system_root = RESULTS / "by_memory_system"
    if by_system_root.exists():
        for system_dir in sorted(by_system_root.glob("*")):
            csvs = sorted(system_dir.glob("*/results.csv"))
            if not csvs:
                continue
            frames = [pd.read_csv(c) for c in csvs]
            bundle = pd.concat(frames, ignore_index=True)
            cols = st.columns(2)
            cols[0].download_button(
                f"{system_dir.name}: CSV",
                bundle.to_csv(index=False).encode("utf-8"),
                file_name=f"{system_dir.name}_results.csv",
                mime="text/csv",
                key=f"{system_dir.name}_csv",
            )
            cols[1].download_button(
                f"{system_dir.name}: JSON",
                json.dumps(bundle.to_dict(orient="records"), default=str).encode("utf-8"),
                file_name=f"{system_dir.name}_results.json",
                mime="application/json",
                key=f"{system_dir.name}_json",
            )


if __name__ == "__main__":
    main()
