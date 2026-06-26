"""Memory Agent Benchmark Dashboard (Streamlit + Plotly).

Reads the aggregated outputs and visualizes all calculated metrics. If no
results exist yet, it shows a friendly message with the commands to run the
benchmarks. Filtered results are downloadable as JSON and CSV, and each memory
system's result bundle can be downloaded too.
"""

from __future__ import annotations

import json
import sys
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

st.set_page_config(page_title="Memory Agent Benchmark Dashboard", layout="wide")


def _load_csv(name: str) -> pd.DataFrame:
    path = AGG / name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return pd.DataFrame()


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
# ... mem0, zep, supermemory
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


def main() -> None:
    st.title("Memory Agent Benchmark Dashboard")

    if st.sidebar.checkbox("Auto-refresh (30s)", value=True):
        st.markdown(
            '<meta http-equiv="refresh" content="30">',
            unsafe_allow_html=True,
        )

    _progress_panel()
    st.divider()

    df = _load_all()
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
    st.subheader("Headline metrics (filtered)")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    _metric_card(c1, "Task success rate", view["success"].astype(float))
    if "official_STATE_Bench_task_completion_pass_5" in view.columns:
        _metric_card(c2, "pass@5", view["official_STATE_Bench_task_completion_pass_5"])
    else:
        _metric_card(c2, "pass@5 (success)", view["success"].astype(float))
    _metric_card(c3, "Progress score", view["progress_score"])
    _metric_card(c4, "Consistency", view.get("consistency_score", pd.Series(dtype=float)))
    _metric_card(c5, "Memory utility gain",
                 view.get("memory_utility_gain_vs_stateless", pd.Series(dtype=float)))
    _metric_card(c6, "Cost / task", view["estimated_cost_usd"], fmt="${:.6f}")
    _metric_card(c7, "Latency (ms)", view["latency_ms"], fmt="{:.0f}")
    st.caption(
        "Memory utility gain is derived at aggregation (this row's score − the matched stateless "
        "row's score). It is **null in the per-system `results.jsonl` by design** and is filled in "
        "only after the `stateless` baseline has run on the same tasks and `make aggregate` has run; "
        "in-progress / per-system-only runs show it empty."
    )

    # ---- Leaderboard ----
    st.subheader("Leaderboard")
    leaderboard = _load_csv("leaderboard.csv")
    if not leaderboard.empty:
        st.dataframe(leaderboard, use_container_width=True)
    else:
        st.dataframe(
            view.groupby("memory_system")["success"].mean().reset_index(),
            use_container_width=True,
        )

    # ---- Charts ----
    st.subheader("Score by memory system")
    by_sys = view.groupby("memory_system")["score"].mean().reset_index()
    st.plotly_chart(px.bar(by_sys, x="memory_system", y="score"), use_container_width=True)

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

    st.subheader("Progress score by iteration (seed)")
    if "seed" in view.columns:
        trend = view.groupby(["memory_system", "seed"])["progress_score"].mean().reset_index()
        st.plotly_chart(
            px.line(trend, x="seed", y="progress_score", color="memory_system", markers=True),
            use_container_width=True,
        )

    st.subheader("Cost vs success")
    model_col = "model_display" if "model_display" in view.columns else "model"
    cvs = view.groupby(["memory_system", model_col]).agg(
        cost=("estimated_cost_usd", "mean"), success=("success", "mean")
    ).reset_index()
    st.plotly_chart(
        px.scatter(cvs, x="cost", y="success", color="memory_system", symbol=model_col,
                   hover_data=["memory_system", model_col]),
        use_container_width=True,
    )

    st.subheader("Failure categories")
    fails = _load_csv("failure_categories.csv")
    if fails.empty and "failure_category" in view.columns:
        tmp = view[view["failure_category"].notna() & (view["failure_category"] != "")]
        if not tmp.empty:
            fails = tmp.groupby(["memory_system", "failure_category"]).size().rename(
                "count").reset_index()
    if not fails.empty:
        st.plotly_chart(
            px.bar(fails, x="failure_category", y="count", color="memory_system", barmode="group"),
            use_container_width=True,
        )
    else:
        st.caption("No failures recorded for the current selection.")

    # ---- Drill-down ----
    st.subheader("Drill-down")
    drill_cols = [
        c
        for c in [
            "task_id", "memory_system", "model_display", "model_slot", "requested_model",
            "resolved_model", "benchmark", "domain", "seed",
            "score", "progress_score", "retrieved_memory_count", "memory_usage_score",
            "failure_category", "error",
        ]
        if c in view.columns
    ]
    st.dataframe(view[drill_cols], use_container_width=True, height=350)

    # ---- Downloads ----
    st.subheader("Download (filtered view)")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="filtered_results.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Download JSON",
        json.dumps(view.to_dict(orient="records"), default=str).encode("utf-8"),
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
