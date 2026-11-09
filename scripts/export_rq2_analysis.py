"""Export a single RQ2-ready workbook: RQ2_subtask_analysis.xlsx (+ figures).

RQ2: How do agent performance and memory-retrieval patterns evolve across
dependent subtasks?

Everything is computed from ONE source — the canonical raw event logs under
results/raw/ (the stateless x Claude cell comes from the 2026-07-23 replication
run) — so the sheets and figures are internally consistent:

  Sheet "Learning curve"      mean judge score by subtask position x system
                              (backbones pooled, and per backbone)
  Sheet "Gap vs stateless"    memory-system score minus stateless score,
                              matched per (model, task, position), seed-pooled
  Sheet "Retrieval patterns"  avg retrieved records + write rate by position
  Sheet "Trend stats"         OLS slope of score on position, early-vs-late
                              contrast (Welch t-test) per system
  Sheet "Figures"             the two figures embedded as images
  Sheet "Notes"               definitions and caveats

Figure files are also written to docs/figures/rq2_*.{png,pdf}.

Run:  .venv/bin/python3 scripts/export_rq2_analysis.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
FIG_DIR = ROOT / "docs" / "figures"
OUT = ROOT / "RQ2_subtask_analysis.xlsx"

RUN_IDS = {
    "benchmark_20260626T142817Z",
    "benchmark_gemini35flash",
    "benchmark_claudeopus48",
    "benchmark_claudeopus48_stateless_rerun",
}
SLOT_LABEL = {
    "openai_frontier": "gpt-5.5",
    "google_fast": "Gemini 3.5 Flash",
    "anthropic_opus": "Claude Opus 4.8",
}
SYS_ORDER = ["stateless", "mem0", "hindsight", "acontext"]
MODEL_ORDER = ["gpt-5.5", "Claude Opus 4.8", "Gemini 3.5 Flash"]

# Figure palette (validated categorical slots on white; baseline is neutral
# gray + dashed so identity never rides on color alone).
COLORS = {
    "stateless": "#898781",
    "mem0": "#2a78d6",
    "hindsight": "#eb6834",
    "acontext": "#1baf7a",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def load_events() -> pd.DataFrame:
    """One row per logged subtask execution (seed-pooled; events do not
    record the seed, so the 3 seeds of a (task, position) appear as up to 3
    rows)."""
    rows = []
    for ev in RAW.glob("*/*/*/*/*/events.jsonl"):
        parts = ev.relative_to(RAW).parts  # run_id, system, slot, bench, domain
        if parts[0] not in RUN_IDS:
            continue
        system, slot = parts[1], parts[2]
        # per (task, subtask): retrieval count arrives before the score event,
        # so carry the latest retrieval/write value until the score closes it
        pending: dict[tuple, dict] = {}
        for line in ev.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            sid = e.get("subtask_id")
            if sid is None:
                continue
            m = re.search(r"(\d+)", str(sid))
            if not m:
                continue
            key = (e.get("task_id"), int(m.group(1)))
            p = e.get("payload") or {}
            et = e.get("event_type")
            if et == "memory_retrieved":
                pending.setdefault(key, {})["retrieved"] = p.get("count")
            elif et == "memory_written":
                pending.setdefault(key, {})["written"] = p.get("written")
            elif et == "evaluator_score":
                d = pending.pop(key, {})
                rows.append({
                    "memory_system": system,
                    "model": SLOT_LABEL.get(slot, slot),
                    "task_id": str(e.get("task_id")),
                    "subtask_index": key[1],
                    "score": p.get("score"),
                    "correct": bool(p.get("correct")),
                    "retrieved": d.get("retrieved"),
                    "written": d.get("written"),
                })
    df = pd.DataFrame(rows)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df.dropna(subset=["score"])


def learning_curve(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = (df.groupby(["memory_system", "subtask_index"])["score"]
                .agg(mean_score="mean", sem="sem", n="size").reset_index())
    per_model = (df.groupby(["memory_system", "model", "subtask_index"])["score"]
                   .agg(mean_score="mean", sem="sem", n="size").reset_index())
    return _order(pooled), _order(per_model)


def gap_vs_stateless(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Memory-system minus stateless score, matched per (model, task,
    position); seeds pooled by the per-cell mean first."""
    cell = (df.groupby(["memory_system", "model", "task_id", "subtask_index"])["score"]
              .mean().reset_index())
    base = cell[cell.memory_system == "stateless"].rename(
        columns={"score": "stateless_score"}).drop(columns=["memory_system"])
    mem = cell[cell.memory_system != "stateless"]
    j = mem.merge(base, on=["model", "task_id", "subtask_index"], how="inner")
    j["gap"] = j["score"] - j["stateless_score"]
    pooled = (j.groupby(["memory_system", "subtask_index"])["gap"]
                .agg(mean_gap="mean", sem="sem", n_pairs="size").reset_index())
    per_model = (j.groupby(["memory_system", "model", "subtask_index"])["gap"]
                   .agg(mean_gap="mean", sem="sem", n_pairs="size").reset_index())
    return _order(pooled), _order(per_model)


def retrieval_patterns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["retrieved"] = pd.to_numeric(d["retrieved"], errors="coerce")
    d["written"] = pd.to_numeric(d["written"], errors="coerce")
    out = (d.groupby(["memory_system", "subtask_index"])
             .agg(**{"retrieved (avg)": ("retrieved", "mean"),
                     "write rate": ("written", "mean"),
                     "n": ("score", "size")}).reset_index())
    return _order(out)


def trend_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all backbones", df)] + [
        (m, df[df.model == m]) for m in MODEL_ORDER]
    for label, d in groups:
        for system in SYS_ORDER:
            s = d[d.memory_system == system]
            if s.empty:
                continue
            lr = stats.linregress(s["subtask_index"], s["score"])
            early = s[s.subtask_index <= 1]["score"]
            late = s[s.subtask_index >= 4]["score"]
            tt = stats.ttest_ind(late, early, equal_var=False)
            rows.append({
                "scope": label, "memory_system": system,
                "n_obs": len(s),
                "ols_slope_per_position": round(lr.slope, 4),
                "slope_p_value": round(lr.pvalue, 4),
                "early_mean_q0_q1": round(early.mean(), 4),
                "late_mean_q4_q5": round(late.mean(), 4),
                "late_minus_early": round(late.mean() - early.mean(), 4),
                "welch_t_p_value": round(tt.pvalue, 4),
            })
    return pd.DataFrame(rows)


def _order(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["__s"] = d["memory_system"].map({s: i for i, s in enumerate(SYS_ORDER)})
    cols = ["__s"] + (["model"] if "model" in d.columns else []) + ["subtask_index"]
    return d.sort_values(cols).drop(columns="__s").reset_index(drop=True)


# ------------------------------------------------------------------ figures
def _style_axes(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"q{i}" for i in range(6)])


def _dodge(labels: list[tuple[str, float]], min_gap: float) -> dict[str, float]:
    """Push label y-positions apart until adjacent ones are >= min_gap."""
    ordered = sorted(labels, key=lambda kv: kv[1])
    ys = [y for _, y in ordered]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    return {name: y for (name, _), y in zip(ordered, ys)}


def fig_learning_curve(pooled: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    ends: list[tuple[str, float]] = []
    for system in SYS_ORDER:
        d = pooled[pooled.memory_system == system]
        dash = (0, (4, 2)) if system == "stateless" else "solid"
        ax.errorbar(d["subtask_index"], d["mean_score"], yerr=d["sem"],
                    color=COLORS[system], linewidth=2, linestyle=dash,
                    marker="o", markersize=4.5, capsize=2, elinewidth=1)
        last = d[d.subtask_index == d.subtask_index.max()].iloc[0]
        ends.append((system, float(last["mean_score"])))
    label_y = _dodge(ends, min_gap=0.024)
    x_end = pooled.subtask_index.max()
    for system, y in label_y.items():
        ax.annotate(system, (x_end, y), xytext=(8, 0),
                    textcoords="offset points", color=COLORS[system],
                    fontsize=9, fontweight="bold", va="center")
    _style_axes(ax)
    ax.set_xlim(-0.2, 6.6)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("Subtask position within task", color=INK, fontsize=10)
    ax.set_ylabel("Mean judge score (0–1)", color=INK, fontsize=10)
    ax.set_title("Within-task learning curve by memory system\n"
                 "(mean ± s.e.m., pooled over 3 LLM backbones, 40 tasks × 3 seeds)",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, format="png", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", facecolor="white")
    plt.close(fig)


def fig_gap(pooled_gap: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
    ax.axhline(0, color="#898781", linewidth=1.2, linestyle=(0, (4, 2)))
    ax.annotate("stateless baseline", (0.02, 0.005), xycoords=("axes fraction", "data"),
                color="#898781", fontsize=8.5, va="bottom")
    for system in ["mem0", "hindsight", "acontext"]:
        d = pooled_gap[pooled_gap.memory_system == system]
        ax.errorbar(d["subtask_index"], d["mean_gap"], yerr=d["sem"],
                    color=COLORS[system], linewidth=2, marker="o",
                    markersize=4.5, capsize=2, elinewidth=1)
        last = d[d.subtask_index == d.subtask_index.max()].iloc[0]
        ax.annotate(system, (last["subtask_index"], last["mean_gap"]),
                    xytext=(8, 0), textcoords="offset points",
                    color=COLORS[system], fontsize=9, fontweight="bold",
                    va="center")
    _style_axes(ax)
    ax.set_xlim(-0.2, 6.6)
    ax.set_xlabel("Subtask position within task", color=INK, fontsize=10)
    ax.set_ylabel("Score gap vs stateless (matched)", color=INK, fontsize=10)
    ax.set_title("Memory benefit across dependent subtasks\n"
                 "(memory − stateless, matched per model × task × position; mean ± s.e.m.)",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, format="png", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ workbook
NOTES = [
    ("RQ2", "How do agent performance and memory-retrieval patterns evolve across "
            "dependent subtasks?"),
    ("Source", "Computed from the canonical raw event logs (results/raw/**/events.jsonl). "
               "The stateless x Claude Opus 4.8 cell uses the complete 2026-07-23 replication "
               "run (benchmark_claudeopus48_stateless_rerun); the original partial log is "
               "archived under results/EXCLUDED_partial_events_stateless_claudeopus48/."),
    ("Observations", "One row per executed subtask with a judge score. Every cell has the "
                     "full 720 observations (40 tasks x 3 seeds x 6 subtasks). The 21 mem0 x "
                     "Claude events lost to a file-locking issue (tasks 61, 75, 130) were "
                     "recovered by a 2026-07-26 patch re-run of those three tasks (identical "
                     "configuration; see results/RERUN_mem0_claudeopus48_taskrows/README.txt)."),
    ("Seed pooling", "Raw events do not record the seed (logged as 0), so the 3 seeds are "
                     "pooled. Gap matching is therefore per (model, task, position) on "
                     "seed-pooled means, not per individual seed."),
    ("mean score", "Mean LLM-judge correctness score (0-1) at a subtask position."),
    ("gap", "Memory-system score minus stateless score for the SAME model, task, and "
            "position (paired difference), averaged over pairs."),
    ("retrieved (avg)", "Average number of memory records returned at that position "
                        "(top-k = 3; the store is empty at q0, so retrieval starts near 0)."),
    ("write rate", "Fraction of subtasks after which a memory record was written "
                   "(stateless: 0 by design)."),
    ("Trend stats", "OLS slope of score on position over all observations in scope; "
                    "early-vs-late compares q0-q1 to q4-q5 with a Welch t-test."),
    ("Scope", "WITHIN-task evolution only. Memory is reset per task and seed; this data "
              "does not measure cross-task transfer."),
]


def main() -> None:
    df = load_events()
    print(f"loaded {len(df)} scored subtask observations "
          f"({df.memory_system.nunique()} systems x {df.model.nunique()} models)")

    lc, lc_model = learning_curve(df)
    gap, gap_model = gap_vs_stateless(df)
    ret = retrieval_patterns(df)
    ts = trend_stats(df)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f1 = FIG_DIR / "rq2_learning_curve.png"
    f2 = FIG_DIR / "rq2_gap_by_position.png"
    fig_learning_curve(lc, f1)
    fig_gap(gap, f2)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        pd.DataFrame(NOTES, columns=["Item", "Meaning"]).to_excel(
            w, sheet_name="Notes", index=False)
        lc.round(4).to_excel(w, sheet_name="Learning curve", index=False,
                             startrow=0)
        # wide view beside the long table
        wide = lc.pivot(index="subtask_index", columns="memory_system",
                        values="mean_score").reindex(columns=SYS_ORDER).round(4)
        wide.to_excel(w, sheet_name="Learning curve", startrow=0, startcol=7)
        lc_model.round(4).to_excel(w, sheet_name="Learning curve (per model)",
                                   index=False)
        gap.round(4).to_excel(w, sheet_name="Gap vs stateless", index=False)
        gap_model.round(4).to_excel(w, sheet_name="Gap vs stateless (per model)",
                                    index=False)
        ret.round(4).to_excel(w, sheet_name="Retrieval patterns", index=False)
        ts.to_excel(w, sheet_name="Trend stats", index=False)

        from openpyxl.drawing.image import Image as XLImage
        wsf = w.book.create_sheet("Figures")
        wsf["A1"] = "Figure R2a — Within-task learning curve (docs/figures/rq2_learning_curve.png/pdf)"
        img1 = XLImage(str(f1)); img1.width, img1.height = 768, 480
        wsf.add_image(img1, "A3")
        wsf["A30"] = "Figure R2b — Memory benefit vs stateless by position (docs/figures/rq2_gap_by_position.png/pdf)"
        img2 = XLImage(str(f2)); img2.width, img2.height = 768, 480
        wsf.add_image(img2, "A32")

        from openpyxl.styles import Font, PatternFill
        hf, ff = PatternFill("solid", fgColor="1F4E78"), Font(bold=True, color="FFFFFF")
        for ws in w.book.worksheets:
            if ws.title == "Figures":
                continue
            ws.freeze_panes = "A2"
            for c in ws[1]:
                if c.value is not None:
                    c.fill, c.font = hf, ff
            for col in ws.columns:
                longest = max((len(str(c.value)) for c in col if c.value is not None),
                              default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max(longest + 2, 11), 60)

    for dst in (ROOT / "docs" / OUT.name, ROOT / "attachment" / OUT.name):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT, dst)
    print(f"Wrote {OUT} (+ docs/, attachment/) and figures in {FIG_DIR}/rq2_*.png|pdf")


if __name__ == "__main__":
    main()
