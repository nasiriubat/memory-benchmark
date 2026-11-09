"""Generate paper-ready figures (PNG) from the aggregated results.

Run inside the project image (matplotlib pip-installed on the fly):
    docker compose -p memory-agent-benchmark run --rm --no-deps -v "$(pwd):/app" \
        --entrypoint sh stateless -c "pip -q install matplotlib && python /app/scripts/make_figures.py"
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "results" / "aggregated"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Clean display names for models (accepts resolved ids OR slot names).
LABEL = {
    "gpt-5.5": "gpt-5.5",
    "openai_frontier": "gpt-5.5",
    "google/gemini-3.5-flash": "Gemini 3.5 Flash",
    "google_fast": "Gemini 3.5 Flash",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "anthropic_opus": "Claude Opus 4.8",
}
# Consistent colour + order per model.
ORDER = ["gpt-5.5", "Gemini 3.5 Flash", "Claude Opus 4.8"]
COLOR = {"gpt-5.5": "#4C78A8", "Gemini 3.5 Flash": "#F58518", "Claude Opus 4.8": "#54A24B"}
SYS_ORDER = ["stateless", "hindsight", "mem0", "acontext"]


def _lab(x: str) -> str:
    return LABEL.get(str(x), str(x))


def _grouped_bar(df, value, title, ylabel, fname, drop_stateless=False):
    d = df.copy()
    d["mlabel"] = d["model"].map(_lab)
    if drop_stateless:
        d = d[d["memory_system"] != "stateless"]
    systems = [s for s in SYS_ORDER if s in d["memory_system"].unique()]
    models = [m for m in ORDER if m in d["mlabel"].unique()]
    piv = d.pivot_table(index="memory_system", columns="mlabel", values=value).reindex(systems)
    x = range(len(systems))
    n = len(models)
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(models):
        vals = piv[m].values if m in piv.columns else [0] * len(systems)
        ax.bar([xi + (i - (n - 1) / 2) * w for xi in x], vals, w, label=m, color=COLOR.get(m))
    ax.set_xticks(list(x))
    ax.set_xticklabels(systems)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(title="LLM backbone")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)
    print("wrote", FIG / fname)


def _learning_curve(trend, fname):
    t = trend.copy()
    t["mlabel"] = t["model"].map(_lab)
    models = [m for m in ORDER if m in t["mlabel"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), sharey=True)
    if len(models) == 1:
        axes = [axes]
    sys_colors = {"stateless": "#999999", "hindsight": "#4C78A8", "mem0": "#F58518", "acontext": "#54A24B"}
    for ax, m in zip(axes, models):
        sub = t[t["mlabel"] == m]
        for s in [s for s in SYS_ORDER if s in sub["memory_system"].unique()]:
            ss = sub[sub["memory_system"] == s].sort_values("subtask_index")
            ax.plot(ss["subtask_index"], ss["mean_score"], marker="o", label=s, color=sys_colors.get(s))
        ax.set_title(m)
        ax.set_xlabel("subtask position")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean subtask score")
    axes[-1].legend(title="memory system", fontsize=8)
    fig.suptitle("Within-task learning: mean score by subtask position")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)
    print("wrote", FIG / fname)


def main():
    mm = pd.read_csv(AGG / "summary_by_memory_model.csv")
    _grouped_bar(mm, "progress_score",
                 "Progress score by memory system and LLM backbone",
                 "progress score (correct subtasks / total)",
                 "fig1_progress_by_system_model.png")
    _grouped_bar(mm, "memory_utility_gain_vs_stateless",
                 "Memory-utility gain vs stateless, by system and backbone",
                 "gain vs stateless (Δ progress score)",
                 "fig2_gain_by_system_model.png", drop_stateless=True)
    trend = pd.read_csv(AGG / "subtask_trend.csv")
    if not trend.empty and "model" in trend.columns:
        _learning_curve(trend, "fig3_learning_curve.png")


if __name__ == "__main__":
    main()
