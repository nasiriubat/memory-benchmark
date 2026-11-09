"""Assemble the email attachment set under ./attachment/.

- Copies the two focused workbooks the email references (subtask trace, per-system).
- Builds a standalone benchmark_statistics.xlsx holding only the statistics sheets
  referenced in the email (extracted from the aggregated stats CSVs).

Run in the project image (mounts /app):
    docker compose -p memory-agent-benchmark run --rm --no-deps -v "$(pwd):/app" \
        --entrypoint python stateless /app/scripts/export_attachments.py
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AGG = ROOT / "results" / "aggregated"
ATT = ROOT / "attachment"

COPY_FILES = ["benchmark_subtask_trace.xlsx", "benchmark_by_system.xlsx"]
STAT_SHEETS = [
    ("pairwise_vs_stateless.csv", "Stats - vs stateless"),
    ("stats_anova.csv", "Stats - ANOVA (2-way)"),
    ("descriptive_stats.csv", "Descriptive stats"),
]
DEFS = [
    ("memory_system / model", "The memory mechanism and the LLM backbone."),
    ("mean_diff", "Metric minus the matched stateless value (positive = memory helped)."),
    ("t_stat, p_value", "Paired comparison vs stateless: t-statistic and p-value (p<0.05 = significant)."),
    ("cohens_d", "Standardised effect size vs stateless (~0.2 small, 0.5 medium, 0.8 large)."),
    ("F, PR(>F)", "ANOVA F-statistic and its p-value (PR(>F)<0.05 = factor significantly affects the metric)."),
    ("eta_squared", "ANOVA effect size: share of variance explained by that factor (0-1)."),
    ("mean / std / count", "Descriptive stats per (memory_system, model) group."),
]


def _style(writer) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    hf, ff = PatternFill("solid", fgColor="1F4E78"), Font(bold=True, color="FFFFFF")
    for ws in writer.book.worksheets:
        ws.freeze_panes = "A2"
        for c in ws[1]:
            c.fill, c.font = hf, ff
            c.alignment = Alignment(vertical="center", wrap_text=True)
        for col in ws.columns:
            longest = max((len(str(c.value)) for c in col if c.value is not None), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max(longest + 2, 12), 60)


def _write_stats(dest: Path) -> bool:
    frames = []
    for csv_name, sheet in STAT_SHEETS:
        p = AGG / csv_name
        if p.exists():
            d = pd.read_csv(p)
            num = d.select_dtypes("float").columns
            d[num] = d[num].round(4)
            frames.append((sheet, d))
    if not frames:
        return False
    # openpyxl's multi-step save flakes in this environment; build in a fresh
    # container-local temp path each attempt, then copy to /app.
    for attempt in range(8):
        tmp = Path(tempfile.mkdtemp()) / dest.name
        try:
            with pd.ExcelWriter(tmp, engine="openpyxl") as w:
                pd.DataFrame(DEFS, columns=["Field", "Meaning"]).to_excel(
                    w, sheet_name="Definitions", index=False)
                for sheet, d in frames:
                    d.to_excel(w, sheet_name=sheet[:31], index=False)  # sheet_name is keyword-only (pandas 3.0)
                _style(w)
            shutil.copyfile(tmp, dest)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.3 * (attempt + 1))
    return False


def main() -> None:
    ATT.mkdir(exist_ok=True)
    for name in COPY_FILES:
        src = ROOT / name
        if src.exists():
            shutil.copyfile(src, ATT / name)
            print("copied", name)
        else:
            print("MISSING", name)
    print("stats:", "written" if _write_stats(ATT / "benchmark_statistics.xlsx") else "FAILED")
    print("attachment/ contents:", sorted(p.name for p in ATT.glob("*")))


if __name__ == "__main__":
    main()
