"""Export a SUBTASK-LEVEL trace from the raw event logs.

This surfaces the within-task memory activity that the aggregated sheets hide:
per (memory_system, model, subtask position) — how much memory was retrieved,
whether memory was written, and the per-subtask correctness. It is WITHIN-task
only (memory is isolated per task); it does NOT show cross-task transfer or which
specific memory item was reused (memory ids/text are not logged).

Run inside the project image:
    docker compose -p memory-agent-benchmark run --rm --no-deps -v "$(pwd):/app" \
        --entrypoint python stateless /app/scripts/export_subtask_trace.py
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "benchmark_subtask_trace.xlsx"

# The three real campaigns (one per LLM backbone), plus the 2026-07-23
# replication of stateless x Claude Opus 4.8 whose complete event log replaces
# the original campaign's partial one (archived under
# results/EXCLUDED_partial_events_stateless_claudeopus48/).
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
MODEL_ORDER = ["gpt-5.5", "Gemini 3.5 Flash", "Claude Opus 4.8"]
SYS_ORDER = ["stateless", "hindsight", "mem0", "acontext"]

DEFS = [
    ("memory_system", "Memory mechanism under test; 'stateless' is the no-memory baseline."),
    ("model", "LLM backbone (agent)."),
    ("subtask_index", "Position of the question within a task (0 = first, executed in order)."),
    ("n", "Number of (task, seed) observations at this subtask position."),
    ("retrieved (avg)", "Average number of memory items retrieved at this subtask."),
    ("written (rate)", "Fraction of subtasks after which the system wrote a memory."),
    ("mean score", "Mean LLM-judge correctness (0-1) at this subtask position."),
    ("correct (rate)", "Fraction of subtasks judged fully correct at this position."),
    ("SCOPE", "WITHIN-task activity across subtasks q0..qN. Memory is isolated per task, so this does "
              "NOT show cross-task transfer, sessions, or which specific memory item was reused "
              "(memory ids/text are not logged)."),
]


def _read(path: Path) -> str:
    for attempt in range(20):
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    # A silently-skipped file drops an entire (system, model) cell from the
    # trace, so make the loss loud instead of producing a quietly missing cell.
    print(f"WARNING: could not read {path} after 20 attempts; its rows are missing from the trace")
    return ""


def _load() -> pd.DataFrame:
    rows: list[dict] = []
    if not RAW.exists():
        return pd.DataFrame()
    for ev in RAW.glob("*/*/*/**/events.jsonl"):
        parts = ev.relative_to(RAW).parts  # (run_id, system, slot, benchmark, domain, ...)
        if len(parts) < 3 or parts[0] not in RUN_IDS:
            continue
        system, slot = parts[1], parts[2]
        cur: dict[tuple, dict] = {}
        for line in _read(ev).splitlines():
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
            key = (e.get("task_id"), int(m.group(1)), e.get("seed"))
            d = cur.setdefault(key, {"retrieved": None, "written": None, "score": None, "correct": None})
            p = e.get("payload") or {}
            et = e.get("event_type")
            if et == "memory_retrieved":
                d["retrieved"] = p.get("count")
            elif et == "memory_written":
                d["written"] = p.get("written")
            elif et == "evaluator_score":
                d["score"] = p.get("score")
                d["correct"] = p.get("correct")
        for (tid, idx, seed), d in cur.items():
            rows.append({
                "memory_system": system, "model": SLOT_LABEL.get(slot, slot),
                "task_id": tid, "subtask_index": idx, "seed": seed,
                "retrieved": d["retrieved"], "written": d["written"],
                "score": d["score"], "correct": d["correct"],
            })
    return pd.DataFrame(rows)


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__m"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    df["__s"] = df["memory_system"].map({s: i for i, s in enumerate(SYS_ORDER)}).fillna(99)
    return df


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
            ws.column_dimensions[col[0].column_letter].width = min(max(longest + 2, 12), 70)


def main() -> None:
    df = _load()
    if df.empty:
        print("no raw events found")
        return
    df["correct_num"] = df["correct"].map(lambda x: 1.0 if x is True else (0.0 if x is False else None))

    g = df.groupby(["memory_system", "model", "subtask_index"])
    prog = g.agg(
        n=("score", "size"),
        **{"retrieved (avg)": ("retrieved", "mean"), "written (rate)": ("written", "mean"),
           "mean score": ("score", "mean"), "correct (rate)": ("correct_num", "mean")},
    ).reset_index()
    prog = _order(prog).sort_values(["__s", "__m", "subtask_index"]).drop(columns=["__m", "__s"])
    for c in ["retrieved (avg)", "written (rate)", "mean score", "correct (rate)"]:
        prog[c] = prog[c].round(3)

    gd = df.groupby(["memory_system", "model", "task_id", "subtask_index"])
    det = gd.agg(retrieved=("retrieved", "mean"), written=("written", "mean"),
                 mean_score=("score", "mean"), correct_rate=("correct_num", "mean")).reset_index()
    det = _order(det).sort_values(["__s", "__m", "task_id", "subtask_index"]).drop(columns=["__m", "__s"])
    for c in ["retrieved", "written", "mean_score", "correct_rate"]:
        det[c] = det[c].round(3)

    # The openpyxl multi-step save flakes intermittently in this environment
    # ("At least one sheet must be visible"). Build the workbook in a FRESH
    # container-local temp path each attempt (reusing the path corrupts state),
    # then copy the finished file to /app.
    docs = ROOT / "docs" / OUT.name
    docs.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(8):
        tmp = Path(tempfile.mkdtemp()) / OUT.name
        try:
            with pd.ExcelWriter(tmp, engine="openpyxl") as w:
                pd.DataFrame(DEFS, columns=["Field", "Meaning"]).to_excel(
                    w, sheet_name="About & definitions", index=False)
                prog.to_excel(w, sheet_name="Subtask progression", index=False)
                det.to_excel(w, sheet_name="By task (detail)", index=False)
                _style(w)
            shutil.copyfile(tmp, OUT)
            shutil.copyfile(tmp, docs)
            print(f"Wrote {OUT} (+ {docs}); progression {len(prog)} rows, "
                  f"detail {len(det)} rows (attempt {attempt + 1}).")
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"could not write workbook after retries: {last}")


if __name__ == "__main__":
    main()
