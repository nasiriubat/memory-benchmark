"""Run progress tracking for the dashboard and operators.

Writes ``results/run_status.json`` so the UI can show done / running / pending
memory systems without parsing Docker state.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from benchmark.utils import results_root

# Canonical run order (stateless first — baseline for memory-utility gain).
MEMORY_SYSTEMS_ORDER: list[str] = [
    "stateless",
    "hindsight",
    "acontext",
    "mem0",
    "zep",
]

Status = Literal["pending", "running", "done", "failed"]


def _status_path(root: Path | None = None) -> Path:
    return (root or results_root()) / "run_status.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _empty_status(run_id: str = "") -> dict[str, Any]:
    return {
        "run_id": run_id,
        "updated_at": _utcnow(),
        "current_system": None,
        "systems": {
            name: {"status": "pending", "tasks_completed": 0, "run_folder": None, "finished_at": None}
            for name in MEMORY_SYSTEMS_ORDER
        },
    }


def load_status(root: Path | None = None) -> dict[str, Any]:
    path = _status_path(root)
    if not path.exists():
        return _empty_status()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_status()


def save_status(status: dict[str, Any], root: Path | None = None) -> None:
    status["updated_at"] = _utcnow()
    path = _status_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def init_run(run_id: str, root: Path | None = None) -> dict[str, Any]:
    status = _empty_status(run_id)
    save_status(status, root)
    return status


def set_running(run_id: str, memory_system: str, root: Path | None = None) -> None:
    status = load_status(root)
    if not status.get("run_id"):
        status["run_id"] = run_id
    elif status["run_id"] != run_id:
        status = _empty_status(run_id)
    status["current_system"] = memory_system
    for name in MEMORY_SYSTEMS_ORDER:
        if name == memory_system:
            status["systems"][name]["status"] = "running"
            status["systems"][name]["run_folder"] = f"by_memory_system/{memory_system}/{run_id}"
        elif status["systems"][name]["status"] == "running":
            status["systems"][name]["status"] = "pending"
    save_status(status, root)


def set_done(
    run_id: str,
    memory_system: str,
    *,
    tasks_completed: int | None = None,
    root: Path | None = None,
) -> None:
    status = load_status(root)
    if tasks_completed is None:
        tasks_completed = _count_tasks(root or results_root(), memory_system, run_id)
    status["systems"][memory_system] = {
        "status": "done",
        "tasks_completed": tasks_completed,
        "run_folder": f"by_memory_system/{memory_system}/{run_id}",
        "finished_at": _utcnow(),
    }
    if status.get("current_system") == memory_system:
        status["current_system"] = None
    save_status(status, root)


def set_failed(run_id: str, memory_system: str, error: str = "", root: Path | None = None) -> None:
    status = load_status(root)
    status["systems"][memory_system]["status"] = "failed"
    status["systems"][memory_system]["error"] = error[:500]
    if status.get("current_system") == memory_system:
        status["current_system"] = None
    save_status(status, root)


def sync_from_disk(run_id: str | None = None, root: Path | None = None) -> dict[str, Any]:
    """Reconcile status with on-disk result folders (e.g. after manual runs)."""
    root = root or results_root()
    status = load_status(root)
    rid = run_id or status.get("run_id") or ""
    if not rid:
        # Pick latest run id from any system folder.
        for sys in MEMORY_SYSTEMS_ORDER:
            dirs = sorted((root / "by_memory_system" / sys).glob("benchmark_*"), reverse=True)
            if dirs:
                rid = dirs[0].name
                break
    if rid and not status.get("run_id"):
        status = _empty_status(rid)

    status["run_id"] = rid
    for name in MEMORY_SYSTEMS_ORDER:
        jsonl = root / "by_memory_system" / name / rid / "results.jsonl"
        if jsonl.exists():
            n = _count_tasks(root, name, rid)
            entry = status["systems"][name]
            if entry.get("status") != "running":
                entry["status"] = "done"
            entry["tasks_completed"] = n
            entry["run_folder"] = f"by_memory_system/{name}/{rid}"
    save_status(status, root)
    return status


def _count_tasks(root: Path, memory_system: str, run_id: str) -> int:
    jsonl = root / "by_memory_system" / memory_system / run_id / "results.jsonl"
    if not jsonl.exists():
        return 0
    return sum(1 for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip())


def summary_for_ui(root: Path | None = None) -> pd.DataFrame:
    """Table rows for Streamlit progress panel."""
    status = sync_from_disk(root=root)
    rows = []
    for i, name in enumerate(MEMORY_SYSTEMS_ORDER, start=1):
        entry = status["systems"].get(name, {})
        st_label: str = entry.get("status", "pending")
        if status.get("current_system") == name and st_label != "done":
            st_label = "running"
        rows.append(
            {
                "step": i,
                "memory_system": name,
                "status": st_label,
                "tasks_completed": entry.get("tasks_completed", 0),
                "run_id": status.get("run_id", ""),
            }
        )
    return pd.DataFrame(rows)
