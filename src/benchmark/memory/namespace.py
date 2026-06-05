"""Namespace helpers enforcing strict memory isolation.

The :class:`MemoryNamespace` schema lives in ``benchmark.schemas``; this module
provides constructors and path helpers used by the runner and storage layers.
"""

from __future__ import annotations

from pathlib import Path

from benchmark.schemas import MemoryNamespace


def make_namespace(
    *,
    run_id: str,
    benchmark: str,
    domain: str,
    memory_system: str,
    model: str,
    task_id: str,
    seed: int,
) -> MemoryNamespace:
    """Build a fully-specified namespace. All seven axes are required."""
    return MemoryNamespace(
        run_id=run_id,
        benchmark=benchmark,
        domain=domain,
        memory_system=memory_system,
        model=model,
        task_id=task_id,
        seed=seed,
    )


def raw_log_path(results_root: Path, ns: MemoryNamespace) -> Path:
    """Path to the per-namespace raw JSONL events file.

    Layout:
        results/raw/{run_id}/{memory_system}/{model}/{benchmark}/{domain}/events.jsonl
    """
    return (
        results_root
        / "raw"
        / _safe(ns.run_id)
        / _safe(ns.memory_system)
        / _safe(ns.model)
        / _safe(ns.benchmark)
        / _safe(ns.domain)
        / "events.jsonl"
    )


def per_system_dir(results_root: Path, memory_system: str, run_id: str) -> Path:
    """Dedicated per-memory-system results folder.

    Layout:
        results/by_memory_system/{memory_system}/{run_id}/
    """
    return results_root / "by_memory_system" / _safe(memory_system) / _safe(run_id)


def _safe(value: str) -> str:
    return str(value).replace("/", "_").replace("..", "_").strip()
