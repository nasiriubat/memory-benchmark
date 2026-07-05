"""Base memory adapter interface.

Every adapter must implement the same interface and must hard-scope every read
and write to a :class:`MemoryNamespace`. Adapters that depend on an external SDK
or self-hosted backend must fail gracefully: if unavailable, they log the reason,
set ``available = False``, and behave as no-ops (never raising into the runner).
"""

from __future__ import annotations

import abc
import threading
from typing import Any

from benchmark.logging import get_logger
from benchmark.schemas import MemoryNamespace, MemoryRecord, MemoryWriteResult


class MemoryAdapter(abc.ABC):
    """Abstract memory adapter."""

    #: Stable id for this memory system (e.g. "mem0").
    name: str = "base"
    #: Whether the backend/SDK is available for use this run.
    available: bool = True
    #: Reason captured when ``available`` is False.
    unavailable_reason: str | None = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.log = get_logger(f"benchmark.memory.{self.name}")
        # Backend was unavailable at startup -> the WHOLE run uses the local
        # fallback (sticky). Set via ``self._unavailable_fallback = True``.
        self._unavailable_fallback = False
        # Per-task fallback marker. ``uses_local_fallback`` is recorded on every
        # result row, so it must reflect whether THAT task's memory ops fell back
        # (not a sticky global flag). Tracked thread-local because tasks run
        # concurrently under TASK_PARALLELISM (one task per worker thread), and
        # cleared at the start of each task by the runner.
        self._task_fb = threading.local()

    @property
    def uses_local_fallback(self) -> bool:
        """True when this task's (or the whole run's) memory ops used the local
        fallback store rather than the real backend."""
        return self._unavailable_fallback or getattr(self._task_fb, "hit", False)

    @uses_local_fallback.setter
    def uses_local_fallback(self, value: bool) -> None:
        # Op-time fallbacks set this True and mark the CURRENT task only.
        self._task_fb.hit = bool(value)

    def clear_task_fallback(self) -> None:
        """Reset the per-task fallback marker; called by the runner before each task."""
        self._task_fb.hit = False

    @abc.abstractmethod
    def setup(self, namespace: MemoryNamespace) -> None:
        """Prepare any per-namespace state."""

    @abc.abstractmethod
    def reset(self, namespace: MemoryNamespace) -> None:
        """Clear all memory for the given namespace (isolation boundary)."""

    @abc.abstractmethod
    def retrieve(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[MemoryRecord]:
        """Return up to ``top_k`` records relevant to ``query`` within namespace."""

    @abc.abstractmethod
    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        """Write a record within its namespace."""

    @abc.abstractmethod
    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        """Update an existing record within its namespace."""

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        """Best-effort size of the namespace's store in bytes. Default 0."""
        return 0

    def close(self) -> None:  # pragma: no cover - optional hook
        """Release resources. Default no-op."""
        return None

    # ------------------------------------------------------------- internals
    def _require_namespace(self, metadata: dict[str, Any]) -> MemoryNamespace:
        ns = metadata.get("namespace")
        if not isinstance(ns, MemoryNamespace):
            raise ValueError(
                "Memory operation requires a 'namespace' MemoryNamespace in metadata "
                "to guarantee isolation."
            )
        return ns
