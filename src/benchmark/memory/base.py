"""Base memory adapter interface.

Every adapter must implement the same interface and must hard-scope every read
and write to a :class:`MemoryNamespace`. Adapters that depend on an external SDK
or self-hosted backend must fail gracefully: if unavailable, they log the reason,
set ``available = False``, and behave as no-ops (never raising into the runner).
"""

from __future__ import annotations

import abc
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
    #: True when memory operations are (or have been) served by the local
    #: namespace-isolated fallback store rather than the real backend SDK. This
    #: is recorded on every result so a run that silently degraded to the local
    #: store is never mistaken for a measurement of the real memory system.
    uses_local_fallback: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.log = get_logger(f"benchmark.memory.{self.name}")

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
