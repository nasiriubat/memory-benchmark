"""Stateless memory adapter (required baseline).

Retrieves nothing and writes nothing. Attempted retrieve/write calls are still
logged as no-ops (the runner logs ``memory_retrieved``/``memory_written``
events) so the stateless baseline is observable. Memory-utility gain for every
other system is computed relative to this baseline, so it must always work.
"""

from __future__ import annotations

from typing import Any

from benchmark.memory.base import MemoryAdapter
from benchmark.schemas import MemoryNamespace, MemoryRecord, MemoryWriteResult


class StatelessAdapter(MemoryAdapter):
    name = "stateless"
    available = True

    def setup(self, namespace: MemoryNamespace) -> None:
        return None

    def reset(self, namespace: MemoryNamespace) -> None:
        return None

    def retrieve(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[MemoryRecord]:
        # No-op: stateless retrieves nothing.
        return []

    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        # No-op: stateless writes nothing, but reports the attempt as handled.
        return MemoryWriteResult(success=True, memory_id=None, error=None)

    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        return MemoryWriteResult(success=True, memory_id=None, error=None)

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        return 0
