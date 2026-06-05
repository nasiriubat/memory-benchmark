"""A simple, namespace-isolated in-process memory store.

Used as the local self-hosted fallback engine for stub adapters when their real
SDK/backend is unavailable, so the benchmark still runs and memory utility can
be measured against the stateless baseline. Retrieval uses token-overlap
similarity (deterministic, no external dependency).

Isolation is enforced by keying everything on the namespace key string; one
namespace can never read another's entries.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone

from benchmark.schemas import MemoryNamespace, MemoryRecord, MemoryWriteResult
from benchmark.utils import token_set


class LocalMemoryStore:
    """Thread-safe, namespace-partitioned record store."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, MemoryRecord]] = {}
        self._lock = threading.Lock()

    def reset(self, namespace: MemoryNamespace) -> None:
        with self._lock:
            self._data.pop(namespace.key(), None)

    def write(self, record: MemoryRecord, namespace: MemoryNamespace) -> MemoryWriteResult:
        with self._lock:
            bucket = self._data.setdefault(namespace.key(), {})
            mem_id = record.id or uuid.uuid4().hex
            record.id = mem_id
            record.namespace = namespace
            record.updated_at = datetime.now(timezone.utc)
            bucket[mem_id] = record
        return MemoryWriteResult(success=True, memory_id=mem_id)

    def update(
        self, memory_id: str, record: MemoryRecord, namespace: MemoryNamespace
    ) -> MemoryWriteResult:
        with self._lock:
            bucket = self._data.get(namespace.key(), {})
            if memory_id not in bucket:
                return MemoryWriteResult(
                    success=False, memory_id=memory_id, error="memory_id not found"
                )
            record.id = memory_id
            record.namespace = namespace
            record.updated_at = datetime.now(timezone.utc)
            bucket[memory_id] = record
        return MemoryWriteResult(success=True, memory_id=memory_id)

    def retrieve(
        self, query: str, top_k: int, namespace: MemoryNamespace
    ) -> list[MemoryRecord]:
        with self._lock:
            bucket = list(self._data.get(namespace.key(), {}).values())
        if not bucket:
            return []
        query_tokens = token_set(query)
        scored: list[tuple[float, MemoryRecord]] = []
        for record in bucket:
            rec_tokens = token_set(record.text)
            if not rec_tokens:
                score = 0.0
            else:
                overlap = len(query_tokens & rec_tokens)
                union = len(query_tokens | rec_tokens) or 1
                score = overlap / union
            copy = record.model_copy()
            copy.score = round(score, 4)
            scored.append((score, copy))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [rec for _, rec in scored[: max(top_k, 0)]]

    def size_bytes(self, namespace: MemoryNamespace) -> int:
        with self._lock:
            bucket = self._data.get(namespace.key(), {})
            return sum(len(json.dumps(r.model_dump(), default=str)) for r in bucket.values())
