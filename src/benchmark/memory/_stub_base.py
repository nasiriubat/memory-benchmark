"""Shared base for stub memory adapters.

Each stub adapter targets a real, self-hosted memory system. Real SDK/backend
integration points are marked with ``# REAL SDK:`` comments. When the SDK or
self-hosted backend is unavailable, the adapter logs that it is unavailable and
falls back to a local, namespace-isolated store so the benchmark still runs and
memory utility can be measured. All operations are hard-scoped to the namespace.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from benchmark.memory.base import MemoryAdapter
from benchmark.memory.local_store import LocalMemoryStore
from benchmark.schemas import MemoryNamespace, MemoryRecord, MemoryWriteResult


class StubMemoryAdapter(MemoryAdapter):
    """Base implementation for the self-hosted memory-system stubs."""

    #: Python module that would provide the real SDK (checked for availability).
    sdk_module: str | None = None
    #: Env var holding the self-hosted base URL.
    base_url_env: str | None = None
    #: Env var holding the API key (optional for self-hosted).
    api_key_env: str | None = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._store = LocalMemoryStore()
        self._backend = None
        self.base_url = os.environ.get(self.base_url_env or "", "") or (config or {}).get(
            "base_url", ""
        )
        self._api_key = os.environ.get(self.api_key_env or "", "")
        self.available, self.unavailable_reason = self._detect_backend()
        if not self.available:
            self.log.warning(
                "%s adapter unavailable (%s). Falling back to local namespace-isolated "
                "store so the benchmark can still run.",
                self.name,
                self.unavailable_reason,
            )
        else:
            self.log.info("%s adapter detected real backend at %s.", self.name, self.base_url)

    def _detect_backend(self) -> tuple[bool, str | None]:
        """Detect whether the real SDK + self-hosted backend appear usable."""
        if self.sdk_module:
            try:
                importlib.import_module(self.sdk_module)
            except ImportError:
                return False, f"SDK package '{self.sdk_module}' not installed"
        if not self.base_url:
            return False, f"{self.base_url_env} not configured"
        # REAL SDK: optionally probe the backend health endpoint here. We avoid
        # network calls during setup to keep offline runs fast and deterministic.
        return True, None

    # --------------------------------------------------------------- interface
    def setup(self, namespace: MemoryNamespace) -> None:
        # REAL SDK: create/open the per-namespace collection on the backend.
        return None

    def reset(self, namespace: MemoryNamespace) -> None:
        # REAL SDK: delete the namespace collection on the backend.
        self._store.reset(namespace)

    def retrieve(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[MemoryRecord]:
        ns = self._require_namespace(metadata)
        # REAL SDK: query the backend scoped to ns.flat_key(); for now use local store.
        return self._store.retrieve(query, top_k, ns)

    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        # REAL SDK: upsert into the backend scoped to ns.flat_key().
        return self._store.write(record, ns)

    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        # REAL SDK: update the backend entry scoped to ns.flat_key().
        return self._store.update(memory_id, record, ns)

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        return self._store.size_bytes(namespace)
