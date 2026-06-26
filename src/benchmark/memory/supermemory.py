"""Supermemory memory adapter — self-hosted (supermemory SDK).

Talks to the self-hosted Supermemory backend (docker-compose ``supermemory-backend``)
at ``SUPERMEMORY_BASE_URL`` via the sync ``supermemory.Supermemory`` client.
Each benchmark namespace maps to a dedicated ``container_tag`` so models/tasks/
seeds never share memory.

Falls back to a local, namespace-isolated store when the SDK/backend is
unavailable, recording ``uses_local_fallback=True``.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

from benchmark.memory.base import MemoryAdapter
from benchmark.memory.local_store import LocalMemoryStore
from benchmark.schemas import MemoryNamespace, MemoryRecord, MemoryWriteResult
from benchmark.utils import utcnow_iso


def _ns_key(namespace: MemoryNamespace) -> str:
    """Backend-safe isolation key (Supermemory container_tag)."""
    key = namespace.flat_key().replace("::", "_")
    if len(key) <= 100 and key.replace("_", "").isalnum():
        return key
    return "bench_" + hashlib.sha256(namespace.flat_key().encode("utf-8")).hexdigest()[:32]


def _items(response: Any) -> list[Any]:
    """Pull the list of hits out of a Supermemory response, shape-tolerant."""
    for attr in ("results", "memories", "documents", "data"):
        val = getattr(response, attr, None)
        if isinstance(val, list):
            return val
    if isinstance(response, list):
        return response
    return []


class SupermemoryAdapter(MemoryAdapter):
    name = "supermemory"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._fallback = LocalMemoryStore()
        self._tls = threading.local()
        self.base_url = (
            os.environ.get("SUPERMEMORY_BASE_URL", "")
            or (config or {}).get("base_url", "")
            or "http://supermemory:8000"
        ).rstrip("/")
        self._api_key = os.environ.get("SUPERMEMORY_API_KEY", "")
        self.available, self.unavailable_reason = self._connect()
        if not self.available:
            self.uses_local_fallback = True
            self.log.warning(
                "supermemory unavailable (%s). Using local namespace-isolated fallback.",
                self.unavailable_reason,
            )
        else:
            self.log.info("supermemory connected at %s", self.base_url)

    # ----------------------------------------------------------------- connect
    def _new_client(self) -> Any:
        from supermemory import Supermemory

        return Supermemory(api_key=self._api_key or None, base_url=self.base_url, timeout=120.0)

    def _connect(self) -> tuple[bool, str | None]:
        try:
            import supermemory  # noqa: F401
        except ImportError:
            return False, "pip package 'supermemory' not installed"
        if not self.base_url:
            return False, "SUPERMEMORY_BASE_URL not configured"
        try:
            client = self._new_client()
            # Lightweight probe: a search returns quickly even with no data.
            client.search.memories(q="ping", container_tag="__healthcheck__", limit=1)
            self._tls.client = client
        except Exception as exc:  # noqa: BLE001
            return False, f"supermemory unreachable at {self.base_url}: {exc}"
        return True, None

    def _client_for_thread(self) -> Any:
        if not self.available:
            return None
        client = getattr(self._tls, "client", None)
        if client is not None:
            return client
        try:
            client = self._new_client()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("supermemory per-thread client init failed (%s); fallback store", exc)
            return None
        self._tls.client = client
        return client

    # --------------------------------------------------------------- interface
    def setup(self, namespace: MemoryNamespace) -> None:
        # Containers are created implicitly on first add; nothing to do.
        return None

    def reset(self, namespace: MemoryNamespace) -> None:
        client = self._client_for_thread()
        if client is not None:
            key = _ns_key(namespace)
            try:
                listed = client.documents.list(container_tags=[key])
                for doc in _items(listed):
                    doc_id = getattr(doc, "id", None) or getattr(doc, "document_id", None)
                    if doc_id:
                        try:
                            client.documents.delete(doc_id)
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as exc:  # noqa: BLE001
                # Namespaces are unique per (task, seed), so stale data is
                # unlikely within a run; degrade quietly if list/delete is unsupported.
                self.log.debug("supermemory reset best-effort failed for %s: %s", key, exc)
        self._fallback.reset(namespace)

    def retrieve(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[MemoryRecord]:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.retrieve(query, top_k, ns)
        key = _ns_key(ns)
        try:
            resp = client.search.memories(q=query, container_tag=key, limit=top_k)
            records: list[MemoryRecord] = []
            for i, item in enumerate(_items(resp)[:top_k]):
                text = (
                    getattr(item, "memory", None)
                    or getattr(item, "content", None)
                    or getattr(item, "text", None)
                    or ""
                )
                if not text:
                    continue
                records.append(
                    MemoryRecord(
                        id=str(getattr(item, "id", None) or f"supermemory-{i}"),
                        text=text,
                        source="supermemory",
                        score=float(getattr(item, "score", 1.0) or 0.0),
                        metadata={"container_tag": key},
                        namespace=ns,
                    )
                )
            return records
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("supermemory search failed (%s); fallback store", exc)
            return self._fallback.retrieve(query, top_k, ns)

    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.write(record, ns)
        key = _ns_key(ns)
        try:
            res = client.add(content=record.text, container_tag=key)
            mem_id = getattr(res, "id", None) or record.id
            return MemoryWriteResult(
                success=True, memory_id=str(mem_id) if mem_id else f"{key}-{utcnow_iso()}"
            )
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("supermemory add failed (%s); fallback store", exc)
            return self._fallback.write(record, ns)

    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        # Supermemory add is upsert-friendly via custom_id; re-add the text.
        return self.write(record, metadata)

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        client = self._client_for_thread()
        if client is not None:
            try:
                listed = client.documents.list(container_tags=[_ns_key(namespace)])
                total = 0
                for doc in _items(listed):
                    total += len(getattr(doc, "content", "") or getattr(doc, "summary", "") or "")
                return total
            except Exception:  # noqa: BLE001
                pass
        return self._fallback.size_bytes(namespace)
