"""Mem0 memory adapter — self-hosted (mem0ai) against a Qdrant vector store.

Uses the ``mem0`` (``mem0ai``) OSS client configured to talk to the self-hosted
Qdrant backend (docker-compose ``mem0-qdrant``) at ``MEM0_BASE_URL``. Each
benchmark namespace maps to a Mem0 ``user_id`` inside one shared collection, so
models/tasks/seeds never share memory.

Mem0 runs its own LLM extraction on writes; that LLM follows the active provider
(``LLM_PROVIDER`` + the matching ``OPENAI_*`` / ``OPENROUTER_*`` env) and uses a
cheap model by default (``MEM0_LLM_MODEL``). Embeddings use OpenAI directly
(``MEM0_EMBED_MODEL``) because OpenRouter does not serve an embeddings API.

Falls back to a local, namespace-isolated store when the SDK or backend is
unavailable, recording ``uses_local_fallback=True`` so degraded runs are never
mistaken for a real Mem0 measurement.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any
from urllib.parse import urlparse

from benchmark.memory.base import MemoryAdapter
from benchmark.memory.local_store import LocalMemoryStore
from benchmark.schemas import MemoryNamespace, MemoryRecord, MemoryWriteResult
from benchmark.utils import utcnow_iso

_COLLECTION = "benchmark_mem0"


def _ns_key(namespace: MemoryNamespace) -> str:
    """Backend-safe isolation key from the namespace (Mem0 ``user_id``)."""
    key = namespace.flat_key().replace("::", "_")
    if len(key) <= 128 and key.replace("_", "").isalnum():
        return key
    return "bench_" + hashlib.sha256(namespace.flat_key().encode("utf-8")).hexdigest()[:32]


class Mem0Adapter(MemoryAdapter):
    name = "mem0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._fallback = LocalMemoryStore()
        # Mem0's Memory bundles a Qdrant client + OpenAI clients; share-across-
        # threads safety is not guaranteed, so (like hindsight) we build one
        # client per worker thread under TASK_PARALLELISM > 1.
        self._tls = threading.local()
        self.base_url = (
            os.environ.get("MEM0_BASE_URL", "")
            or (config or {}).get("base_url", "")
            or "http://mem0-qdrant:6333"
        ).rstrip("/")
        self.available, self.unavailable_reason = self._connect()
        if not self.available:
            self.uses_local_fallback = True
            self.log.warning(
                "mem0 unavailable (%s). Using local namespace-isolated fallback.",
                self.unavailable_reason,
            )
        else:
            self.log.info("mem0 connected to Qdrant at %s", self.base_url)

    # ----------------------------------------------------------------- connect
    def _connect(self) -> tuple[bool, str | None]:
        try:
            import mem0  # noqa: F401
        except ImportError:
            return False, "pip package 'mem0ai' not installed"
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            return False, "pip package 'qdrant-client' not installed"
        if not self.base_url:
            return False, "MEM0_BASE_URL not configured"
        try:
            parsed = urlparse(self.base_url)
            probe = QdrantClient(host=parsed.hostname, port=parsed.port or 6333, timeout=10.0)
            probe.get_collections()  # health probe
            try:
                probe.close()
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            return False, f"Qdrant unreachable at {self.base_url}: {exc}"
        return True, None

    def _mem0_config(self) -> dict[str, Any]:
        parsed = urlparse(self.base_url)
        provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()
        if provider == "openai":
            llm_key = os.environ.get("OPENAI_API_KEY", "")
            llm_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            llm_model = os.environ.get("MEM0_LLM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5.5"))
        else:
            llm_key = os.environ.get("OPENROUTER_API_KEY", "")
            llm_base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            llm_model = os.environ.get("MEM0_LLM_MODEL", "openai/gpt-5.5-mini")
        # Embeddings: OpenRouter has no embeddings API, so embeddings always go to
        # OpenAI directly (key may differ from the chat key under openrouter).
        embed_key = os.environ.get("MEM0_EMBED_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        embed_model = os.environ.get("MEM0_EMBED_MODEL", "text-embedding-3-small")
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": parsed.hostname,
                    "port": parsed.port or 6333,
                    "collection_name": _COLLECTION,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {"model": llm_model, "api_key": llm_key, "openai_base_url": llm_base},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": embed_model, "api_key": embed_key},
            },
        }

    def _client_for_thread(self) -> Any:
        """Return a Mem0 ``Memory`` bound to the current thread (lazy, cached)."""
        if not self.available:
            return None
        client = getattr(self._tls, "client", None)
        if client is not None:
            return client
        try:
            from mem0 import Memory

            client = Memory.from_config(self._mem0_config())
        except Exception as exc:  # noqa: BLE001
            self.log.warning("mem0 per-thread client init failed (%s); fallback store", exc)
            return None
        self._tls.client = client
        return client

    # --------------------------------------------------------------- interface
    def setup(self, namespace: MemoryNamespace) -> None:
        # Collection is created lazily by Mem0 on first write; nothing to do.
        return None

    def reset(self, namespace: MemoryNamespace) -> None:
        client = self._client_for_thread()
        if client is not None:
            try:
                client.delete_all(user_id=_ns_key(namespace))
            except Exception as exc:  # noqa: BLE001
                self.log.warning("mem0 reset failed for %s: %s", _ns_key(namespace), exc)
        self._fallback.reset(namespace)

    @staticmethod
    def _search_items(response: Any) -> list[dict[str, Any]]:
        """Normalize Mem0 return shape (list or {"results": [...]})."""
        if isinstance(response, dict):
            return response.get("results", []) or []
        return response or []

    def retrieve(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[MemoryRecord]:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.retrieve(query, top_k, ns)
        try:
            # Newer mem0 requires scoping via filters=; older accepts user_id=.
            try:
                response = client.search(query, limit=top_k, filters={"user_id": _ns_key(ns)})
            except Exception:  # noqa: BLE001
                response = client.search(query, limit=top_k, user_id=_ns_key(ns))
            records: list[MemoryRecord] = []
            for i, item in enumerate(self._search_items(response)[:top_k]):
                text = item.get("memory") or item.get("text") or ""
                if not text:
                    continue
                records.append(
                    MemoryRecord(
                        id=str(item.get("id") or f"mem0-{i}"),
                        text=text,
                        source="mem0",
                        score=float(item.get("score", 1.0) or 0.0),
                        metadata={"user_id": _ns_key(ns)},
                        namespace=ns,
                    )
                )
            return records
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("mem0 search failed (%s); fallback store", exc)
            return self._fallback.retrieve(query, top_k, ns)

    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.write(record, ns)
        try:
            res = client.add(record.text, user_id=_ns_key(ns))
            mem_id = record.id
            items = self._search_items(res)
            if items and items[0].get("id"):
                mem_id = str(items[0]["id"])
            return MemoryWriteResult(
                success=True,
                memory_id=mem_id or f"{_ns_key(ns)}-{utcnow_iso()}",
            )
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("mem0 add failed (%s); fallback store", exc)
            return self._fallback.write(record, ns)

    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.update(memory_id, record, ns)
        try:
            client.update(memory_id=memory_id, data=record.text)
            return MemoryWriteResult(success=True, memory_id=memory_id)
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("mem0 update failed (%s); fallback store", exc)
            return self._fallback.update(memory_id, record, ns)

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        client = self._client_for_thread()
        if client is not None:
            try:
                try:
                    res = client.get_all(filters={"user_id": _ns_key(namespace)})
                except Exception:  # noqa: BLE001
                    res = client.get_all(user_id=_ns_key(namespace))
                total = 0
                for m in self._search_items(res):
                    total += len((m.get("memory") or m.get("text") or ""))
                return total
            except Exception:  # noqa: BLE001
                pass
        return self._fallback.size_bytes(namespace)
