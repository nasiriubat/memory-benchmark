"""Hindsight (Vectorize) memory adapter — self-hosted API integration.

Uses the official ``hindsight`` Python client against a running
``ghcr.io/vectorize-io/hindsight`` container. Each benchmark namespace maps to
a dedicated Hindsight *bank_id* so models/tasks never share memory.

Falls back to :class:`LocalMemoryStore` when the client is missing or the API
is unreachable.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Any

from benchmark.memory.base import MemoryAdapter
from benchmark.memory.local_store import LocalMemoryStore
from benchmark.schemas import MemoryNamespace, MemoryRecord, MemoryWriteResult
from benchmark.utils import utcnow_iso

def _bank_id(namespace: MemoryNamespace) -> str:
    """Stable bank id from the benchmark namespace (path-safe, bounded length)."""
    key = namespace.flat_key()
    if len(key) <= 128:
        return key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return f"bench_{digest}"


class HindsightAdapter(MemoryAdapter):
    name = "hindsight"
    base_url_env = "HINDSIGHT_BASE_URL"
    api_key_env = "HINDSIGHT_API_KEY"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._fallback = LocalMemoryStore()
        self._client: Any = None
        # The hindsight sync client binds an aiohttp session to the event loop
        # of the thread that creates it. Under task_parallelism > 1 the adapter
        # is called from worker threads, so each thread gets its own client.
        self._tls = threading.local()
        self.base_url = (
            os.environ.get(self.base_url_env, "")
            or (config or {}).get("base_url", "")
            or "http://hindsight-server:8888"
        ).rstrip("/")
        self._api_key = os.environ.get(self.api_key_env, "")
        self.available, self.unavailable_reason = self._connect()
        if not self.available:
            self.uses_local_fallback = True
            self.log.warning(
                "hindsight unavailable (%s). Using local namespace-isolated fallback.",
                self.unavailable_reason,
            )
        else:
            self.log.info("hindsight connected at %s", self.base_url)

    def _connect(self) -> tuple[bool, str | None]:
        try:
            from hindsight import HindsightClient
        except ImportError:
            return False, "pip package 'hindsight-all' not installed"
        client: Any = None
        try:
            client = HindsightClient(
                base_url=self.base_url,
                api_key=self._api_key or None,
                timeout=120.0,
            )
            # Probe API (banks is a BanksAPI object, not callable).
            client.banks.list()
            self._client = client
            self._tls.client = client  # reuse the probe client on this thread
            return True, None
        except Exception as exc:  # noqa: BLE001
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
            return False, str(exc)

    def _client_for_thread(self) -> Any:
        """Return a HindsightClient bound to the *current* thread's event loop.

        The hindsight sync wrapper drives async calls through a per-thread event
        loop (hindsight_client._run_async), and an aiohttp session is bound to
        the loop that created it. With task_parallelism > 1 the adapter runs in
        ThreadPoolExecutor worker threads, so each thread needs its own client —
        otherwise every call raises "Timeout context manager should be used
        inside a task" and silently falls back to the local store. Cached per
        thread; created lazily.
        """
        if not self.available:
            return None
        client = getattr(self._tls, "client", None)
        if client is not None:
            return client
        try:
            from hindsight import HindsightClient

            client = HindsightClient(
                base_url=self.base_url,
                api_key=self._api_key or None,
                timeout=120.0,
            )
            # Bind this client's aiohttp session to the current thread's loop.
            client.banks.list()
        except Exception as exc:  # noqa: BLE001
            self.log.warning(
                "hindsight per-thread client init failed (%s); fallback store", exc
            )
            return None
        self._tls.client = client
        return client

    def _ensure_bank(self, client: Any, namespace: MemoryNamespace) -> str:
        bid = _bank_id(namespace)
        try:
            client.create_bank(bid, name=bid)
        except Exception:  # noqa: BLE001
            # Bank likely already exists for this namespace.
            pass
        return bid

    def setup(self, namespace: MemoryNamespace) -> None:
        client = self._client_for_thread()
        if client is not None:
            self._ensure_bank(client, namespace)

    def reset(self, namespace: MemoryNamespace) -> None:
        client = self._client_for_thread()
        if client is not None:
            bid = _bank_id(namespace)
            try:
                client.delete_bank(bid)
            except Exception:  # noqa: BLE001
                pass
            try:
                client.create_bank(bid, name=bid)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("hindsight reset failed for %s: %s", bid, exc)
        self._fallback.reset(namespace)

    def retrieve(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[MemoryRecord]:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.retrieve(query, top_k, ns)

        bid = _bank_id(ns)
        # Note: end-to-end retrieval latency is measured by the runner around
        # this call (see SequentialBenchmarkAdapter._run_subtask) and recorded
        # as memory_retrieval_latency_ms; no separate timer is needed here.
        try:
            response = client.recall(bid, query, max_tokens=4096)
            records: list[MemoryRecord] = []
            for i, item in enumerate((response.results or [])[:top_k]):
                text = getattr(item, "text", None) or str(item)
                records.append(
                    MemoryRecord(
                        id=getattr(item, "id", None) or f"hindsight-{i}",
                        text=text,
                        source="hindsight",
                        score=1.0,
                        metadata={"bank_id": bid},
                        namespace=ns,
                    )
                )
            return records
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("hindsight recall failed (%s); fallback store", exc)
            return self._fallback.retrieve(query, top_k, ns)

    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.write(record, ns)

        bid = self._ensure_bank(client, ns)
        t0 = time.perf_counter()
        try:
            client.retain(bid, record.text)
            latency = (time.perf_counter() - t0) * 1000
            return MemoryWriteResult(
                success=True,
                memory_id=record.id or f"{bid}-{utcnow_iso()}",
                latency_ms=latency,
            )
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("hindsight retain failed (%s); fallback store", exc)
            return self._fallback.write(record, ns)

    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        # Hindsight retain is upsert-oriented; re-retain updated text.
        record.id = memory_id
        return self.write(record, metadata)

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        client = self._client_for_thread()
        if client is not None:
            try:
                bid = _bank_id(namespace)
                mems = client.memories.list(bid, limit=1000)
                total = 0
                for m in mems or []:
                    total += len(getattr(m, "text", "") or "")
                return total
            except Exception:  # noqa: BLE001
                pass
        return self._fallback.size_bytes(namespace)

    def close(self) -> None:
        if self._client and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
