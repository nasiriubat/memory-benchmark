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
        self.base_url = (
            os.environ.get(self.base_url_env, "")
            or (config or {}).get("base_url", "")
            or "http://hindsight-server:8888"
        ).rstrip("/")
        self._api_key = os.environ.get(self.api_key_env, "")
        self.available, self.unavailable_reason = self._connect()
        if not self.available:
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
            return True, None
        except Exception as exc:  # noqa: BLE001
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
            return False, str(exc)

    def _ensure_bank(self, namespace: MemoryNamespace) -> str:
        bid = _bank_id(namespace)
        assert self._client is not None
        try:
            self._client.create_bank(bid, name=bid)
        except Exception:  # noqa: BLE001
            # Bank likely already exists for this namespace.
            pass
        return bid

    def setup(self, namespace: MemoryNamespace) -> None:
        if self.available and self._client:
            self._ensure_bank(namespace)

    def reset(self, namespace: MemoryNamespace) -> None:
        if self.available and self._client:
            bid = _bank_id(namespace)
            try:
                self._client.delete_bank(bid)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._client.create_bank(bid, name=bid)
            except Exception as exc:  # noqa: BLE001
                self.log.warning("hindsight reset failed for %s: %s", bid, exc)
        self._fallback.reset(namespace)

    def retrieve(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[MemoryRecord]:
        ns = self._require_namespace(metadata)
        if not self.available or not self._client:
            return self._fallback.retrieve(query, top_k, ns)

        bid = _bank_id(ns)
        t0 = time.perf_counter()
        try:
            response = self._client.recall(bid, query, max_tokens=4096)
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
            self.log.warning("hindsight recall failed (%s); fallback store", exc)
            return self._fallback.retrieve(query, top_k, ns)
        finally:
            _ = (time.perf_counter() - t0) * 1000

    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        if not self.available or not self._client:
            return self._fallback.write(record, ns)

        bid = self._ensure_bank(ns)
        t0 = time.perf_counter()
        try:
            self._client.retain(bid, record.text)
            latency = (time.perf_counter() - t0) * 1000
            return MemoryWriteResult(
                success=True,
                memory_id=record.id or f"{bid}-{utcnow_iso()}",
                latency_ms=latency,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.warning("hindsight retain failed (%s); fallback store", exc)
            return self._fallback.write(record, ns)

    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        # Hindsight retain is upsert-oriented; re-retain updated text.
        record.id = memory_id
        return self.write(record, metadata)

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        if self.available and self._client:
            try:
                bid = _bank_id(namespace)
                mems = self._client.memories.list(bid, limit=1000)
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
