"""Zep memory adapter — self-hosted Zep Community Edition (zep_python v2).

Talks to the self-hosted Zep backend (docker-compose ``zep-backend``) at
``ZEP_BASE_URL`` via the sync ``zep_python.Zep`` client. Zep organises memory as
User -> Session -> Messages and does its own temporal/graph extraction inside the
backend. Each benchmark namespace maps to a dedicated Zep user_id AND session_id
(same key) so models/tasks/seeds never share memory.

The sync client wraps an httpx.Client; we still keep one client per worker thread
(``threading.local``) for uniformity and safety under TASK_PARALLELISM > 1.
Falls back to a local, namespace-isolated store when the SDK/backend is
unavailable, recording ``uses_local_fallback=True``.
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


def _ns_key(namespace: MemoryNamespace) -> str:
    """Backend-safe isolation key (Zep user_id and session_id)."""
    key = namespace.flat_key().replace("::", "_")
    if len(key) <= 100 and key.replace("_", "").isalnum():
        return key
    return "bench_" + hashlib.sha256(namespace.flat_key().encode("utf-8")).hexdigest()[:32]


class ZepAdapter(MemoryAdapter):
    name = "zep"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._fallback = LocalMemoryStore()
        self._tls = threading.local()
        self.base_url = (
            os.environ.get("ZEP_BASE_URL", "")
            or (config or {}).get("base_url", "")
            or "http://zep:8000"
        ).rstrip("/")
        self._api_key = os.environ.get("ZEP_API_KEY", "")
        self.available, self.unavailable_reason = self._connect()
        if not self.available:
            self._unavailable_fallback = True
            self.log.warning(
                "zep unavailable (%s). Using local namespace-isolated fallback.",
                self.unavailable_reason,
            )
        else:
            self.log.info("zep connected at %s", self.base_url)

    # ----------------------------------------------------------------- connect
    def _new_client(self) -> Any:
        from zep_python.client import Zep

        return Zep(base_url=self.base_url, api_key=self._api_key or None, timeout=120.0)

    def _connect(self) -> tuple[bool, str | None]:
        try:
            import zep_python  # noqa: F401
        except ImportError:
            return False, "pip package 'zep-python' not installed"
        if not self.base_url:
            return False, "ZEP_BASE_URL not configured"
        # Zep CE starts slowly (Postgres + schema migrations), so a single probe
        # at init loses the race and trips the STICKY unavailable-fallback for the
        # whole run. Poll the health endpoint for up to ~90s before giving up.
        deadline_attempts = int(os.environ.get("ZEP_CONNECT_ATTEMPTS", "30"))
        last_exc: Exception | None = None
        for attempt in range(deadline_attempts):
            try:
                client = self._new_client()
                client.user.list_ordered(page_size=1)  # health probe
                self._tls.client = client
                if attempt:
                    self.log.info("zep ready after %d attempt(s)", attempt + 1)
                return True, None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(3.0)
        return False, f"zep unreachable at {self.base_url} after {deadline_attempts} attempts: {last_exc}"

    def _client_for_thread(self) -> Any:
        if not self.available:
            return None
        client = getattr(self._tls, "client", None)
        if client is not None:
            return client
        try:
            client = self._new_client()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("zep per-thread client init failed (%s); fallback store", exc)
            return None
        self._tls.client = client
        return client

    def _ensure_session(self, client: Any, key: str) -> None:
        try:
            client.user.add(user_id=key)
        except Exception:  # noqa: BLE001
            pass  # user likely already exists
        try:
            client.memory.add_session(session_id=key, user_id=key)
        except Exception:  # noqa: BLE001
            pass  # session likely already exists

    # --------------------------------------------------------------- interface
    def setup(self, namespace: MemoryNamespace) -> None:
        client = self._client_for_thread()
        if client is not None:
            self._ensure_session(client, _ns_key(namespace))

    def reset(self, namespace: MemoryNamespace) -> None:
        client = self._client_for_thread()
        if client is not None:
            key = _ns_key(namespace)
            try:
                client.user.delete(key)  # cascades the user's sessions/messages
            except Exception:  # noqa: BLE001
                pass
            self._ensure_session(client, key)
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
            resp = client.memory.search_sessions(
                user_id=key, text=query, search_scope="messages", limit=top_k
            )
            results = getattr(resp, "results", None) or []
            records: list[MemoryRecord] = []
            for i, r in enumerate(results[:top_k]):
                msg = getattr(r, "message", None)
                fact = getattr(r, "fact", None)
                summ = getattr(r, "summary", None)
                text = (
                    (getattr(msg, "content", None) if msg else None)
                    or (getattr(fact, "fact", None) if fact else None)
                    or (getattr(summ, "content", None) if summ else None)
                    or ""
                )
                if not text:
                    continue
                records.append(
                    MemoryRecord(
                        id=str(getattr(r, "uuid", None) or f"zep-{i}"),
                        text=text,
                        source="zep",
                        score=float(getattr(r, "score", 1.0) or 0.0),
                        metadata={"session_id": key},
                        namespace=ns,
                    )
                )
            return records
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("zep search failed (%s); fallback store", exc)
            return self._fallback.retrieve(query, top_k, ns)

    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.write(record, ns)
        key = _ns_key(ns)
        try:
            from zep_python import Message

            self._ensure_session(client, key)
            client.memory.add(
                key, messages=[Message(role_type="user", content=record.text)]
            )
            return MemoryWriteResult(
                success=True, memory_id=record.id or f"{key}-{utcnow_iso()}"
            )
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("zep add failed (%s); fallback store", exc)
            return self._fallback.write(record, ns)

    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        # Zep messages are append-only; re-add the updated text.
        record.id = memory_id
        return self.write(record, metadata)

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        client = self._client_for_thread()
        if client is not None:
            try:
                msgs = client.memory.get_session_messages(_ns_key(namespace))
                items = getattr(msgs, "messages", None) or []
                return sum(len(getattr(m, "content", "") or "") for m in items)
            except Exception:  # noqa: BLE001
                pass
        return self._fallback.size_bytes(namespace)
