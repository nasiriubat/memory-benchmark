"""AContext memory adapter — self-hosted skill-based memory (acontext SDK).

Talks to a self-hosted AContext server (``memodb-io/Acontext``; bring it up with
``acontext server up`` or its compose) via the ``acontext`` Python SDK. AContext
is skill-based, not vector-search: messages are stored in a session, the server
*learns* reusable skills from them, and retrieval lists those skills.

Mapping to the benchmark interface:
  * namespace -> a dedicated LearningSpace + Session (server-assigned ids,
    tracked per namespace) so models/tasks/seeds never share memory.
  * write   -> store the text as a session message, then trigger learning.
  * retrieve-> list the learned skills (name + description) for the space.
  * reset   -> delete the namespace's space + session.

Connection: ``ACONTEXT_BASE_URL`` (e.g. http://host.docker.internal:8029/api/v1)
and ``ACONTEXT_API_KEY`` (the server's ROOT_API_BEARER_TOKEN). Falls back to a
local, namespace-isolated store when the SDK/backend is unavailable, recording
``uses_local_fallback=True``.
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

# Cap on how long a single write waits for server-side skill learning.
_LEARN_TIMEOUT = float(os.environ.get("ACONTEXT_LEARN_TIMEOUT", "60"))


def _ns_key(namespace: MemoryNamespace) -> str:
    key = namespace.flat_key().replace("::", "_")
    if len(key) <= 100 and key.replace("_", "").isalnum():
        return key
    return "bench_" + hashlib.sha256(namespace.flat_key().encode("utf-8")).hexdigest()[:32]


class AContextAdapter(MemoryAdapter):
    name = "acontext"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._fallback = LocalMemoryStore()
        self._tls = threading.local()
        self.base_url = (
            os.environ.get("ACONTEXT_BASE_URL", "")
            or (config or {}).get("base_url", "")
            or "http://localhost:8029/api/v1"
        ).rstrip("/")
        self._api_key = os.environ.get("ACONTEXT_API_KEY", "")
        self.available, self.unavailable_reason = self._connect()
        if not self.available:
            self.uses_local_fallback = True
            self.log.warning(
                "acontext unavailable (%s). Using local namespace-isolated fallback.",
                self.unavailable_reason,
            )
        else:
            self.log.info("acontext connected at %s", self.base_url)

    # ----------------------------------------------------------------- connect
    def _new_client(self) -> Any:
        from acontext import AcontextClient

        return AcontextClient(api_key=self._api_key or None, base_url=self.base_url, timeout=120.0)

    def _connect(self) -> tuple[bool, str | None]:
        try:
            import acontext  # noqa: F401
        except ImportError:
            return False, "pip package 'acontext' not installed"
        if not self.base_url:
            return False, "ACONTEXT_BASE_URL not configured"
        try:
            client = self._new_client()
            client.ping()
            self._tls.client = client
            self._tls.spaces = {}
        except Exception as exc:  # noqa: BLE001
            return False, f"acontext unreachable at {self.base_url}: {exc}"
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
            self.log.warning("acontext per-thread client init failed (%s); fallback store", exc)
            return None
        self._tls.client = client
        self._tls.spaces = {}
        return client

    def _ids_for(self, client: Any, key: str) -> tuple[str, str] | None:
        """Return (space_id, session_id) for a namespace, creating them once."""
        spaces = getattr(self._tls, "spaces", None)
        if spaces is None:
            spaces = self._tls.spaces = {}
        if key in spaces:
            return spaces[key]
        try:
            session = client.sessions.create(user=key)
            space = client.learning_spaces.create(user=key)
            spaces[key] = (space.id, session.id)
            return spaces[key]
        except Exception as exc:  # noqa: BLE001
            self.log.warning("acontext space/session create failed for %s: %s", key, exc)
            return None

    # --------------------------------------------------------------- interface
    def setup(self, namespace: MemoryNamespace) -> None:
        client = self._client_for_thread()
        if client is not None:
            self._ids_for(client, _ns_key(namespace))

    def reset(self, namespace: MemoryNamespace) -> None:
        client = self._client_for_thread()
        key = _ns_key(namespace)
        if client is not None:
            spaces = getattr(self._tls, "spaces", {}) or {}
            ids = spaces.pop(key, None)
            if ids is not None:
                space_id, session_id = ids
                for fn in (
                    lambda: client.learning_spaces.delete(space_id),
                    lambda: client.sessions.delete(session_id),
                ):
                    try:
                        fn()
                    except Exception:  # noqa: BLE001
                        pass
        self._fallback.reset(namespace)

    def retrieve(
        self, query: str, top_k: int, metadata: dict[str, Any]
    ) -> list[MemoryRecord]:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.retrieve(query, top_k, ns)
        ids = self._ids_for(client, _ns_key(ns))
        if ids is None:
            self.uses_local_fallback = True
            return self._fallback.retrieve(query, top_k, ns)
        space_id, _ = ids
        try:
            # AContext is skill-based: it lists learned skills rather than doing a
            # semantic top-k search, so we return the learned skills (capped).
            skills = client.learning_spaces.list_skills(space_id) or []
            records: list[MemoryRecord] = []
            for i, sk in enumerate(skills[:top_k]):
                name = getattr(sk, "name", "") or ""
                desc = getattr(sk, "description", "") or ""
                text = (f"{name}: {desc}" if name and desc else name or desc).strip()
                if not text:
                    continue
                records.append(
                    MemoryRecord(
                        id=str(getattr(sk, "id", None) or f"acontext-{i}"),
                        text=text,
                        source="acontext",
                        score=1.0,
                        metadata={"space_id": space_id},
                        namespace=ns,
                    )
                )
            return records
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("acontext list_skills failed (%s); fallback store", exc)
            return self._fallback.retrieve(query, top_k, ns)

    def write(self, record: MemoryRecord, metadata: dict[str, Any]) -> MemoryWriteResult:
        ns = self._require_namespace(metadata)
        client = self._client_for_thread()
        if client is None:
            return self._fallback.write(record, ns)
        ids = self._ids_for(client, _ns_key(ns))
        if ids is None:
            self.uses_local_fallback = True
            return self._fallback.write(record, ns)
        space_id, session_id = ids
        try:
            client.sessions.store_message(
                session_id,
                blob={"role": "user", "content": record.text},
                format="openai",
            )
            # Trigger server-side skill extraction so later retrievals see it.
            try:
                client.learning_spaces.learn(space_id, session_id=session_id)
                client.learning_spaces.wait_for_learning(
                    space_id, session_id=session_id, timeout=_LEARN_TIMEOUT
                )
            except Exception as exc:  # noqa: BLE001 - message stored; learning best-effort
                self.log.debug("acontext learn/wait best-effort failed: %s", exc)
            return MemoryWriteResult(
                success=True, memory_id=record.id or f"{space_id}-{utcnow_iso()}"
            )
        except Exception as exc:  # noqa: BLE001
            self.uses_local_fallback = True
            self.log.warning("acontext store_message failed (%s); fallback store", exc)
            return self._fallback.write(record, ns)

    def update(
        self, memory_id: str, record: MemoryRecord, metadata: dict[str, Any]
    ) -> MemoryWriteResult:
        # Skills are server-learned; re-store the text and re-learn.
        return self.write(record, metadata)

    def store_size_bytes(self, namespace: MemoryNamespace) -> int:
        client = self._client_for_thread()
        if client is not None:
            ids = getattr(self._tls, "spaces", {}).get(_ns_key(namespace))
            if ids is not None:
                try:
                    skills = client.learning_spaces.list_skills(ids[0]) or []
                    return sum(
                        len(getattr(s, "name", "") or "") + len(getattr(s, "description", "") or "")
                        for s in skills
                    )
                except Exception:  # noqa: BLE001
                    pass
        return self._fallback.size_bytes(namespace)
