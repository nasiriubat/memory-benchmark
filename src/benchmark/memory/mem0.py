"""Mem0: managed long-term memory (self-hosted stub adapter).

Self-hosted with a Qdrant vector store (see docker-compose ``mem0-qdrant``).
Supports structured store/update/retrieve operations. Real integration points
are marked ``# REAL SDK:`` (use the ``mem0ai`` package configured against the
self-hosted Qdrant endpoint).
"""

from __future__ import annotations

from benchmark.memory._stub_base import StubMemoryAdapter


class Mem0Adapter(StubMemoryAdapter):
    name = "mem0"
    # REAL SDK: "mem0" (mem0ai). Detection is enabled; if installed AND the
    # Qdrant backend is reachable, wire real calls in the overridden methods.
    sdk_module = "mem0"
    base_url_env = "MEM0_BASE_URL"
    api_key_env = "MEM0_API_KEY"
