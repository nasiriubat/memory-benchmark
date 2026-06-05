"""Supermemory: retrieval-based persistent memory (self-hosted stub adapter).

Persistent retrieval-based memory for storing and retrieving prior task
information. Real integration points are marked ``# REAL SDK:``.
"""

from __future__ import annotations

from benchmark.memory._stub_base import StubMemoryAdapter


class SupermemoryAdapter(StubMemoryAdapter):
    name = "supermemory"
    sdk_module = None
    base_url_env = "SUPERMEMORY_BASE_URL"
    api_key_env = "SUPERMEMORY_API_KEY"
