"""Zep: temporal knowledge-graph memory (self-hosted stub adapter).

Targets Zep Community Edition (see docker-compose ``zep-backend``). Organizes
memory around temporal relationships and evolving knowledge. Real integration
points are marked ``# REAL SDK:`` (use the ``zep_python`` client against the
self-hosted Zep endpoint).
"""

from __future__ import annotations

from benchmark.memory._stub_base import StubMemoryAdapter


class ZepAdapter(StubMemoryAdapter):
    name = "zep"
    # REAL SDK: "zep_python".
    sdk_module = "zep_python"
    base_url_env = "ZEP_BASE_URL"
    api_key_env = "ZEP_API_KEY"
