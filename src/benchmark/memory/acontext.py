"""AContext: skill-based memory (self-hosted stub adapter).

Stores reusable skills / learned interaction patterns. Real integration points
are marked ``# REAL SDK:``.
"""

from __future__ import annotations

from benchmark.memory._stub_base import StubMemoryAdapter


class AContextAdapter(StubMemoryAdapter):
    name = "acontext"
    sdk_module = None
    base_url_env = "ACONTEXT_BASE_URL"
    api_key_env = "ACONTEXT_API_KEY"
