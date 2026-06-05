"""AutoMemory/AutoDream: built-in proprietary memory (optional, disabled).

Listed in the paper's Table I as a built-in proprietary memory mechanism. It is
optional and disabled by default in ``configs/memory_systems.yaml``; the final
description and integration are to be completed with the industrial partner.
"""

from __future__ import annotations

from benchmark.memory._stub_base import StubMemoryAdapter


class AutoMemoryAdapter(StubMemoryAdapter):
    name = "automemory"
    sdk_module = None
    base_url_env = "AUTOMEMORY_BASE_URL"
    api_key_env = "AUTOMEMORY_API_KEY"
