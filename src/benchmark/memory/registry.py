"""Memory adapter registry / factory.

Loads adapters from ``configs/memory_systems.yaml`` by their ``module:Class``
path, so adding a new memory system requires no runner changes.
"""

from __future__ import annotations

import importlib
from typing import Any

from benchmark.memory.base import MemoryAdapter


def load_adapter(memory_system: str, registry_config: dict[str, Any]) -> MemoryAdapter:
    """Instantiate the adapter for ``memory_system`` from the registry config."""
    systems = (registry_config or {}).get("memory_systems", {})
    if memory_system not in systems:
        raise KeyError(
            f"Unknown memory system '{memory_system}'. "
            f"Known: {sorted(systems.keys())}"
        )
    spec = systems[memory_system]
    if spec.get("enabled") is False:
        raise ValueError(
            f"Memory system '{memory_system}' is disabled in memory_systems.yaml"
        )
    dotted = spec["adapter"]
    module_path, _, class_name = dotted.partition(":")
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls(config=spec)
