"""Shared helper for deprecated direct-provider clients.

OpenRouter is the only LLM API provider used for model calls. The direct
per-provider clients are deprecated; they exist only to preserve the documented
file layout and to point callers at :class:`OpenRouterClient`.
"""

from __future__ import annotations

from typing import Any

from benchmark.llms.base import LLMClient
from benchmark.llms.openrouter_client import OpenRouterClient
from benchmark.logging import get_logger

_log = get_logger("benchmark.llm.deprecated")


def deprecated_client(provider_name: str) -> type[LLMClient]:
    """Build a deprecated client class for ``provider_name``.

    Constructing it logs a deprecation warning and returns an OpenRouterClient
    configured for the same provider family, so nothing silently breaks.
    """

    class _DeprecatedClient(OpenRouterClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _log.warning(
                "%s direct client is deprecated. All LLM calls go through "
                "OpenRouter. Use OpenRouterClient (provider_family=%s).",
                provider_name,
                provider_name,
            )
            kwargs.setdefault("provider_family", provider_name)
            super().__init__(*args, **kwargs)

    _DeprecatedClient.__name__ = f"{provider_name.capitalize()}Client"
    _DeprecatedClient.__qualname__ = _DeprecatedClient.__name__
    return _DeprecatedClient
