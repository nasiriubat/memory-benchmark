"""Base LLM client interface."""

from __future__ import annotations

import abc
from typing import Any

from benchmark.schemas import LLMResponse


class LLMClient(abc.ABC):
    """Abstract LLM client.

    Implementations must return a normalized :class:`LLMResponse` capturing
    text, tool calls, token usage, cost, latency, and routing metadata.
    """

    #: Provider family label used for analysis (openai/anthropic/google/meta/deepseek).
    provider_family: str = "unknown"
    #: The model id requested by configuration (may be a "~latest" alias).
    requested_model: str = ""
    #: The concrete model id actually used after resolution.
    resolved_model: str = ""

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a completion for ``messages``."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - optional hook
        """Release any resources. Default no-op."""
        return None
