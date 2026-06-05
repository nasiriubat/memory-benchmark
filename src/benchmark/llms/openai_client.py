"""DEPRECATED. Use OpenRouterClient. Kept only for layout compatibility."""

from __future__ import annotations

from benchmark.llms._deprecated import deprecated_client

# OpenRouter is the only LLM API provider. This shim forwards to OpenRouterClient.
OpenAIClient = deprecated_client("openai")
