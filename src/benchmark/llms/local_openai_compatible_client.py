"""DEPRECATED. Use OpenRouterClient. Kept only for layout compatibility.

Local OpenAI-compatible endpoints (e.g. vLLM/Ollama) are not used for model
calls in this harness; all model calls route through OpenRouter for a single,
reproducible provider surface.
"""

from __future__ import annotations

from benchmark.llms._deprecated import deprecated_client

LocalOpenAICompatibleClient = deprecated_client("local")
