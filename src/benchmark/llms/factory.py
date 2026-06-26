"""LLM provider selection and client construction."""

from __future__ import annotations

import os
from typing import Any

from benchmark.config import AppConfig
from benchmark.llms.base import LLMClient
from benchmark.llms.model_resolver import ModelResolver, ResolvedModel
from benchmark.llms.openai_resolver import OpenAIModelResolver


def llm_provider() -> str:
    """Return the active LLM provider: ``openai`` or ``openrouter`` (default)."""
    return os.environ.get("LLM_PROVIDER", "openrouter").strip().lower() or "openrouter"


def build_model_resolver(*, results_root: Any, config: AppConfig) -> ModelResolver | OpenAIModelResolver:
    """Instantiate the resolver matching ``LLM_PROVIDER``."""
    policy = config.openrouter_policy
    if llm_provider() == "openai":
        return OpenAIModelResolver(
            results_root=results_root,
            pricing_config=config.pricing,
            fail_if_model_unavailable=bool(policy.get("fail_if_model_unavailable", False)),
        )
    return ModelResolver(
        results_root=results_root,
        base_url=policy.get("base_url"),
        provider_policy=config.provider_policy,
        fail_if_model_unavailable=bool(policy.get("fail_if_model_unavailable", False)),
    )


def build_llm_client(rm: ResolvedModel, config: AppConfig) -> LLMClient:
    """Build the LLM client for a resolved model slot."""
    if llm_provider() == "openai":
        from benchmark.llms.openai_client import OpenAIClient

        return OpenAIClient(
            requested_model=rm.requested_model,
            resolved_model=rm.resolved_model,
            provider_family=rm.provider_family,
            benchmark_tier=rm.benchmark_tier,
            pricing=rm.pricing,
            temperature=rm.temperature,
            max_output_tokens=rm.max_output_tokens,
            reasoning=rm.reasoning,
        )

    from benchmark.llms.openrouter_client import OpenRouterClient

    return OpenRouterClient(
        requested_model=rm.requested_model,
        resolved_model=rm.resolved_model,
        provider_family=rm.provider_family,
        benchmark_tier=rm.benchmark_tier,
        supported_parameters=rm.supported_parameters,
        pricing=rm.pricing,
        provider_policy=config.provider_policy,
        temperature=rm.temperature,
        max_output_tokens=rm.max_output_tokens,
        reasoning=rm.reasoning,
        base_url=config.openrouter_policy.get("base_url"),
    )
