"""LLM provider factory."""

from __future__ import annotations

import os

from benchmark.config import load_config
from benchmark.llms.factory import build_llm_client, build_model_resolver, llm_provider
from benchmark.llms.model_resolver import ResolvedModel
from benchmark.llms.openai_client import OpenAIClient
from benchmark.llms.openai_resolver import OpenAIModelResolver
from benchmark.llms.openrouter_client import OpenRouterClient


def test_llm_provider_defaults_to_openrouter(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_provider() == "openrouter"


def test_build_model_resolver_openai(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = load_config()
    resolver = build_model_resolver(results_root=tmp_path, config=cfg)
    assert isinstance(resolver, OpenAIModelResolver)


def test_build_llm_client_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = load_config()
    rm = ResolvedModel(
        slot="openai_frontier",
        provider_family="openai",
        benchmark_tier="frontier",
        requested_model="gpt-4o",
        resolved_model="gpt-4o",
        available=True,
    )
    client = build_llm_client(rm, cfg)
    assert isinstance(client, OpenAIClient)


def test_build_llm_client_openrouter(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = load_config()
    rm = ResolvedModel(
        slot="openai_frontier",
        provider_family="openai",
        benchmark_tier="frontier",
        requested_model="~openai/gpt-latest",
        resolved_model="openai/gpt-5.5-pro",
        available=True,
    )
    client = build_llm_client(rm, cfg)
    assert isinstance(client, OpenRouterClient)
