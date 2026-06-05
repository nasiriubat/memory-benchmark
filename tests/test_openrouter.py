"""OpenRouter client + model resolver tests.

These tests run offline (no OPENROUTER_API_KEY) and exercise: alias resolution
to the concrete fallback, resolve-once stability, parameter stripping, strict
mode disabling fallback routing, snapshot-pricing cost calculation, and result
provenance fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.evaluators.cost import estimate_cost_usd
from benchmark.llms.model_resolver import ModelResolver
from benchmark.llms.openrouter_client import OpenRouterClient


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    # Force offline mode for deterministic, network-free tests.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    yield


def _models_config():
    return {
        "models": {
            "openai_frontier": {
                "enabled": True,
                "provider": "openrouter",
                "provider_family": "openai",
                "requested_model": "~openai/gpt-latest",
                "concrete_fallback_model": "openai/gpt-5.5-pro",
                "benchmark_tier": "frontier",
                "temperature": 0,
                "max_output_tokens": 4096,
                "reasoning": {"enabled": True, "effort": "high"},
                "use_latest_alias": True,
            }
        }
    }


def test_missing_model_falls_back_to_concrete_fallback(tmp_path: Path):
    resolver = ModelResolver(results_root=tmp_path)
    result = resolver.resolve_all("run_test", _models_config())
    rm = result["resolved"]["openai_frontier"]
    # Offline + alias unresolved -> uses the concrete fallback.
    assert rm.resolved_model == "openai/gpt-5.5-pro"
    assert rm.available is True


def test_latest_alias_resolved_once_and_saved(tmp_path: Path):
    resolver = ModelResolver(results_root=tmp_path)
    result = resolver.resolve_all("run_test", _models_config())
    assert Path(result["model_catalog_snapshot_path"]).exists()
    assert Path(result["resolved_models_path"]).exists()


def test_resolved_model_does_not_change_during_run(tmp_path: Path):
    resolver = ModelResolver(results_root=tmp_path)
    result = resolver.resolve_all("run_test", _models_config())
    rm = result["resolved"]["openai_frontier"]
    client = OpenRouterClient(
        requested_model=rm.requested_model,
        resolved_model=rm.resolved_model,
        provider_family=rm.provider_family,
        pricing=rm.pricing,
    )
    r1 = client.generate([{"role": "user", "content": "hi"}], metadata={"json_output": True})
    r2 = client.generate([{"role": "user", "content": "again"}], metadata={"json_output": True})
    assert r1.resolved_model == r2.resolved_model == "openai/gpt-5.5-pro"
    assert r1.requested_model == "~openai/gpt-latest"


def test_unsupported_parameters_are_stripped():
    client = OpenRouterClient(
        requested_model="x",
        resolved_model="openai/gpt-5.5-pro",
        supported_parameters=["max_tokens"],  # tools/response_format/temperature unsupported
    )
    params, stripped = client._build_params(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f"}}],
        want_json=True,
    )
    assert "tools" in stripped
    assert "response_format" in stripped
    assert "temperature" in stripped
    assert "tools" not in params
    assert "response_format" not in params
    assert "max_tokens" in params


def test_strict_mode_disables_fallback_routing():
    client = OpenRouterClient(
        requested_model="x",
        resolved_model="openai/gpt-5.5-pro",
        supported_parameters=["max_tokens", "temperature"],
        provider_policy={"allow_fallbacks": False, "require_parameters": True,
                         "data_collection": "deny"},
    )
    params, _ = client._build_params([{"role": "user", "content": "hi"}], None, False)
    provider = params["extra_body"]["provider"]
    assert provider["allow_fallbacks"] is False
    assert provider["require_parameters"] is True
    assert provider["data_collection"] == "deny"


def test_cost_uses_snapshot_pricing():
    # Normalized per-token pricing as stored in resolved_models.yaml.
    pricing = {"prompt": 0.000003, "completion": 0.000015, "cached_input": 0.0000003,
               "reasoning": 0.000015}
    cost = estimate_cost_usd(
        input_tokens=1000, output_tokens=500, cached_input_tokens=200,
        reasoning_tokens=100, pricing=pricing,
    )
    # non-cached input = 800 * 3e-6 = 0.0024
    # cached = 200 * 3e-7 = 0.00006
    # output = 500 * 1.5e-5 = 0.0075
    # reasoning = 100 * 1.5e-5 = 0.0015
    expected = 0.0024 + 0.00006 + 0.0075 + 0.0015
    assert abs(cost - expected) < 1e-9


def test_offline_client_flags_unavailable_reason():
    client = OpenRouterClient(requested_model="x", resolved_model="openai/gpt-5.5-pro")
    resp = client.generate([{"role": "user", "content": "hi"}], metadata={"json_output": True})
    assert resp.model_unavailable_reason == "offline_mock_no_api_key"
