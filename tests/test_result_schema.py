"""Result schema validation tests."""

from __future__ import annotations

from benchmark.schemas import TaskResult


def _result(**overrides):
    base = dict(
        run_id="r1",
        benchmark="memoryarena",
        domain="bundled_shopping",
        memory_system="mem0",
        model="openai_frontier",
        task_id="t1",
        seed=1,
        requested_model="~openai/gpt-latest",
        resolved_model="openai/gpt-5.5-pro",
        provider_family="openai",
    )
    base.update(overrides)
    return TaskResult(**base)


def test_result_stores_requested_and_resolved_model():
    r = _result()
    assert r.requested_model == "~openai/gpt-latest"
    assert r.resolved_model == "openai/gpt-5.5-pro"
    assert r.provider_family == "openai"


def test_result_has_all_required_fields():
    r = _result()
    data = r.model_dump()
    for field in [
        "success", "score", "progress_score", "rule_compliance", "consistency_score",
        "ux_score", "latency_ms", "turn_count", "tool_call_count", "input_tokens",
        "output_tokens", "cached_input_tokens", "total_tokens", "estimated_cost_usd",
        "retrieved_memory_count", "written_memory_count", "memory_store_size_bytes",
        "memory_retrieval_latency_ms", "memory_write_latency_ms", "memory_relevance_score",
        "memory_correctness_score", "memory_usage_score", "memory_utility_gain_vs_stateless",
        "failure_category", "error", "reasoning_tokens", "model_unavailable_reason",
        "practical_correctness", "usefulness", "dfx_constraint_adherence",
        "reasoning_clarity", "engineering_applicability",
    ]:
        assert field in data


def test_flatten_prefixes_official_metrics():
    r = _result(official_metrics={"STATE_Bench_ux_score": 0.9})
    flat = r.flatten()
    assert "official_STATE_Bench_ux_score" in flat
    assert flat["official_STATE_Bench_ux_score"] == 0.9
    assert "official_metrics" not in flat


def test_dfx_expert_fields_default_none():
    r = _result()
    assert r.practical_correctness is None
    assert r.engineering_applicability is None
