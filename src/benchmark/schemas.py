"""Pydantic schemas for the benchmark harness.

These models define the strict, typed contracts used across the harness:
memory namespace + records, LLM responses, benchmark tasks, run results, and
the full persisted task result schema (including OpenRouter and DFx fields).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Memory namespace + records
# ---------------------------------------------------------------------------
class MemoryNamespace(BaseModel):
    """Composite key guaranteeing isolation across every experimental axis.

    No memory read or write may cross any of these fields. If one model could
    read memory written by another model (or another task/seed/benchmark), the
    benchmark would be invalid.
    """

    run_id: str
    benchmark: str
    domain: str
    memory_system: str
    model: str
    task_id: str
    seed: int

    def key(self) -> str:
        """Return the deterministic, path-safe composite namespace key."""
        parts = [
            self.run_id,
            self.benchmark,
            self.domain,
            self.memory_system,
            self.model,
            self.task_id,
            str(self.seed),
        ]
        # Sanitize each part so it is safe for filesystem paths and store keys.
        safe = [str(p).replace("/", "_").replace("..", "_").strip() for p in parts]
        return "/".join(safe)

    def flat_key(self) -> str:
        """Single-token key (no slashes) for backends that need a flat id."""
        return self.key().replace("/", "::")


class MemoryRecord(BaseModel):
    """A single memory entry."""

    id: str
    text: str
    source: str = "agent"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    namespace: MemoryNamespace | None = None


class MemoryWriteResult(BaseModel):
    """Outcome of a memory write/update operation."""

    success: bool
    memory_id: str | None = None
    error: str | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------
class LLMResponse(BaseModel):
    """Normalized LLM response with full usage + routing metadata."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)

    # OpenRouter routing metadata.
    requested_model: str | None = None
    resolved_model: str | None = None
    provider_family: str | None = None
    supported_parameters: list[str] = Field(default_factory=list)
    stripped_parameters: list[str] = Field(default_factory=list)
    estimated_cost_usd: float = 0.0
    openrouter_generation_id: str | None = None
    openrouter_request_id: str | None = None
    model_unavailable_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Benchmark task + run/score results
# ---------------------------------------------------------------------------
class Subtask(BaseModel):
    """A single subtask within a benchmark task sequence."""

    id: str
    question: str
    expected_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkTask(BaseModel):
    """A benchmark task, possibly composed of ordered subtasks."""

    id: str
    benchmark: str
    domain: str
    prompt: str = ""
    subtasks: list[Subtask] = Field(default_factory=list)
    backgrounds: list[str] = Field(default_factory=list)
    expected_answers: list[str] = Field(default_factory=list)
    gold_memory_facts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubtaskRun(BaseModel):
    """Execution record for one subtask."""

    subtask_id: str
    iteration: int = 0
    output: str = ""
    expected_answer: str | None = None
    used_memory_ids: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""
    correct: bool = False
    score: float = 0.0
    retrieved_memory_count: int = 0
    written_memory_count: int = 0
    llm: LLMResponse | None = None
    error: str | None = None


class TaskRunResult(BaseModel):
    """Raw execution result of a single task run (one iteration/seed)."""

    namespace: MemoryNamespace
    task_id: str
    iteration: int = 0
    subtask_runs: list[SubtaskRun] = Field(default_factory=list)

    # Aggregated execution stats for the task run.
    turn_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0

    # LLM-judge usage (relevance/usage/consistency scoring). Tracked separately so
    # the agent metrics above stay clean and total cost is not understated.
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_reasoning_tokens: int = 0
    judge_cost_usd: float = 0.0

    retrieved_memory_count: int = 0
    written_memory_count: int = 0
    memory_store_size_bytes: int = 0
    memory_retrieval_latency_ms: float = 0.0
    memory_write_latency_ms: float = 0.0

    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskScore(BaseModel):
    """Scored outcome for a task run."""

    success: bool = False
    score: float = 0.0
    progress_score: float = 0.0
    rule_compliance: float | None = None
    consistency_score: float | None = None
    ux_score: float | None = None

    memory_relevance_score: float | None = None
    memory_correctness_score: float | None = None
    memory_usage_score: float | None = None
    memory_utility_gain_vs_stateless: float | None = None

    failure_category: str | None = None

    # Official, clearly-namespaced benchmark metrics.
    official_metrics: dict[str, Any] = Field(default_factory=dict)


class FailureCategory(str, Enum):
    MISSING_MEMORY_WRITE = "missing_memory_write"
    IRRELEVANT_MEMORY_WRITE = "irrelevant_memory_write"
    FAILED_RETRIEVAL = "failed_retrieval"
    INCORRECT_OR_OUTDATED_RETRIEVAL = "incorrect_or_outdated_retrieval"
    INCORRECT_USE_OF_RETRIEVED_MEMORY = "incorrect_use_of_retrieved_memory"
    OVER_RELIANCE_ON_PRIOR_FEEDBACK = "over_reliance_on_prior_feedback"
    INCONSISTENCY_ACROSS_ITERATIONS = "inconsistency_across_iterations"
    MODEL_REASONING_ERROR = "model_reasoning_error"
    TOOL_USE_ERROR = "tool_use_error"
    BENCHMARK_ENVIRONMENT_ERROR = "benchmark_environment_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Full persisted task result
# ---------------------------------------------------------------------------
class TaskResult(BaseModel):
    """The complete, persisted per-task result row.

    This is the canonical schema written to per-memory-system folders, SQLite,
    and Parquet. It is the unit aggregated by scripts/aggregate_results.py.
    """

    # Identity / namespace.
    run_id: str
    benchmark: str
    domain: str
    memory_system: str
    model: str
    task_id: str
    seed: int
    iteration_count: int = 1

    # Core scores.
    success: bool = False
    score: float = 0.0
    progress_score: float = 0.0
    rule_compliance: float | None = None
    consistency_score: float | None = None
    ux_score: float | None = None

    # Efficiency.
    latency_ms: float = 0.0
    turn_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    # LLM-judge usage (separate from the agent cost above).
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_reasoning_tokens: int = 0
    judge_cost_usd: float = 0.0
    # Agent + judge combined (the true total spend for this task row).
    total_cost_usd: float = 0.0

    # Memory behavior.
    retrieved_memory_count: int = 0
    written_memory_count: int = 0
    memory_store_size_bytes: int = 0
    memory_retrieval_latency_ms: float = 0.0
    memory_write_latency_ms: float = 0.0
    memory_relevance_score: float | None = None
    memory_correctness_score: float | None = None
    memory_usage_score: float | None = None
    memory_utility_gain_vs_stateless: float | None = None
    # Backend provenance: whether the real memory backend was available this run,
    # and whether operations were actually served by the local fallback store.
    # ``memory_uses_local_fallback=True`` means this row does NOT measure the
    # real memory system and should be excluded/annotated in cross-system analysis.
    memory_backend_available: bool = True
    memory_uses_local_fallback: bool = False

    # Failure analysis.
    failure_category: str | None = None
    error: str | None = None

    # OpenRouter / model provenance (reproducibility).
    requested_model: str | None = None
    resolved_model: str | None = None
    provider_family: str | None = None
    benchmark_tier: str | None = None
    openrouter_model_catalog_created_at: str | None = None
    openrouter_model_pricing: dict[str, Any] = Field(default_factory=dict)
    openrouter_supported_parameters: list[str] = Field(default_factory=list)
    openrouter_provider_policy: dict[str, Any] = Field(default_factory=dict)
    openrouter_generation_id: str | None = None
    openrouter_request_id: str | None = None
    reasoning_tokens: int = 0
    model_unavailable_reason: str | None = None

    # DFx industrial expert-evaluation fields (empty unless expert files exist).
    practical_correctness: float | None = None
    usefulness: float | None = None
    dfx_constraint_adherence: float | None = None
    reasoning_clarity: float | None = None
    engineering_applicability: float | None = None

    # Official benchmark metrics, clearly namespaced (e.g. official_STATE_Bench_*).
    official_metrics: dict[str, Any] = Field(default_factory=dict)

    def flatten(self) -> dict[str, Any]:
        """Flatten to a tabular dict suitable for pandas/Parquet/CSV."""
        data = self.model_dump()
        # Flatten official metrics with an explicit, unambiguous prefix.
        official = data.pop("official_metrics", {}) or {}
        for key, value in official.items():
            data[f"official_{key}"] = value
        # JSON-encode remaining nested structures for tabular storage.
        for nested in (
            "openrouter_model_pricing",
            "openrouter_provider_policy",
            "openrouter_supported_parameters",
        ):
            value = data.get(nested)
            if isinstance(value, (dict, list)):
                import json

                data[nested] = json.dumps(value)
        return data


class RunMetadata(BaseModel):
    """Run-level metadata captured once per run."""

    run_id: str
    memory_system: str
    created_at: datetime = Field(default_factory=_utcnow)
    models: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    model_catalog_snapshot_path: str | None = None
    resolved_models_path: str | None = None
    model_resolution_time: str | None = None
    openrouter_base_url: str | None = None
    # Memory backend provenance captured at run start (runtime fallbacks are
    # additionally recorded per-row on TaskResult.memory_uses_local_fallback).
    memory_backend_available: bool = True
    memory_uses_local_fallback: bool = False
    memory_unavailable_reason: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
