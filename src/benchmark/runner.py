"""Benchmark runner.

Runs ONE memory system across the configured models, benchmarks, and domains.
Memory namespaces are always isolated by
run_id/benchmark/domain/memory_system/model/task_id/seed. Repeated runs vary the
seed, so iterations never contaminate each other.

The run continues after individual task failures (errors are logged with
context); it only aborts early when ``fail_fast`` is true.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.agents.simple_tool_agent import SimpleToolAgent
from benchmark.benchmarks.base import SequentialBenchmarkAdapter
from benchmark.benchmarks.dfx_adapter import DFxAdapter
from benchmark.benchmarks.memoryarena_adapter import MemoryArenaAdapter
from benchmark.benchmarks.state_bench_adapter import StateBenchAdapter
from benchmark.config import AppConfig
from benchmark.llms.model_resolver import ResolvedModel
from benchmark.llms.openrouter_client import OpenRouterClient
from benchmark.logging import EventLogger, get_logger
from benchmark.memory.base import MemoryAdapter
from benchmark.memory.namespace import make_namespace, raw_log_path
from benchmark.schemas import (
    BenchmarkTask,
    FailureCategory,
    MemoryNamespace,
    TaskResult,
    TaskRunResult,
    TaskScore,
)
from benchmark.storage.results_store import ResultsStore

_log = get_logger("benchmark.runner")

_BASE_SEED = 1000

_ADAPTERS: dict[str, type[SequentialBenchmarkAdapter]] = {
    "state_bench": StateBenchAdapter,
    "memoryarena": MemoryArenaAdapter,
    "dfx": DFxAdapter,
}

# Map config benchmark keys to adapter keys + their domain/config list key.
_BENCHMARK_DOMAIN_KEY = {
    "state_bench": "domains",
    "memoryarena": "configs",
    "dfx": "domains",
}


@dataclass
class Runner:
    config: AppConfig
    run_id: str
    memory_system: str
    memory: MemoryAdapter
    resolved_models: dict[str, ResolvedModel]
    results_root: Path
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.store = ResultsStore(self.results_root, self.memory_system, self.run_id)
        self.agent = SimpleToolAgent()
        self._judge_client: OpenRouterClient | None = None

    # ------------------------------------------------------------------ run
    def run(self) -> list[TaskResult]:
        results: list[TaskResult] = []
        available_models = {
            slot: rm for slot, rm in self.resolved_models.items() if rm.available
        }
        if not available_models:
            _log.warning("No available models resolved; nothing to run.")
            return results

        self._judge_client = self._build_judge_client(available_models)

        model_par = max(1, self.config.run.model_parallelism)

        def run_model(item: tuple[str, ResolvedModel]) -> list[TaskResult]:
            slot, rm = item
            return self._run_for_model(slot, rm)

        if model_par > 1 and len(available_models) > 1:
            with ThreadPoolExecutor(max_workers=model_par) as pool:
                for partial in pool.map(run_model, available_models.items()):
                    results.extend(partial)
        else:
            for item in available_models.items():
                results.extend(run_model(item))

        self.store.flush_tabular()
        self._print_summary(results)
        return results

    # ---------------------------------------------------------- per-model
    def _run_for_model(self, slot: str, rm: ResolvedModel) -> list[TaskResult]:
        client = self._build_client(rm)
        out: list[TaskResult] = []
        # Optional cost guardrail: restrict to a subset of benchmarks.
        only = {b.strip() for b in os.environ.get("BENCHMARKS", "").split(",") if b.strip()}
        for bench_name, bench_spec in self.config.enabled_benchmarks().items():
            if bench_name not in _ADAPTERS:
                _log.warning("No adapter for benchmark '%s'; skipping.", bench_name)
                continue
            if only and bench_name not in only:
                continue
            domains = bench_spec.get(_BENCHMARK_DOMAIN_KEY.get(bench_name, "domains"), [])
            for domain in domains:
                out.extend(self._run_domain(bench_name, bench_spec, domain, slot, rm, client))
        return out

    def _run_domain(
        self,
        bench_name: str,
        bench_spec: dict[str, Any],
        domain: str,
        slot: str,
        rm: ResolvedModel,
        client: OpenRouterClient,
    ) -> list[TaskResult]:
        # Fresh adapter per (model, benchmark, domain) to avoid shared state.
        adapter_cfg = dict(bench_spec)
        adapter_cfg["retrieve_top_k"] = self.config.run.retrieve_top_k
        adapter = _ADAPTERS[bench_name](config=adapter_cfg)
        adapter.prepare()

        # One EventLogger per (run_id, memory_system, model, benchmark, domain).
        ns_for_log = make_namespace(
            run_id=self.run_id,
            benchmark=bench_name,
            domain=domain,
            memory_system=self.memory_system,
            model=slot,
            task_id="_",
            seed=0,
        )
        log_path = raw_log_path(self.results_root, ns_for_log)
        event_logger = EventLogger(log_path, ns_for_log)
        adapter.set_context(event_logger=event_logger, judge_model=self._judge_client)

        event_logger.emit(
            "run_started",
            {"memory_system": self.memory_system, "model": slot, "benchmark": bench_name,
             "domain": domain, "resolved_model": rm.resolved_model},
        )

        try:
            tasks = adapter.list_tasks(domain)
        except Exception as exc:  # noqa: BLE001
            event_logger.error(f"list_tasks failed: {exc}", context={"domain": domain})
            if self.config.run.fail_fast:
                raise
            return []

        # Optional cost guardrail: cap tasks per domain (profile or MAX_TASKS_PER_DOMAIN).
        max_tasks = self.config.run.max_tasks_per_domain
        if max_tasks > 0:
            tasks = tasks[:max_tasks]

        num_runs = self.config.run.num_runs
        task_par = max(1, self.config.run.task_parallelism)

        jobs: list[tuple[BenchmarkTask, int]] = [
            (task, it) for task in tasks for it in range(num_runs)
        ]

        out: list[TaskResult] = []

        def run_job(job: tuple[BenchmarkTask, int]) -> TaskResult | None:
            task, iteration = job
            return self._run_task_iteration(
                adapter=adapter,
                client=client,
                event_logger=event_logger,
                bench_name=bench_name,
                domain=domain,
                slot=slot,
                rm=rm,
                task=task,
                iteration=iteration,
                num_runs=num_runs,
            )

        if task_par > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=task_par) as pool:
                for res in pool.map(run_job, jobs):
                    if res is not None:
                        out.append(res)
                        self.store.append(res)
        else:
            for job in jobs:
                res = run_job(job)
                if res is not None:
                    out.append(res)
                    self.store.append(res)

        event_logger.emit(
            "run_finished",
            {"memory_system": self.memory_system, "model": slot, "benchmark": bench_name,
             "domain": domain, "tasks": len(tasks), "results": len(out)},
        )
        return out

    # ------------------------------------------------------ task iteration
    def _run_task_iteration(
        self,
        *,
        adapter: SequentialBenchmarkAdapter,
        client: OpenRouterClient,
        event_logger: EventLogger,
        bench_name: str,
        domain: str,
        slot: str,
        rm: ResolvedModel,
        task: BenchmarkTask,
        iteration: int,
        num_runs: int,
    ) -> TaskResult | None:
        seed = _BASE_SEED + iteration
        ns = self._build_memory_namespace(bench_name, domain, slot, task.id, seed)
        try:
            # Each (task, seed) namespace starts clean to guarantee isolation.
            adapter_mem_reset = not (
                self.config.run.cross_task_memory and iteration == 0
            )
            if adapter_mem_reset:
                self.memory.reset(ns)
            run_result: TaskRunResult = adapter.run_task(task, self.agent, self.memory, client, ns)
            score: TaskScore = adapter.score_task(run_result)
            result = self._assemble_result(
                bench_name=bench_name,
                domain=domain,
                slot=slot,
                rm=rm,
                task=task,
                seed=seed,
                iteration_count=num_runs,
                run_result=run_result,
                score=score,
            )
            return result
        except Exception as exc:  # noqa: BLE001 - never kill the whole benchmark
            event_logger.error(
                f"task run failed: {exc}",
                context={"task_id": task.id, "model": slot, "benchmark": bench_name},
                task_id=task.id,
                iteration=iteration,
            )
            if self.config.run.fail_fast:
                raise
            return self._error_result(bench_name, domain, slot, rm, task, seed, num_runs, str(exc))

    def _build_memory_namespace(
        self, bench_name: str, domain: str, slot: str, task_id: str, seed: int
    ) -> MemoryNamespace:
        # Default: per-task isolation. cross_task_memory shares memory across
        # task IDs within (model, benchmark, domain, seed) via a shared task key,
        # while results/logs still record the real task id. All other isolation
        # axes are preserved.
        effective_task_id = task_id
        if self.config.run.cross_task_memory:
            effective_task_id = f"__xtask__:{bench_name}:{domain}"
        return make_namespace(
            run_id=self.run_id,
            benchmark=bench_name,
            domain=domain,
            memory_system=self.memory_system,
            model=slot,
            task_id=effective_task_id,
            seed=seed,
        )

    # ----------------------------------------------------------- assembly
    def _assemble_result(
        self,
        *,
        bench_name: str,
        domain: str,
        slot: str,
        rm: ResolvedModel,
        task: BenchmarkTask,
        seed: int,
        iteration_count: int,
        run_result: TaskRunResult,
        score: TaskScore,
    ) -> TaskResult:
        failure = None if score.success else self._classify_failure(run_result, score)
        return TaskResult(
            run_id=self.run_id,
            benchmark=bench_name,
            domain=domain,
            memory_system=self.memory_system,
            model=slot,
            task_id=task.id,
            seed=seed,
            iteration_count=iteration_count,
            success=score.success,
            score=score.score,
            progress_score=score.progress_score,
            rule_compliance=score.rule_compliance,
            consistency_score=score.consistency_score,
            ux_score=score.ux_score,
            latency_ms=run_result.latency_ms,
            turn_count=run_result.turn_count,
            tool_call_count=run_result.tool_call_count,
            input_tokens=run_result.input_tokens,
            output_tokens=run_result.output_tokens,
            cached_input_tokens=run_result.cached_input_tokens,
            total_tokens=run_result.input_tokens + run_result.output_tokens,
            estimated_cost_usd=run_result.estimated_cost_usd,
            retrieved_memory_count=run_result.retrieved_memory_count,
            written_memory_count=run_result.written_memory_count,
            memory_store_size_bytes=run_result.memory_store_size_bytes,
            memory_retrieval_latency_ms=run_result.memory_retrieval_latency_ms,
            memory_write_latency_ms=run_result.memory_write_latency_ms,
            memory_relevance_score=score.memory_relevance_score,
            memory_correctness_score=score.memory_correctness_score,
            memory_usage_score=score.memory_usage_score,
            # Gain-vs-stateless is computed during aggregation (needs the matched
            # stateless baseline). For the stateless system itself the gain is 0.
            memory_utility_gain_vs_stateless=0.0 if self.memory_system == "stateless" else None,
            failure_category=failure,
            error=run_result.error,
            requested_model=rm.requested_model,
            resolved_model=rm.resolved_model,
            provider_family=rm.provider_family,
            benchmark_tier=rm.benchmark_tier,
            openrouter_model_catalog_created_at=rm.catalog_created_at,
            openrouter_model_pricing=rm.pricing,
            openrouter_supported_parameters=rm.supported_parameters,
            openrouter_provider_policy=self.config.provider_policy,
            reasoning_tokens=run_result.reasoning_tokens,
            official_metrics=score.official_metrics,
        )

    def _error_result(
        self, bench_name: str, domain: str, slot: str, rm: ResolvedModel,
        task: BenchmarkTask, seed: int, iteration_count: int, error: str,
    ) -> TaskResult:
        return TaskResult(
            run_id=self.run_id,
            benchmark=bench_name,
            domain=domain,
            memory_system=self.memory_system,
            model=slot,
            task_id=task.id,
            seed=seed,
            iteration_count=iteration_count,
            success=False,
            failure_category=FailureCategory.UNKNOWN.value,
            error=error,
            requested_model=rm.requested_model,
            resolved_model=rm.resolved_model,
            provider_family=rm.provider_family,
            benchmark_tier=rm.benchmark_tier,
        )

    @staticmethod
    def _classify_failure(run_result: TaskRunResult, score: TaskScore) -> str:
        """Heuristic failure categorization (paper Section III-K)."""
        if run_result.error:
            return FailureCategory.BENCHMARK_ENVIRONMENT_ERROR.value
        if run_result.retrieved_memory_count == 0 and run_result.written_memory_count == 0:
            # No memory activity at all on a failed task.
            return FailureCategory.MISSING_MEMORY_WRITE.value
        if run_result.retrieved_memory_count == 0:
            return FailureCategory.FAILED_RETRIEVAL.value
        if score.memory_correctness_score is not None and score.memory_correctness_score < 0.5:
            return FailureCategory.INCORRECT_OR_OUTDATED_RETRIEVAL.value
        if score.memory_usage_score is not None and score.memory_usage_score < 0.3:
            return FailureCategory.INCORRECT_USE_OF_RETRIEVED_MEMORY.value
        if score.consistency_score is not None and score.consistency_score < 0.5:
            return FailureCategory.INCONSISTENCY_ACROSS_ITERATIONS.value
        return FailureCategory.MODEL_REASONING_ERROR.value

    # ------------------------------------------------------------- clients
    def _build_client(self, rm: ResolvedModel) -> OpenRouterClient:
        policy = self.config.provider_policy
        return OpenRouterClient(
            requested_model=rm.requested_model,
            resolved_model=rm.resolved_model,
            provider_family=rm.provider_family,
            benchmark_tier=rm.benchmark_tier,
            supported_parameters=rm.supported_parameters,
            pricing=rm.pricing,
            provider_policy=policy,
            temperature=rm.temperature,
            max_output_tokens=rm.max_output_tokens,
            reasoning=rm.reasoning,
            base_url=self.config.openrouter_policy.get("base_url"),
        )

    def _build_judge_client(
        self, available_models: dict[str, ResolvedModel]
    ) -> OpenRouterClient | None:
        judge_slot = (
            self.config.benchmark.get("evaluators", {}).get("llm_judge_model")
        )
        rm = available_models.get(judge_slot)
        if rm is None:
            # Fall back to any available model for judging.
            rm = next(iter(available_models.values()), None)
        if rm is None:
            return None
        return self._build_client(rm)

    # ------------------------------------------------------------- summary
    def _print_summary(self, results: list[TaskResult]) -> None:
        if not results:
            _log.info("No results produced for memory system '%s'.", self.memory_system)
            return
        total = len(results)
        successes = sum(1 for r in results if r.success)
        mean_score = sum(r.score for r in results) / total
        cost = sum(r.estimated_cost_usd for r in results)
        _log.info(
            "Summary [%s]: tasks=%d success_rate=%.3f mean_score=%.3f total_cost_usd=%.6f",
            self.memory_system, total, successes / total, mean_score, cost,
        )


def build_memory_adapter(memory_system: str, config: AppConfig) -> MemoryAdapter:
    """Instantiate the memory adapter for a memory system from config."""
    from benchmark.memory.registry import load_adapter

    return load_adapter(memory_system, config.memory_systems)
