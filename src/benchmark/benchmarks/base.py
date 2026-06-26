"""Base benchmark adapter + shared sequential task execution.

The :class:`BenchmarkAdapter` interface is intentionally small. Most benchmarks
(MemoryArena, DFx, STATE-Bench fallback) share the same execution shape: run a
task's subtasks in order, retrieving memory before each subtask and writing
memory after, with memory persisting within the task sequence. That shared logic
lives in :meth:`SequentialBenchmarkAdapter.run_task`.
"""

from __future__ import annotations

import abc
import time
import uuid
from typing import Any

from benchmark.agents.base_agent import Agent
from benchmark.evaluators.consistency import FactLedger, score_consistency
from benchmark.evaluators.exact_match import score_answer
from benchmark.evaluators.llm_judge import judge as judge_answer
from benchmark.evaluators.memory_utility import (
    memory_usage_score,
    retrieval_correctness_score,
    retrieval_relevance_score,
)
from benchmark.llms.base import LLMClient
from benchmark.logging import EventLogger
from benchmark.memory.base import MemoryAdapter
from benchmark.metrics import progress_score
from benchmark.schemas import (
    BenchmarkTask,
    LLMResponse,
    MemoryNamespace,
    MemoryRecord,
    SubtaskRun,
    TaskRunResult,
    TaskScore,
)


class _JudgeUsageTracker(LLMClient):
    """Wraps the judge LLM client to tally token usage + cost across all judge
    calls for one task. A fresh tracker is created per task in ``score_task``
    (which runs in that task's own thread), so accumulation needs no lock.
    The evaluator functions call ``model.generate`` unchanged.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        self.provider_family = inner.provider_family
        self.requested_model = inner.requested_model
        self.resolved_model = inner.resolved_model

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        resp = self._inner.generate(messages, tools=tools, metadata=metadata)
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.reasoning_tokens += resp.reasoning_tokens
        self.cost_usd += resp.estimated_cost_usd
        return resp


class BenchmarkAdapter(abc.ABC):
    """Abstract benchmark adapter."""

    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abc.abstractmethod
    def prepare(self) -> None:
        """Download/load datasets, build learnings, etc."""

    @abc.abstractmethod
    def list_tasks(self, domain: str) -> list[BenchmarkTask]:
        """Return the tasks for a domain/config."""

    @abc.abstractmethod
    def run_task(
        self,
        task: BenchmarkTask,
        agent: Agent,
        memory: MemoryAdapter,
        model: LLMClient,
        namespace: MemoryNamespace,
    ) -> TaskRunResult:
        """Execute a single task run."""

    @abc.abstractmethod
    def score_task(self, result: TaskRunResult) -> TaskScore:
        """Score a completed task run."""


class SequentialBenchmarkAdapter(BenchmarkAdapter):
    """Reusable adapter that runs ordered subtasks with within-task memory."""

    #: F1 threshold above which a subtask answer counts as correct.
    correct_threshold: float = 0.6
    #: Build a consistency fact ledger across subtasks (DFx/MemoryArena).
    build_ledger: bool = True
    #: Whether the agent writes memory after each subtask. STATE-Bench Agent
    #: Learning Track sets this False to expose a read-only retrieval hook.
    write_during_run: bool = True

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        event_logger: EventLogger | None = None,
        judge_model: LLMClient | None = None,
    ) -> None:
        super().__init__(config)
        self.event_logger = event_logger
        self.judge_model = judge_model
        # How subtask answers are graded: "token_f1" (deterministic) or
        # "llm_judge" (semantic, needed for MemoryArena's id/structured answers).
        self.answer_scorer = (self.config.get("answer_correctness_scorer") or "token_f1")
        self.answer_rubric = self.config.get("answer_correctness_rubric") or ""
        # Auxiliary per-subtask signals captured during run_task and consumed by
        # score_task (keyed by (task_id, seed, subtask_id, subtask_index) so
        # concurrent iterations/tasks never collide).
        self._signals: dict[tuple[str, int, str, int], dict[str, Any]] = {}

    def set_context(
        self, *, event_logger: EventLogger | None, judge_model: LLMClient | None
    ) -> None:
        self.event_logger = event_logger
        self.judge_model = judge_model

    # ----------------------------------------------------------------- run
    def run_task(
        self,
        task: BenchmarkTask,
        agent: Agent,
        memory: MemoryAdapter,
        model: LLMClient,
        namespace: MemoryNamespace,
    ) -> TaskRunResult:
        log = self.event_logger
        run_result = TaskRunResult(namespace=namespace, task_id=task.id, iteration=namespace.seed)
        ledger = FactLedger()
        memory.setup(namespace)

        if log:
            log.emit("task_started", {"task_id": task.id, "num_subtasks": len(task.subtasks)},
                     task_id=task.id)

        for index, subtask in enumerate(task.subtasks):
            sub_run = self._run_subtask(
                task=task,
                subtask_index=index,
                agent=agent,
                memory=memory,
                model=model,
                namespace=namespace,
                ledger=ledger,
            )
            run_result.subtask_runs.append(sub_run)
            self._accumulate(run_result, sub_run)

        if log:
            log.emit(
                "task_finished",
                {
                    "task_id": task.id,
                    "correct_subtasks": sum(1 for s in run_result.subtask_runs if s.correct),
                    "total_subtasks": len(run_result.subtask_runs),
                },
                task_id=task.id,
            )
        run_result.memory_store_size_bytes = memory.store_size_bytes(namespace)
        return run_result

    def _run_subtask(
        self,
        *,
        task: BenchmarkTask,
        subtask_index: int,
        agent: Agent,
        memory: MemoryAdapter,
        model: LLMClient,
        namespace: MemoryNamespace,
        ledger: FactLedger,
    ) -> SubtaskRun:
        subtask = task.subtasks[subtask_index]
        log = self.event_logger
        meta = {"namespace": namespace}
        if log:
            log.emit("subtask_started", {"question": subtask.question}, task_id=task.id,
                     subtask_id=subtask.id, iteration=subtask_index)

        # 1. Retrieve memory (logged even for stateless no-ops).
        top_k = int(self.config.get("retrieve_top_k", 3))
        retr_start = time.perf_counter()
        try:
            retrieved = memory.retrieve(subtask.question, top_k, meta)
        except Exception as exc:  # noqa: BLE001 - never kill the run on memory errors
            retrieved = []
            if log:
                log.error(f"memory.retrieve failed: {exc}", context={"task_id": task.id},
                          task_id=task.id, subtask_id=subtask.id)
        retr_latency = (time.perf_counter() - retr_start) * 1000.0
        if log:
            log.emit("memory_retrieved", {"count": len(retrieved), "top_k": top_k},
                     task_id=task.id, subtask_id=subtask.id, iteration=subtask_index)

        # 2. Agent acts.
        if log:
            log.emit("prompt_built", {"subtask_id": subtask.id}, task_id=task.id,
                     subtask_id=subtask.id, iteration=subtask_index)
        answer = agent.act(
            question=subtask.question,
            backgrounds=task.backgrounds,
            retrieved=retrieved,
            model=model,
            namespace=namespace,
            memory=memory,
            metadata=meta,
        )
        if log and answer.llm is not None:
            log.emit(
                "llm_response",
                {
                    "requested_model": answer.llm.requested_model,
                    "resolved_model": answer.llm.resolved_model,
                    "provider_family": answer.llm.provider_family,
                    "supported_parameters": answer.llm.supported_parameters,
                    "stripped_parameters": answer.llm.stripped_parameters,
                    "usage": {
                        "input_tokens": answer.llm.input_tokens,
                        "output_tokens": answer.llm.output_tokens,
                        "cached_input_tokens": answer.llm.cached_input_tokens,
                        "reasoning_tokens": answer.llm.reasoning_tokens,
                        "total_tokens": answer.llm.total_tokens,
                    },
                    "estimated_cost_usd": answer.llm.estimated_cost_usd,
                    "latency_ms": answer.llm.latency_ms,
                },
                task_id=task.id,
                subtask_id=subtask.id,
                iteration=subtask_index,
            )

        # 3. Score the subtask answer. token_f1 is deterministic token overlap;
        # llm_judge grades answer-vs-expected semantically (needed for answers
        # keyed on ids/structured formats the agent cannot reproduce, e.g.
        # MemoryArena ASINs). The judge's usage/cost is tallied per subtask.
        judge_usage = None
        if self.answer_scorer == "llm_judge" and self.judge_model is not None:
            judge_usage = _JudgeUsageTracker(self.judge_model)
            verdict = judge_answer(
                model=judge_usage,
                question=subtask.question,
                prediction=answer.final_answer,
                expected=subtask.expected_answer,
                rubric=self.answer_rubric,
            )
            sub_score = verdict.score
        else:
            sub_score = score_answer(answer.final_answer, subtask.expected_answer,
                                     f1_threshold=self.correct_threshold)
        correct = sub_score >= self.correct_threshold
        if log:
            log.emit("evaluator_score",
                     {"subtask_id": subtask.id, "score": sub_score, "correct": correct,
                      "scorer": self.answer_scorer,
                      "output": (answer.final_answer or "")[:1000],
                      "expected": (subtask.expected_answer or "")[:1000]},
                     task_id=task.id, subtask_id=subtask.id, iteration=subtask_index)

        # 4. Update the fact ledger and write memory after the subtask.
        if self.build_ledger:
            ledger.add_from_subtask(
                question=subtask.question,
                expected=subtask.expected_answer,
                metadata=subtask.metadata,
            )
        written = 0
        write_latency = 0.0
        if self.write_during_run:
            record = MemoryRecord(
                id=uuid.uuid4().hex,
                text=f"Q: {subtask.question}\nA: {answer.final_answer}",
                source="agent",
                metadata={"subtask_id": subtask.id, "reasoning_summary": answer.reasoning_summary},
                namespace=namespace,
            )
            w_start = time.perf_counter()
            try:
                write_res = memory.write(record, meta)
                if write_res.success and write_res.memory_id is not None:
                    written = 1
            except Exception as exc:  # noqa: BLE001
                if log:
                    log.error(f"memory.write failed: {exc}", context={"task_id": task.id},
                              task_id=task.id, subtask_id=subtask.id)
            write_latency = (time.perf_counter() - w_start) * 1000.0
            if log:
                log.emit("memory_written", {"written": written}, task_id=task.id,
                         subtask_id=subtask.id, iteration=subtask_index)
        if log:
            log.emit("subtask_finished", {"score": sub_score}, task_id=task.id,
                     subtask_id=subtask.id, iteration=subtask_index)

        sub_run = SubtaskRun(
            subtask_id=subtask.id,
            iteration=subtask_index,
            output=answer.final_answer,
            expected_answer=subtask.expected_answer,
            used_memory_ids=answer.used_memory_ids,
            reasoning_summary=answer.reasoning_summary,
            correct=correct,
            score=sub_score,
            retrieved_memory_count=len(retrieved),
            written_memory_count=written,
            llm=answer.llm,
            judge_input_tokens=judge_usage.input_tokens if judge_usage else 0,
            judge_output_tokens=judge_usage.output_tokens if judge_usage else 0,
            judge_reasoning_tokens=judge_usage.reasoning_tokens if judge_usage else 0,
            judge_cost_usd=judge_usage.cost_usd if judge_usage else 0.0,
        )
        # Stash extra signals for scoring on the subtask metadata.
        sub_run_meta = {
            "retrieved": retrieved,
            "gold_memory_facts": task.gold_memory_facts,
            "ledger": ledger,
            "retrieval_latency_ms": retr_latency,
            "write_latency_ms": write_latency,
        }
        self._signals[(task.id, namespace.seed, subtask.id, subtask_index)] = sub_run_meta
        return sub_run

    @staticmethod
    def _accumulate(run_result: TaskRunResult, sub: SubtaskRun) -> None:
        run_result.turn_count += 1
        run_result.tool_call_count += len(sub.llm.tool_calls) if sub.llm else 0
        run_result.retrieved_memory_count += sub.retrieved_memory_count
        run_result.written_memory_count += sub.written_memory_count
        # Per-subtask correctness-judge usage (memory-utility judge usage is
        # added separately in score_task).
        run_result.judge_input_tokens += sub.judge_input_tokens
        run_result.judge_output_tokens += sub.judge_output_tokens
        run_result.judge_reasoning_tokens += sub.judge_reasoning_tokens
        run_result.judge_cost_usd += sub.judge_cost_usd
        if sub.llm is not None:
            run_result.input_tokens += sub.llm.input_tokens
            run_result.output_tokens += sub.llm.output_tokens
            run_result.cached_input_tokens += sub.llm.cached_input_tokens
            run_result.reasoning_tokens += sub.llm.reasoning_tokens
            run_result.latency_ms += sub.llm.latency_ms
            run_result.estimated_cost_usd += sub.llm.estimated_cost_usd

    # ---------------------------------------------------------------- score
    def score_task(self, result: TaskRunResult) -> TaskScore:
        subs = result.subtask_runs
        total = len(subs)
        correct = sum(1 for s in subs if s.correct)
        ps = progress_score(correct, total)
        success = total > 0 and correct == total
        mean_score = sum(s.score for s in subs) / total if total else 0.0

        # Wrap the judge client so all judge LLM calls below have their token
        # usage + cost tallied into this task's judge_* fields (None when there is
        # no judge model, e.g. offline runs).
        judge = _JudgeUsageTracker(self.judge_model) if self.judge_model is not None else None

        # Memory utility signals (aggregate over subtasks that retrieved memory).
        relevance_vals: list[float] = []
        correctness_vals: list[float] = []
        usage_vals: list[float] = []
        consistency_vals: list[float] = []
        for s in subs:
            signals = self._signals.get(
                (result.task_id, result.namespace.seed, s.subtask_id, s.iteration), {}
            )
            retrieved = signals.get("retrieved", [])
            ledger = signals.get("ledger")
            if retrieved:
                relevance_vals.append(
                    retrieval_relevance_score(
                        query=s.output or "",
                        retrieved=retrieved,
                        gold_memory_facts=signals.get("gold_memory_facts"),
                        model=judge,
                    )
                )
                correctness_vals.append(
                    retrieval_correctness_score(retrieved=retrieved, ledger=ledger)
                )
                usage_vals.append(
                    memory_usage_score(
                        final_answer=s.output or "",
                        used_memory_ids=s.used_memory_ids,
                        retrieved=retrieved,
                        model=judge,
                    )
                )
            if ledger is not None:
                consistency_vals.append(
                    score_consistency(
                        output=s.output or "",
                        ledger=ledger,
                        model=judge,
                    )
                )
            result.memory_retrieval_latency_ms += signals.get("retrieval_latency_ms", 0.0)
            result.memory_write_latency_ms += signals.get("write_latency_ms", 0.0)

        # Record the judge's cumulative usage/cost for this task.
        if judge is not None:
            result.judge_input_tokens += judge.input_tokens
            result.judge_output_tokens += judge.output_tokens
            result.judge_reasoning_tokens += judge.reasoning_tokens
            result.judge_cost_usd += judge.cost_usd

        def _avg(xs: list[float]) -> float | None:
            return sum(xs) / len(xs) if xs else None

        return TaskScore(
            success=success,
            score=mean_score,
            progress_score=ps,
            consistency_score=_avg(consistency_vals),
            memory_relevance_score=_avg(relevance_vals),
            memory_correctness_score=_avg(correctness_vals),
            memory_usage_score=_avg(usage_vals),
        )
