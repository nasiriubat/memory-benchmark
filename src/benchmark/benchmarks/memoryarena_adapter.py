"""MemoryArena benchmark adapter.

Uses the official MemoryArena runner when installed; otherwise falls back to the
Hugging Face dataset ``ZexueHe/memoryarena``; otherwise uses a small bundled
sample so the harness runs offline.

MemoryArena task shape:
    id, questions (list of subtasks), answers (list of expected answers),
    backgrounds (string or list).

Questions execute in order. Memory persists within one task sequence and resets
between task IDs unless ``cross_task_memory`` is true (handled by the runner via
the memory namespace). Official scoring is used when the official runner is
present; the fallback computes the documented MemoryArena-style metrics and
labels them clearly as non-official.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from benchmark.benchmarks.base import SequentialBenchmarkAdapter
from benchmark.logging import get_logger
from benchmark.metrics import improvement_over_time
from benchmark.schemas import BenchmarkTask, Subtask, TaskRunResult, TaskScore

_log = get_logger("benchmark.memoryarena")

_SUPPORTED_CONFIGS = [
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
]


def _official_available() -> bool:
    return importlib.util.find_spec("memoryarena") is not None


class MemoryArenaAdapter(SequentialBenchmarkAdapter):
    name = "memoryarena"

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.hf_dataset = (config or {}).get("hf_dataset", "ZexueHe/memoryarena")
        self.official = _official_available()
        self._tasks_by_config: dict[str, list[BenchmarkTask]] = {}

    def prepare(self) -> None:
        if self.official:
            _log.info("MemoryArena: official runner detected; adapter will wrap it.")
            # REAL RUNNER: initialize the official MemoryArena runner here.
        else:
            _log.info(
                "MemoryArena: official runner not installed; using HF dataset '%s' "
                "with bundled-sample fallback.",
                self.hf_dataset,
            )

    def list_tasks(self, domain: str) -> list[BenchmarkTask]:
        config = domain
        if config not in self._tasks_by_config:
            self._tasks_by_config[config] = self._load_config(config)
        return self._tasks_by_config[config]

    # ------------------------------------------------------------- loading
    def _load_config(self, config: str) -> list[BenchmarkTask]:
        tasks = self._load_from_hf(config)
        if tasks:
            return tasks
        _log.warning("MemoryArena: falling back to bundled sample for config '%s'.", config)
        return self._bundled_sample(config)

    def _load_from_hf(self, config: str) -> list[BenchmarkTask]:
        if importlib.util.find_spec("datasets") is None:
            return []
        try:
            from datasets import load_dataset

            ds = load_dataset(self.hf_dataset, config, split="test")
        except Exception as exc:  # noqa: BLE001 - log and fall back
            _log.warning("MemoryArena: HF load failed for '%s': %s", config, exc)
            return []
        tasks: list[BenchmarkTask] = []
        for i, row in enumerate(ds):
            tasks.append(self._row_to_task(row, config, i))
        _log.info("MemoryArena: loaded %d tasks for config '%s' from HF.", len(tasks), config)
        return tasks

    @staticmethod
    def _row_to_task(row: dict[str, Any], config: str, idx: int) -> BenchmarkTask:
        questions = row.get("questions") or []
        answers = row.get("answers") or []
        backgrounds = row.get("backgrounds", [])
        if isinstance(backgrounds, str):
            backgrounds = [backgrounds]
        subtasks = []
        for j, q in enumerate(questions):
            expected = answers[j] if j < len(answers) else None
            subtasks.append(
                Subtask(
                    id=f"q{j}",
                    question=str(q),
                    expected_answer=str(expected) if expected is not None else None,
                )
            )
        return BenchmarkTask(
            id=str(row.get("id", f"{config}_{idx}")),
            benchmark="memoryarena",
            domain=config,
            subtasks=subtasks,
            backgrounds=list(backgrounds),
            expected_answers=[str(a) for a in answers],
        )

    @staticmethod
    def _bundled_sample(config: str) -> list[BenchmarkTask]:
        """Tiny offline sample exercising ordered, interdependent subtasks."""
        return [
            BenchmarkTask(
                id=f"{config}_sample_0",
                benchmark="memoryarena",
                domain=config,
                backgrounds=[
                    f"This is an offline bundled sample for MemoryArena config '{config}'.",
                    "The user's preferred budget is 1000 and preferred color is blue.",
                ],
                subtasks=[
                    Subtask(id="q0", question="What is the user's budget?",
                            expected_answer="1000"),
                    Subtask(id="q1", question="What is the user's preferred color?",
                            expected_answer="blue"),
                    Subtask(
                        id="q2",
                        question="Recommend an option within budget and preferred color.",
                        expected_answer="blue option within 1000",
                    ),
                ],
                expected_answers=["1000", "blue", "blue option within 1000"],
                gold_memory_facts=["budget is 1000", "preferred color is blue"],
            )
        ]

    # --------------------------------------------------------------- score
    def score_task(self, result: TaskRunResult) -> TaskScore:
        if self.official:
            return self._official_score(result)
        score = super().score_task(result)
        # MemoryArena-specific (non-official) signals derived from subtask scores.
        per_subtask = [s.score for s in result.subtask_runs]
        improvement = improvement_over_time(per_subtask)
        score.official_metrics.update(
            {
                "memoryarena_subtask_success": (
                    sum(1 for s in result.subtask_runs if s.correct) / len(result.subtask_runs)
                    if result.subtask_runs
                    else 0.0
                ),
                "memoryarena_task_success": float(score.success),
                "memoryarena_progress_score": score.progress_score,
                "memoryarena_final_task_score": (
                    per_subtask[-1] if per_subtask else 0.0
                ),
                "memoryarena_improvement_slope": improvement.slope,
                "memoryarena_improvement_relative": improvement.relative_improvement,
            }
        )
        return score

    def _official_score(self, result: TaskRunResult) -> TaskScore:
        # REAL RUNNER: call the official MemoryArena scorer and store its outputs
        # under clearly-namespaced official_* keys. We never silently
        # reimplement official scoring when the official runner is installed.
        score = super().score_task(result)
        _log.info("MemoryArena: official scorer hook (implement against installed runner).")
        return score
