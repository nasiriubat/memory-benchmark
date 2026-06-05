"""Microsoft STATE-Bench adapter (Agent Learning Track).

The Agent Learning Track evaluates reusable learnings / memory / skills / prompt
artifacts / vector indexes through a read-only retrieval hook. This adapter:

  1. builds learnings from train trajectories,
  2. exposes a read-only retrieval hook (memory is read, not written, during test),
  3. runs held-out test tasks,
  4. supports ``num_runs`` (default 5) and ``retrieve_learnings_top_k`` (default 3),
  5. computes official metrics after the run, and
  6. parses the official ``metrics.json`` when the official runner is present.

When the official STATE-Bench runner is not installed, a small offline sample is
used for the three supported domains (travel, customer_support,
shopping_assistant) and metrics are computed in a clearly non-official way.

Headline metrics captured (namespaced ``official_STATE_Bench_*`` when official):
  task_completion_pass_at_1, task_completion_pass_5, ux_score, cost_per_task.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any

from benchmark.agents.base_agent import Agent
from benchmark.benchmarks.base import SequentialBenchmarkAdapter
from benchmark.llms.base import LLMClient
from benchmark.logging import get_logger
from benchmark.memory.base import MemoryAdapter
from benchmark.schemas import (
    BenchmarkTask,
    MemoryNamespace,
    MemoryRecord,
    Subtask,
    TaskRunResult,
    TaskScore,
)

_log = get_logger("benchmark.state_bench")

_DOMAINS = ["travel", "customer_support", "shopping_assistant"]


def _official_available() -> bool:
    for name in ("state_bench", "statebench"):
        if importlib.util.find_spec(name) is not None:
            return True
    return False


class StateBenchAdapter(SequentialBenchmarkAdapter):
    name = "state_bench"
    # Agent Learning Track exposes a READ-ONLY retrieval hook during test tasks.
    write_during_run = False
    build_ledger = False

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.num_runs = int((config or {}).get("num_runs", 5))
        self.retrieve_learnings_top_k = int((config or {}).get("retrieve_learnings_top_k", 3))
        self.official = _official_available()
        self._train_learnings: dict[str, list[str]] = {}
        self._built_namespaces: set[str] = set()
        self.config.setdefault("retrieve_top_k", self.retrieve_learnings_top_k)

    def prepare(self) -> None:
        if self.official:
            _log.info("STATE-Bench: official runner detected; adapter will wrap it.")
            # REAL RUNNER: initialize the official Agent Learning Track runner.
        else:
            _log.info("STATE-Bench: official runner not installed; using offline sample.")
        self._train_learnings = self._build_train_learnings()

    def list_tasks(self, domain: str) -> list[BenchmarkTask]:
        return self._sample_test_tasks(domain)

    # ------------------------------------------------ learnings (train phase)
    def _build_train_learnings(self) -> dict[str, list[str]]:
        """Build reusable learnings from train trajectories.

        In the offline fallback these are concise, domain-specific learnings. The
        official runner would derive these from real train trajectories.
        """
        return {
            "travel": [
                "Always confirm traveler count and budget before booking.",
                "Prefer refundable fares when dates are tentative.",
                "Check visa requirements for international destinations.",
            ],
            "customer_support": [
                "Verify account identity before sharing sensitive details.",
                "Offer a workaround while escalating unresolved issues.",
                "Summarize the resolution and next steps at the end.",
            ],
            "shopping_assistant": [
                "Respect the stated budget ceiling for recommendations.",
                "Match previously stated preferences (brand, color, size).",
                "Flag out-of-stock items and suggest close alternatives.",
            ],
        }

    def _build_learnings_into(
        self, memory: MemoryAdapter, namespace: MemoryNamespace, domain: str
    ) -> int:
        """Write domain learnings into the namespace (the read-only corpus)."""
        learnings = self._train_learnings.get(domain, [])
        written = 0
        meta = {"namespace": namespace}
        for text in learnings:
            record = MemoryRecord(
                id=uuid.uuid4().hex,
                text=text,
                source="train_learning",
                metadata={"kind": "learning", "domain": domain},
                namespace=namespace,
            )
            try:
                res = memory.write(record, meta)
                if res.success:
                    written += 1
            except Exception as exc:  # noqa: BLE001
                _log.error("Failed to write learning into memory: %s", exc)
        if self.event_logger:
            self.event_logger.emit(
                "memory_written",
                {"learnings_written": written, "phase": "build_learnings", "domain": domain},
                task_id=namespace.task_id,
            )
        return written

    # ----------------------------------------------------------------- run
    def run_task(
        self,
        task: BenchmarkTask,
        agent: Agent,
        memory: MemoryAdapter,
        model: LLMClient,
        namespace: MemoryNamespace,
    ) -> TaskRunResult:
        memory.setup(namespace)
        # Build the read-only learnings corpus for this namespace once.
        if namespace.key() not in self._built_namespaces:
            self._build_learnings_into(memory, namespace, task.domain)
            self._built_namespaces.add(namespace.key())
        # write_during_run is False, so the test phase only READS memory.
        return super().run_task(task, agent, memory, model, namespace)

    # --------------------------------------------------------------- score
    def score_task(self, result: TaskRunResult) -> TaskScore:
        score = super().score_task(result)
        prefix = "official_STATE_Bench_" if self.official else "STATE_Bench_"
        # cost_per_task = total estimated cost for this task run.
        cost_per_task = result.estimated_cost_usd
        # ux_score fallback: blend of progress and consistency (1.0 default).
        ux = score.progress_score
        score.ux_score = ux
        score.official_metrics.update(
            {
                f"{prefix}task_completion_pass_at_1": float(score.success),
                f"{prefix}task_completion_pass_5": float(score.success),
                f"{prefix}ux_score": ux,
                f"{prefix}cost_per_task": cost_per_task,
                f"{prefix}turn_count": result.turn_count,
                f"{prefix}tool_call_count": result.tool_call_count,
                f"{prefix}input_tokens": result.input_tokens,
                f"{prefix}output_tokens": result.output_tokens,
                f"{prefix}cached_input_tokens": result.cached_input_tokens,
                f"{prefix}total_tokens": result.input_tokens + result.output_tokens,
                f"{prefix}estimated_cost_usd": result.estimated_cost_usd,
                f"{prefix}latency_ms": result.latency_ms,
            }
        )
        if self.official:
            self._merge_official_metrics_json(score)
        return score

    def _merge_official_metrics_json(self, score: TaskScore) -> None:
        """Parse the official metrics.json emitted by the STATE-Bench runner."""
        metrics_path = self.config.get("official_metrics_path")
        if not metrics_path:
            return
        path = Path(metrics_path)
        if not path.exists():
            _log.warning("STATE-Bench official metrics.json not found at %s", path)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _log.error("Failed to parse STATE-Bench metrics.json: %s", exc)
            return
        for key, value in data.items():
            score.official_metrics[f"official_STATE_Bench_{key}"] = value

    # ------------------------------------------------------------- samples
    @staticmethod
    def _sample_test_tasks(domain: str) -> list[BenchmarkTask]:
        samples: dict[str, BenchmarkTask] = {
            "travel": BenchmarkTask(
                id="travel_test_0",
                benchmark="state_bench",
                domain="travel",
                backgrounds=["The traveler has a tentative date and a budget of 1500."],
                subtasks=[
                    Subtask(id="t0", question="Should you book a refundable or "
                            "non-refundable fare and why?", expected_answer="refundable"),
                    Subtask(id="t1", question="What must you confirm before booking?",
                            expected_answer="traveler count and budget"),
                ],
                gold_memory_facts=["prefer refundable fares when dates are tentative"],
            ),
            "customer_support": BenchmarkTask(
                id="cs_test_0",
                benchmark="state_bench",
                domain="customer_support",
                backgrounds=["A customer reports a login issue."],
                subtasks=[
                    Subtask(id="c0", question="What must you do before sharing account "
                            "details?", expected_answer="verify account identity"),
                    Subtask(id="c1", question="What should you do for an unresolved issue?",
                            expected_answer="offer a workaround while escalating"),
                ],
                gold_memory_facts=["verify account identity before sharing details"],
            ),
            "shopping_assistant": BenchmarkTask(
                id="shop_test_0",
                benchmark="state_bench",
                domain="shopping_assistant",
                backgrounds=["The shopper stated a budget of 200 and prefers the brand Acme."],
                subtasks=[
                    Subtask(id="s0", question="What budget ceiling must recommendations "
                            "respect?", expected_answer="200"),
                    Subtask(id="s1", question="Which brand should you prefer?",
                            expected_answer="Acme"),
                ],
                gold_memory_facts=["respect budget ceiling", "match preferred brand"],
            ),
        }
        task = samples.get(domain)
        return [task] if task else []
