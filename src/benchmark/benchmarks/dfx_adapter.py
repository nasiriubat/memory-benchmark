"""DFx benchmark adapter.

Loads local scenario JSON files from ``data/dfx/*.json``. Each scenario is a
sequence of subtasks where earlier subtasks introduce goals, decisions,
feedback, and constraints, and later subtasks require reuse of that prior
information. No confidential data is included; scenarios are abstracted.

Scenario JSON shape (see data/dfx/sample_scenario.json):
    {
      "id": "...",
      "domain": "...",
      "backgrounds": "..." | ["..."],
      "subtasks": [
        {
          "id": "...",
          "question": "...",
          "expected_answer": "...",
          "constraints": ["..."],
          "accepted_decisions": ["..."],
          "user_feedback": ["..."],
          "prior_recommendations": ["..."],
          "expected_facts": ["..."]
        }
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.benchmarks.base import SequentialBenchmarkAdapter
from benchmark.logging import get_logger
from benchmark.schemas import BenchmarkTask, Subtask
from benchmark.utils import project_root

_log = get_logger("benchmark.dfx")


class DFxAdapter(SequentialBenchmarkAdapter):
    name = "dfx"

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        path = (config or {}).get("path", "data/dfx")
        self.data_dir = Path(path)
        if not self.data_dir.is_absolute():
            self.data_dir = project_root() / path
        self._tasks: list[BenchmarkTask] = []

    def prepare(self) -> None:
        self._tasks = self._load_scenarios()
        _log.info("DFx: loaded %d scenario(s) from %s", len(self._tasks), self.data_dir)

    def list_tasks(self, domain: str) -> list[BenchmarkTask]:
        if not self._tasks:
            self.prepare()
        if domain in (None, "", "all"):
            return self._tasks
        return [t for t in self._tasks if t.domain == domain]

    def _load_scenarios(self) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        if not self.data_dir.exists():
            _log.warning("DFx data dir %s does not exist.", self.data_dir)
            return tasks
        for path in sorted(self.data_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                _log.error("Failed to parse DFx scenario %s: %s", path, exc)
                continue
            tasks.append(self._to_task(raw, path))
        return tasks

    @staticmethod
    def _to_task(raw: dict[str, Any], path: Path) -> BenchmarkTask:
        backgrounds = raw.get("backgrounds", [])
        if isinstance(backgrounds, str):
            backgrounds = [backgrounds]
        subtasks = []
        for i, st in enumerate(raw.get("subtasks", [])):
            subtasks.append(
                Subtask(
                    id=str(st.get("id", f"st{i}")),
                    question=str(st.get("question", "")),
                    expected_answer=st.get("expected_answer"),
                    metadata={
                        "constraints": st.get("constraints", []),
                        "accepted_decisions": st.get("accepted_decisions", []),
                        "user_feedback": st.get("user_feedback", []),
                        "prior_recommendations": st.get("prior_recommendations", []),
                        "corrected_mistakes": st.get("corrected_mistakes", []),
                        "expected_facts": st.get("expected_facts", []),
                    },
                )
            )
        return BenchmarkTask(
            id=str(raw.get("id", path.stem)),
            benchmark="dfx",
            domain=str(raw.get("domain", "dfx")),
            subtasks=subtasks,
            backgrounds=list(backgrounds),
            expected_answers=[s.expected_answer or "" for s in subtasks],
            gold_memory_facts=raw.get("gold_memory_facts", []),
            metadata={"source_file": str(path)},
        )
