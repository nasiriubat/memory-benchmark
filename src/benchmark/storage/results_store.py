"""Result store: writes task results to dedicated per-memory-system folders.

Layout (per the latest requirements):
    results/by_memory_system/{memory_system}/{run_id}/
        results.jsonl          # one TaskResult JSON per line
        results.csv            # flattened tabular rows (also UI-downloadable)
        results.parquet        # flattened tabular rows
        run_metadata.json      # run-level metadata + resolved models

Results are also mirrored into SQLite (optional) and a global parquet during
aggregation. This module focuses on the canonical per-system folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.logging import get_logger
from benchmark.memory.namespace import per_system_dir
from benchmark.schemas import RunMetadata, TaskResult
from benchmark.utils import ensure_dir

_log = get_logger("benchmark.storage")


class ResultsStore:
    """Append task results into a dedicated per-memory-system run folder."""

    def __init__(self, results_root: Path, memory_system: str, run_id: str) -> None:
        self.dir = ensure_dir(per_system_dir(results_root, memory_system, run_id))
        self.memory_system = memory_system
        self.run_id = run_id
        self._jsonl = self.dir / "results.jsonl"
        self._rows: list[dict[str, Any]] = []

    def write_metadata(self, metadata: RunMetadata) -> None:
        path = self.dir / "run_metadata.json"
        path.write_text(json.dumps(metadata.model_dump(), indent=2, default=str), encoding="utf-8")
        _log.info("Wrote run metadata to %s", path)

    def append(self, result: TaskResult) -> None:
        """Append a single task result (JSONL immediately; tabular on flush)."""
        with self._jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.model_dump(), default=str) + "\n")
        self._rows.append(result.flatten())

    def flush_tabular(self) -> None:
        """Write the accumulated rows to CSV and Parquet."""
        if not self._rows:
            _log.warning("No results to flush for %s/%s.", self.memory_system, self.run_id)
            return
        import pandas as pd

        df = pd.DataFrame(self._rows)
        csv_path = self.dir / "results.csv"
        df.to_csv(csv_path, index=False)
        try:
            parquet_path = self.dir / "results.parquet"
            df.to_parquet(parquet_path, index=False)
            _log.info("Wrote %d rows to %s and %s", len(df), csv_path, parquet_path)
        except Exception as exc:  # noqa: BLE001 - parquet engine optional
            _log.warning("Parquet write failed (%s); CSV still written to %s", exc, csv_path)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows
