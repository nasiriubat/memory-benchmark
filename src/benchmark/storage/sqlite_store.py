"""SQLite result store (optional, used by aggregation for queryable results)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from benchmark.schemas import TaskResult
from benchmark.utils import ensure_dir


class SQLiteStore:
    """Stores flattened task results in a single SQLite table."""

    TABLE = "task_results"

    def __init__(self, db_path: Path) -> None:
        ensure_dir(db_path.parent)
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "run_id TEXT, benchmark TEXT, domain TEXT, memory_system TEXT, "
            "model TEXT, task_id TEXT, seed INTEGER, payload TEXT)"
        )
        self._conn.commit()

    def insert(self, result: TaskResult) -> None:
        self._conn.execute(
            f"INSERT INTO {self.TABLE} "
            "(run_id, benchmark, domain, memory_system, model, task_id, seed, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.run_id,
                result.benchmark,
                result.domain,
                result.memory_system,
                result.model,
                result.task_id,
                result.seed,
                json.dumps(result.model_dump(), default=str),
            ),
        )
        self._conn.commit()

    def fetch_all(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(f"SELECT payload FROM {self.TABLE}")
        return [json.loads(row[0]) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
