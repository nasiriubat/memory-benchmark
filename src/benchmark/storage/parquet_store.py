"""Parquet helpers for reading/writing flattened result tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.logging import get_logger
from benchmark.utils import ensure_dir

_log = get_logger("benchmark.storage.parquet")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to a Parquet file (creating parent dirs)."""
    import pandas as pd

    ensure_dir(path.parent)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    _log.info("Wrote %d rows to %s", len(df), path)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")
