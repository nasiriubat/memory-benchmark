#!/usr/bin/env python
"""Run a single memory system across the configured models.

At the start of each run this:
  1. creates a run_id (if not provided),
  2. calls the ModelResolver (saving the catalog snapshot + resolved mapping),
  3. runs the benchmark using the resolved concrete model ids,
  4. saves results into the dedicated per-memory-system folder, and
  5. aggregates partial results + prints a summary.

Examples:
    python scripts/run_memory_system.py --memory-system stateless
    python scripts/run_memory_system.py --memory-system hindsight --models openai_frontier,google_frontier
"""

from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401  (sets up sys.path)

from benchmark.aggregate import aggregate
from benchmark.cli import run_memory_system
from benchmark.logging import get_logger
from benchmark.utils import results_root

_log = get_logger("scripts.run_memory_system")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one memory system.")
    parser.add_argument("--memory-system", required=True, help="Memory system id.")
    parser.add_argument("--models", default="", help="Comma-separated model slots.")
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", ""), help="Run id.")
    parser.add_argument(
        "--no-aggregate", action="store_true", help="Skip partial aggregation at the end."
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    run_memory_system(args.memory_system, models=models, run_id=args.run_id or None)

    if not args.no_aggregate:
        _log.info("Aggregating partial results...")
        aggregate(results_root())


if __name__ == "__main__":
    main()
