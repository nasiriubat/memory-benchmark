#!/usr/bin/env python
"""Reset memory state.

For self-hosted backends this would clear collections; for the local fallback
store it is a no-op (state is in-process). Optionally clears results for a run.

Examples:
    python scripts/reset_memory.py --memory-system mem0
    python scripts/reset_memory.py --clear-results --run-id run_2026...
"""

from __future__ import annotations

import argparse
import shutil

import _bootstrap  # noqa: F401

from benchmark.config import load_config
from benchmark.logging import get_logger
from benchmark.runner import build_memory_adapter
from benchmark.utils import results_root

_log = get_logger("scripts.reset_memory")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset memory / results.")
    parser.add_argument("--memory-system", default="", help="Memory system id to reset.")
    parser.add_argument("--clear-results", action="store_true", help="Delete a run's results.")
    parser.add_argument("--run-id", default="", help="Run id to clear results for.")
    args = parser.parse_args()

    if args.memory_system:
        cfg = load_config()
        adapter = build_memory_adapter(args.memory_system, cfg)
        _log.info(
            "Memory system '%s' available=%s. Self-hosted backends should be reset "
            "via their own tooling / by recreating the container volume.",
            args.memory_system, adapter.available,
        )
        adapter.close()

    if args.clear_results and args.run_id:
        target = results_root() / "by_memory_system"
        removed = 0
        for system_dir in target.glob("*"):
            run_dir = system_dir / args.run_id
            if run_dir.exists():
                shutil.rmtree(run_dir)
                removed += 1
        _log.info("Cleared results for run_id=%s (%d folders).", args.run_id, removed)


if __name__ == "__main__":
    main()
