#!/usr/bin/env python
"""Import / prepare external benchmark data.

Detects whether the official STATE-Bench and MemoryArena runners are installed,
and (when the ``datasets`` package is available) pre-downloads the MemoryArena
Hugging Face fallback dataset so subsequent runs are offline-friendly.

This script does NOT reimplement official scoring; it only prepares data and
reports which official runners are available so the adapters can wrap them.
"""

from __future__ import annotations

import argparse
import importlib.util

import _bootstrap  # noqa: F401

from benchmark.config import load_config
from benchmark.logging import get_logger

_log = get_logger("scripts.import_external")


def _available(*names: str) -> bool:
    return any(importlib.util.find_spec(n) is not None for n in names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external benchmarks.")
    parser.add_argument("--download-hf", action="store_true",
                        help="Pre-download the MemoryArena HF fallback dataset.")
    args = parser.parse_args()

    cfg = load_config()
    _log.info("STATE-Bench official runner installed: %s", _available("state_bench", "statebench"))
    _log.info("MemoryArena official runner installed: %s", _available("memoryarena"))

    if args.download_hf:
        if importlib.util.find_spec("datasets") is None:
            _log.warning("`datasets` not installed; cannot pre-download. pip install '.[datasets]'")
            return
        from datasets import load_dataset

        ma = cfg.benchmark.get("benchmarks", {}).get("memoryarena", {})
        hf = ma.get("hf_dataset", "ZexueHe/memoryarena")
        for config in ma.get("configs", []):
            try:
                load_dataset(hf, config, split="test")
                _log.info("Cached MemoryArena config '%s' from %s.", config, hf)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Could not cache config '%s': %s", config, exc)


if __name__ == "__main__":
    main()
