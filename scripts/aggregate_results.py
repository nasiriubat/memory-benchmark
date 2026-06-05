#!/usr/bin/env python
"""Aggregate all raw + per-system results into results/aggregated/."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from benchmark.aggregate import aggregate
from benchmark.utils import results_root


def main() -> None:
    aggregate(results_root())


if __name__ == "__main__":
    main()
