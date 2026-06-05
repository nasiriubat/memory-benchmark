#!/usr/bin/env python
"""Snapshot the OpenRouter model catalog and print resolved models.

Prints: number of models found, enabled model slots, requested alias, resolved
concrete model, context length, pricing, supported parameters, and availability.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from benchmark.snapshot import snapshot


def main() -> None:
    snapshot()


if __name__ == "__main__":
    main()
