"""Helpers for capping benchmark task counts."""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def domain_task_limits(domain_sizes: dict[str, int], max_total: int) -> dict[str, int]:
    """Split ``max_total`` tasks evenly across domains (remainder to first domains).

    Each domain is capped by its available task count. Returns per-domain slice
    limits (not offsets). When ``max_total <= 0``, every domain keeps all tasks.
    """
    if max_total <= 0:
        return {domain: size for domain, size in domain_sizes.items()}

    domains = list(domain_sizes.keys())
    if not domains:
        return {}

    base, extra = divmod(max_total, len(domains))
    limits: dict[str, int] = {}
    for index, domain in enumerate(domains):
        cap = base + (1 if index < extra else 0)
        limits[domain] = min(domain_sizes[domain], cap)
    return limits


def select_labeled_tasks(
    labeled_tasks: list[tuple[str, T]],
    *,
    max_total: int,
    strategy: str = "sequential",
    seed: int = 42,
) -> dict[str, list[T]]:
    """Select up to ``max_total`` tasks grouped by domain label.

    ``strategy``:
      - ``random``: shuffle the full pool with ``seed``, then take ``max_total``.
      - ``sequential``: even per-domain caps (see :func:`domain_task_limits`).
    """
    if max_total <= 0:
        grouped: dict[str, list[T]] = {}
        for domain, task in labeled_tasks:
            grouped.setdefault(domain, []).append(task)
        return grouped

    if strategy == "random":
        pool = list(labeled_tasks)
        rng = random.Random(seed)
        rng.shuffle(pool)
        pool = pool[:max_total]
        selected: dict[str, list[T]] = {}
        for domain, task in pool:
            selected.setdefault(domain, []).append(task)
        return selected

    sizes: dict[str, int] = {}
    by_domain: dict[str, list[T]] = {}
    for domain, task in labeled_tasks:
        by_domain.setdefault(domain, []).append(task)
        sizes[domain] = len(by_domain[domain])
    limits = domain_task_limits(sizes, max_total)
    return {domain: by_domain[domain][: limits[domain]] for domain in by_domain}
