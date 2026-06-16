"""Configuration loading.

Reads YAML config files and environment variables. Nothing about model names or
run parameters is hard-coded in the runner; it all comes from configs + env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from benchmark.utils import configs_root


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_benchmark_profile(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Merge BENCHMARK_PROFILE preset into run (and optional benchmark toggles)."""
    profile_name = os.environ.get("BENCHMARK_PROFILE", "").strip()
    if not profile_name:
        return benchmark

    profiles = benchmark.get("profiles", {}) or {}
    if profile_name not in profiles:
        return benchmark

    benchmark = dict(benchmark)
    prof = dict(profiles[profile_name] or {})
    bench_over = prof.pop("benchmarks", None)

    run = dict(benchmark.get("run", {}) or {})
    run.update(prof)
    benchmark["run"] = run

    if bench_over:
        benches = dict(benchmark.get("benchmarks", {}) or {})
        for bname, bspec in bench_over.items():
            merged = dict(benches.get(bname, {}) or {})
            merged.update(bspec)
            benches[bname] = merged
        benchmark["benchmarks"] = benches

    benchmark["active_profile"] = profile_name
    return benchmark


@dataclass
class RunConfig:
    num_runs: int = 5
    retrieve_top_k: int = 3
    model_parallelism: int = 1
    task_parallelism: int = 1
    fail_fast: bool = False
    cross_task_memory: bool = False
    max_tasks_per_domain: int = 0
    max_tasks_total: int = 0
    task_sample_strategy: str = "sequential"
    benchmark_task_seed: int = 42
    profile: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        run = (data or {}).get("run", {})
        profile_name = (data or {}).get("active_profile") or os.environ.get(
            "BENCHMARK_PROFILE", ""
        ).strip()
        # Environment variables override config file / profile values.
        max_tasks = int(
            os.environ.get(
                "MAX_TASKS_PER_DOMAIN",
                run.get("max_tasks_per_domain", 0),
            )
        )
        max_tasks_total = int(
            os.environ.get(
                "MAX_TASKS_TOTAL",
                run.get("max_tasks_total", 0),
            )
        )
        task_sample_strategy = str(
            os.environ.get(
                "TASK_SAMPLE_STRATEGY",
                run.get("task_sample_strategy", "sequential"),
            )
        ).strip().lower() or "sequential"
        benchmark_task_seed = int(
            os.environ.get(
                "BENCHMARK_TASK_SEED",
                run.get("benchmark_task_seed", 42),
            )
        )
        return cls(
            num_runs=int(os.environ.get("DEFAULT_NUM_RUNS", run.get("num_runs", 5))),
            retrieve_top_k=int(os.environ.get("RETRIEVE_TOP_K", run.get("retrieve_top_k", 3))),
            model_parallelism=int(
                os.environ.get("MODEL_PARALLELISM", run.get("model_parallelism", 1))
            ),
            task_parallelism=int(
                os.environ.get("TASK_PARALLELISM", run.get("task_parallelism", 1))
            ),
            fail_fast=bool(run.get("fail_fast", False)),
            cross_task_memory=bool(run.get("cross_task_memory", False)),
            max_tasks_per_domain=max_tasks,
            max_tasks_total=max_tasks_total,
            task_sample_strategy=task_sample_strategy,
            benchmark_task_seed=benchmark_task_seed,
            profile=profile_name,
        )


@dataclass
class AppConfig:
    """All loaded configuration for a run."""

    benchmark: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    memory_systems: dict[str, Any] = field(default_factory=dict)
    pricing: dict[str, Any] = field(default_factory=dict)
    openrouter: dict[str, Any] = field(default_factory=dict)
    run: RunConfig = field(default_factory=RunConfig)

    @property
    def openrouter_policy(self) -> dict[str, Any]:
        return (self.openrouter.get("openrouter", {}) or {})

    @property
    def provider_policy(self) -> dict[str, Any]:
        return self.openrouter_policy.get("provider", {}) or {}

    def enabled_models(self) -> list[str]:
        return [
            slot
            for slot, spec in (self.models.get("models", {}) or {}).items()
            if spec.get("enabled", False)
        ]

    def enabled_benchmarks(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for name, spec in (self.benchmark.get("benchmarks", {}) or {}).items():
            if spec.get("enabled", False):
                out[name] = spec
        return out


def load_config(configs_dir: Path | None = None) -> AppConfig:
    """Load all configuration files into an :class:`AppConfig`."""
    root = configs_dir or configs_root()
    benchmark = _apply_benchmark_profile(_load_yaml(root / "benchmark.yaml"))
    models = _load_yaml(root / "models.yaml")
    memory_systems = _load_yaml(root / "memory_systems.yaml")
    pricing = _load_yaml(root / "pricing.yaml")
    openrouter = _load_yaml(root / "openrouter.yaml")
    return AppConfig(
        benchmark=benchmark,
        models=models,
        memory_systems=memory_systems,
        pricing=pricing,
        openrouter=openrouter,
        run=RunConfig.from_dict(benchmark),
    )
