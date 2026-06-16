"""Config profile loading."""

from __future__ import annotations

import os

from benchmark.config import RunConfig, load_config


def test_paper_profile_sets_max_tasks(monkeypatch):
    monkeypatch.setenv("BENCHMARK_PROFILE", "paper")
    monkeypatch.delenv("MAX_TASKS_PER_DOMAIN", raising=False)
    cfg = load_config()
    assert cfg.run.profile == "paper"
    assert cfg.run.num_runs == 5
    assert cfg.run.max_tasks_per_domain == 50


def test_env_overrides_profile_max_tasks(monkeypatch):
    monkeypatch.setenv("BENCHMARK_PROFILE", "paper")
    monkeypatch.setenv("MAX_TASKS_PER_DOMAIN", "10")
    cfg = load_config()
    assert cfg.run.max_tasks_per_domain == 10


def test_smoke_profile_disables_memoryarena(monkeypatch):
    monkeypatch.setenv("BENCHMARK_PROFILE", "smoke")
    monkeypatch.delenv("BENCHMARKS", raising=False)
    cfg = load_config()
    enabled = set(cfg.enabled_benchmarks())
    assert "state_bench" in enabled
    assert "memoryarena" not in enabled


def test_run_config_from_dict_without_profile():
    rc = RunConfig.from_dict({"run": {"num_runs": 3, "max_tasks_per_domain": 7}})
    assert rc.num_runs == 3
    assert rc.max_tasks_per_domain == 7


def test_memoryarena_100_profile(monkeypatch):
    monkeypatch.setenv("BENCHMARK_PROFILE", "memoryarena_100")
    monkeypatch.delenv("MAX_TASKS_TOTAL", raising=False)
    cfg = load_config()
    assert cfg.run.profile == "memoryarena_100"
    assert cfg.run.max_tasks_total == 100
    assert cfg.run.task_sample_strategy == "random"
    assert cfg.run.benchmark_task_seed == 42
    enabled = set(cfg.enabled_benchmarks())
    assert enabled == {"memoryarena"}


def test_env_overrides_max_tasks_total(monkeypatch):
    monkeypatch.setenv("BENCHMARK_PROFILE", "memoryarena_100")
    monkeypatch.setenv("MAX_TASKS_TOTAL", "50")
    cfg = load_config()
    assert cfg.run.max_tasks_total == 50


def test_memoryarena_reasoning_profile(monkeypatch):
    monkeypatch.setenv("BENCHMARK_PROFILE", "memoryarena_reasoning")
    monkeypatch.delenv("MAX_TASKS_TOTAL", raising=False)
    cfg = load_config()
    assert cfg.run.profile == "memoryarena_reasoning"
    assert cfg.run.max_tasks_total == 0
    assert cfg.run.task_sample_strategy == "sequential"
    enabled = set(cfg.enabled_benchmarks())
    assert enabled == {"memoryarena"}
    ma = cfg.benchmark["benchmarks"]["memoryarena"]
    assert ma["configs"] == ["formal_reasoning_math", "formal_reasoning_phys"]
    assert cfg.benchmark["benchmarks"]["state_bench"]["enabled"] is False
