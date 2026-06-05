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
