"""Namespace uniqueness + isolation tests."""

from __future__ import annotations

from itertools import product

from benchmark.memory.namespace import make_namespace


def _ns(**overrides):
    base = dict(
        run_id="run1",
        benchmark="memoryarena",
        domain="bundled_shopping",
        memory_system="mem0",
        model="openai_frontier",
        task_id="t1",
        seed=1,
    )
    base.update(overrides)
    return make_namespace(**base)


def test_namespace_key_is_deterministic():
    assert _ns().key() == _ns().key()


def test_namespace_key_unique_across_every_axis():
    keys = set()
    for run_id, model, system, seed, task in product(
        ["r1", "r2"], ["m1", "m2"], ["mem0", "zep"], [1, 2], ["t1", "t2"]
    ):
        ns = _ns(run_id=run_id, model=model, memory_system=system, seed=seed, task_id=task)
        keys.add(ns.key())
    # 2 * 2 * 2 * 2 * 2 = 32 distinct combinations -> 32 distinct keys.
    assert len(keys) == 32


def test_changing_any_axis_changes_key():
    base = _ns().key()
    assert _ns(model="other").key() != base
    assert _ns(seed=999).key() != base
    assert _ns(memory_system="zep").key() != base
    assert _ns(task_id="t2").key() != base
    assert _ns(benchmark="dfx").key() != base
    assert _ns(domain="travel").key() != base
    assert _ns(run_id="r2").key() != base


def test_key_is_path_safe():
    ns = _ns(task_id="a/b/../c")
    assert "/.." not in ns.key().replace("run1/memoryarena", "")
    assert ns.flat_key().count("::") >= 6
