"""Stateless adapter behavior tests."""

from __future__ import annotations

from benchmark.memory.namespace import make_namespace
from benchmark.memory.stateless import StatelessAdapter
from benchmark.schemas import MemoryRecord


def _ns():
    return make_namespace(
        run_id="r", benchmark="b", domain="d", memory_system="stateless",
        model="m", task_id="t", seed=1,
    )


def test_stateless_retrieves_empty_list():
    adapter = StatelessAdapter()
    ns = _ns()
    adapter.setup(ns)
    assert adapter.retrieve("anything", 5, {"namespace": ns}) == []


def test_stateless_write_is_noop_but_reports_handled():
    adapter = StatelessAdapter()
    ns = _ns()
    rec = MemoryRecord(id="x", text="hello", namespace=ns)
    result = adapter.write(rec, {"namespace": ns})
    assert result.success is True
    assert result.memory_id is None
    # Even after a write, retrieval returns nothing.
    assert adapter.retrieve("hello", 5, {"namespace": ns}) == []


def test_stateless_store_size_is_zero():
    adapter = StatelessAdapter()
    assert adapter.store_size_bytes(_ns()) == 0
