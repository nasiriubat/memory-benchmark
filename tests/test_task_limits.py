"""Task limit helpers."""

from benchmark.task_limits import domain_task_limits, select_labeled_tasks


def test_domain_task_limits_even_split():
    limits = domain_task_limits(
        {
            "bundled_shopping": 150,
            "progressive_search": 221,
            "group_travel_planner": 270,
            "formal_reasoning_math": 40,
            "formal_reasoning_phys": 20,
        },
        100,
    )
    assert sum(limits.values()) == 100
    assert limits == {
        "bundled_shopping": 20,
        "progressive_search": 20,
        "group_travel_planner": 20,
        "formal_reasoning_math": 20,
        "formal_reasoning_phys": 20,
    }


def test_domain_task_limits_no_cap():
    sizes = {"a": 10, "b": 20}
    assert domain_task_limits(sizes, 0) == sizes


def test_select_labeled_tasks_random_is_reproducible():
    labeled = [(f"d{i % 3}", f"t{i}") for i in range(20)]
    first = select_labeled_tasks(labeled, max_total=7, strategy="random", seed=42)
    second = select_labeled_tasks(labeled, max_total=7, strategy="random", seed=42)
    assert first == second
    assert sum(len(v) for v in first.values()) == 7
