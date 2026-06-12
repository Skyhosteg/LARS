"""
tests/test_benchmark.py — Smoke test for the benchmark harness
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lars.baselines import ALL_BASELINES
from lars.benchmark import (
    MethodResult,
    aggregate_by_method,
    run_benchmark,
    to_table,
)
from lars.embeddings import HashEmbedder
from lars.llm import MockLLM
from lars.tasks import ALL_TASKS, PLANNING_TASKS, REASONING_TASKS


def test_all_tasks_have_required_fields():
    """Every task has a goal, CoT, interrupt, and expected intent."""
    for task in ALL_TASKS:
        assert task.id, "task must have an id"
        assert task.domain in ("planning", "reasoning"), f"bad domain: {task.domain}"
        assert task.goal, f"empty goal in {task.id}"
        assert task.initial_cot, f"empty initial_cot in {task.id}"
        assert task.interrupt, f"empty interrupt in {task.id}"
        assert task.intent, f"empty intent in {task.id}"
    print(f"✓ test_all_tasks_have_required_fields  ({len(ALL_TASKS)} tasks)")


def test_task_domain_split():
    """6 planning + 6 reasoning."""
    assert len(PLANNING_TASKS) == 6
    assert len(REASONING_TASKS) == 6
    assert len(ALL_TASKS) == 12
    print(f"✓ test_task_domain_split  (planning={len(PLANNING_TASKS)}, reasoning={len(REASONING_TASKS)})")


def test_benchmark_runs():
    """The full benchmark should run without error and produce results."""
    results = run_benchmark(
        llm=MockLLM(),
        embedder=HashEmbedder(),
        methods=["no_interrupt", "restart_from_scratch", "langgraph_checkpoint", "lars"],
        tasks=ALL_TASKS[:3],  # subset for speed
    )
    assert len(results) == 3 * 4  # 3 tasks × 4 methods
    assert all(isinstance(r, MethodResult) for r in results)
    print(f"✓ test_benchmark_runs  ({len(results)} results)")


def test_benchmark_lars_dominates_tradeoff():
    """LARS should be the only method with high RPR AND used interrupt=True."""
    results = run_benchmark(
        llm=MockLLM(),
        embedder=HashEmbedder(),
        methods=["no_interrupt", "restart_from_scratch", "langgraph_checkpoint", "lars"],
        tasks=ALL_TASKS,
    )
    summary = aggregate_by_method(results)

    # LARS should have high RPR
    assert summary["lars"]["mean_rpr"] >= 0.8, f"LARS RPR too low: {summary['lars']['mean_rpr']}"
    # LARS should always use the interrupt
    assert summary["lars"]["used_interrupt_rate"] == 1.0, "LARS should always use the interrupt"
    # no_interrupt should never use the interrupt
    assert summary["no_interrupt"]["used_interrupt_rate"] == 0.0
    # restart and langgraph should always use the interrupt
    assert summary["restart_from_scratch"]["used_interrupt_rate"] == 1.0
    assert summary["langgraph_checkpoint"]["used_interrupt_rate"] == 1.0

    # The "trade-off" claim: only LARS has BOTH high RPR and used=1
    for method, stats in summary.items():
        if method != "lars":
            # Every other method has either RPR < 0.5 OR used_int = 0.0
            is_dominated = (stats["mean_rpr"] < 0.5) or (stats["used_interrupt_rate"] == 0.0)
            assert is_dominated, f"{method} unexpectedly dominates LARS: {stats}"

    print(f"✓ test_benchmark_lars_dominates_tradeoff  (LARS RPR={summary['lars']['mean_rpr']}, used=100%)")


def test_benchmark_to_table_is_string():
    """The to_table helper should produce a string."""
    results = run_benchmark(
        llm=MockLLM(),
        embedder=HashEmbedder(),
        methods=["no_interrupt", "lars"],
        tasks=ALL_TASKS[:2],
    )
    table = to_table(results)
    assert isinstance(table, str)
    assert "task" in table  # header
    assert "method" in table
    print("✓ test_benchmark_to_table_is_string")


def test_all_baselines_present():
    """All 3 baselines are defined."""
    assert len(ALL_BASELINES) == 3
    method_names = [name for name, _ in ALL_BASELINES]
    assert "no_interrupt" in method_names
    assert "restart_from_scratch" in method_names
    assert "langgraph_checkpoint" in method_names
    print(f"✓ test_all_baselines_present  ({method_names})")


if __name__ == "__main__":
    test_all_tasks_have_required_fields()
    test_task_domain_split()
    test_benchmark_runs()
    test_benchmark_lars_dominates_tradeoff()
    test_benchmark_to_table_is_string()
    test_all_baselines_present()
    print("\nAll benchmark tests passed.")
