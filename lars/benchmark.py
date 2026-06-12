"""
benchmark.py — The benchmark harness for LARS evaluation

Runs the 12 tasks through each method (3 baselines + LARS), measures:

  - RPR_semantic : Reasoning Preservation Rate (semantic similarity)
  - Latency      : M2, time to produce S(t+1)
  - Cost         : M3, proxy = word count of CoT generated
  - Used interrupt? : did the method even incorporate the user's input?

Output: a results table ready for the paper.

Note on cost: in production with OpenAI we'd use response.usage.prompt_tokens
+ completion_tokens. For offline benchmarking we use word count as a
deterministic proxy. The relative ordering of methods is preserved.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from .baselines import (
    method_lars,
    method_langgraph_checkpoint,
    method_no_interrupt,
    method_restart_from_scratch,
)
from .delta_u import DeltaUParserMock
from .embeddings import Embedder, HashEmbedder
from .extractor import StateExtractor
from .llm import LLMClient
from .merger import StateMerger
from .metrics import rpr_semantic
from .state import StateVector
from .tasks import ALL_TASKS, BenchmarkTask


@dataclass
class MethodResult:
    task_id: str
    domain: str
    method: str
    rpr_semantic: float
    latency_ms: float
    cost_words: int
    used_interrupt: bool
    intent_matched: bool  # did the method "see" the interrupt correctly


def _word_count(s: str) -> int:
    return len(s.split())


def _intent_matched(actual: str, expected: str) -> bool:
    """Did the method handle the right intent type?"""
    return actual == expected


def run_method(
    method_name: str,
    task: BenchmarkTask,
    llm: LLMClient,
    extractor: StateExtractor,
    parser: DeltaUParserMock,
    merger: StateMerger,
    embedder: Embedder,
    ground_truth: StateVector,
) -> MethodResult:
    """Run a single method on a single task and measure all metrics."""
    t0 = time.perf_counter()

    if method_name == "no_interrupt":
        sv, info = method_no_interrupt(task.goal, task.initial_cot, task.interrupt, llm, extractor)
        cost = _word_count(task.initial_cot)
    elif method_name == "restart_from_scratch":
        sv, info = method_restart_from_scratch(task.goal, task.initial_cot, task.interrupt, llm, extractor)
        cost = _word_count(task.initial_cot) + _word_count(task.interrupt)
    elif method_name == "langgraph_checkpoint":
        sv, info = method_langgraph_checkpoint(task.goal, task.initial_cot, task.interrupt, llm, extractor)
        cost = _word_count(task.initial_cot) + _word_count(task.interrupt)
    elif method_name == "lars":
        sv, info = method_lars(task.goal, task.initial_cot, task.interrupt, llm, extractor, parser, merger)
        cost = _word_count(task.initial_cot)  # LARS reads the same CoT, applies merge (no re-gen)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # RPR vs ground truth (the original plan, with no interrupt)
    rpr = rpr_semantic(ground_truth, sv, embedder=embedder, threshold=0.7)

    # Did the method's intent match what we expect?
    intent_actual = info.get("intent_type", "n/a")
    intent_ok = (
        info.get("used_interrupt", False)
        and (method_name in ("lars", "langgraph_checkpoint", "restart_from_scratch"))
        and _intent_matched(intent_actual, task.intent)
    ) or (
        method_name == "no_interrupt"
        and not info.get("used_interrupt", False)
    )

    return MethodResult(
        task_id=task.id,
        domain=task.domain,
        method=method_name,
        rpr_semantic=rpr,
        latency_ms=round(elapsed_ms, 3),
        cost_words=cost,
        used_interrupt=info.get("used_interrupt", False),
        intent_matched=intent_ok,
    )


def run_benchmark(
    llm: LLMClient,
    embedder: Embedder | None = None,
    methods: list[str] | None = None,
    tasks: list[BenchmarkTask] | None = None,
) -> list[MethodResult]:
    """
    Run the full benchmark.

    Returns a list of MethodResult, one per (task, method) combination.
    """
    embedder = embedder or HashEmbedder()
    extractor = StateExtractor(llm)
    parser = DeltaUParserMock()
    merger = StateMerger()
    methods = methods or ["no_interrupt", "restart_from_scratch", "langgraph_checkpoint", "lars"]
    tasks = tasks or ALL_TASKS

    results: list[MethodResult] = []

    for task in tasks:
        # Ground truth = the state without any interrupt applied
        ground_truth, _ = method_no_interrupt(task.goal, task.initial_cot, None, llm, extractor)

        for method_name in methods:
            r = run_method(method_name, task, llm, extractor, parser, merger, embedder, ground_truth)
            results.append(r)

    return results


def to_table(results: list[MethodResult]) -> str:
    """Format results as an ASCII table for the paper / console."""
    headers = ["task", "domain", "method", "RPR", "latency(ms)", "cost(w)", "used_int", "intent_ok"]
    rows = []
    for r in results:
        rows.append([
            r.task_id[:30],
            r.domain,
            r.method,
            f"{r.rpr_semantic:.3f}",
            f"{r.latency_ms:.1f}",
            str(r.cost_words),
            "✓" if r.used_interrupt else "✗",
            "✓" if r.intent_matched else "✗",
        ])

    # Compute column widths
    widths = [max(len(str(c)) for c in [h] + [r[i] for r in rows]) for i, h in enumerate(headers)]

    def fmt_row(row):
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths))

    out = []
    out.append(fmt_row(headers))
    out.append("  ".join("-" * w for w in widths))
    for row in rows:
        out.append(fmt_row(row))
    return "\n".join(out)


def aggregate_by_method(results: list[MethodResult]) -> dict[str, dict[str, float]]:
    """Aggregate RPR / latency / cost by method, for the paper's main table."""
    agg: dict[str, list[MethodResult]] = {}
    for r in results:
        agg.setdefault(r.method, []).append(r)

    summary = {}
    for method, rs in agg.items():
        summary[method] = {
            "n": len(rs),
            "mean_rpr": round(sum(x.rpr_semantic for x in rs) / len(rs), 3),
            "mean_latency_ms": round(sum(x.latency_ms for x in rs) / len(rs), 2),
            "mean_cost": round(sum(x.cost_words for x in rs) / len(rs), 1),
            "used_interrupt_rate": round(sum(1 for x in rs if x.used_interrupt) / len(rs), 2),
        }
    return summary


def to_json(results: list[MethodResult], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
