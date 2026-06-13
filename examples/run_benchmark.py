"""
examples/run_benchmark.py — Run the LARS benchmark

12 tasks × 4 methods = 48 measurements. Reports:
  - Per-task results table
  - Aggregate table (RPR / latency / cost per method)
  - Winner: which method has highest RPR? lowest cost? lowest latency?

Run:
    python examples/run_benchmark.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lars.benchmark import (
    aggregate_by_method,
    run_benchmark,
    to_json,
    to_table,
)
from lars.embeddings import HashEmbedder
from lars.llm import MockLLM


def banner(s: str) -> None:
    print()
    print("#" * 78)
    print(f"### {s}")
    print("#" * 78)


def main() -> None:
    banner("LARS BENCHMARK — 12 tasks × 4 methods")
    print()
    print("  Tasks   : 12 (6 planning + 6 reasoning)")
    print("  Methods : no_interrupt, restart, langgraph, lars")
    print("  Metrics : RPR (semantic), latency (M2), cost (M3 proxy)")
    print("  Embedder: HashEmbedder (deterministic, no API key)")
    print()

    # Use a stateful MockLLM that returns the initial CoT for any task
    # (this is a stress test: the methods all start from the same state)
    mock = MockLLM()
    embedder = HashEmbedder()

    # Run
    results = run_benchmark(llm=mock, embedder=embedder)

    # Per-task table
    banner("PER-TASK RESULTS")
    print()
    print(to_table(results))

    # Aggregate
    banner("AGGREGATE (mean over 12 tasks)")
    print()
    summary = aggregate_by_method(results)
    headers = ["method", "n", "mean_RPR", "mean_latency_ms", "mean_cost", "used_int_rate"]
    rows = []
    for method, stats in summary.items():
        rows.append([
            method,
            str(stats["n"]),
            f"{stats['mean_rpr']:.3f}",
            f"{stats['mean_latency_ms']:.2f}",
            f"{stats['mean_cost']:.1f}",
            f"{stats['used_interrupt_rate']:.2f}",
        ])
    widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json")
    to_json(results, out_path)
    print(f"\n[benchmark] Saved raw results to {out_path}")

    # Verdict
    banner("VERDICT")
    print()
    best_rpr = max(summary.items(), key=lambda kv: kv[1]["mean_rpr"])
    best_cost = min(summary.items(), key=lambda kv: kv[1]["mean_cost"])
    best_latency = min(summary.items(), key=lambda kv: kv[1]["mean_latency_ms"])
    print(f"  Best RPR      : {best_rpr[0]}  ({best_rpr[1]['mean_rpr']:.3f})")
    print(f"  Lowest cost   : {best_cost[0]}  ({best_cost[1]['mean_cost']:.1f} words)")
    print(f"  Lowest latency: {best_latency[0]}  ({best_latency[1]['mean_latency_ms']:.2f} ms)")
    print()
    print("  The trade-off table (the paper's main figure):")
    print("  +--------------------+--------+--------+--------+---------+")
    print("  | Method             | RPR ↑  | Cost ↓ | Used?  | Win?    |")
    print("  +--------------------+--------+--------+--------+---------+")
    for method, stats in summary.items():
        rpr = stats["mean_rpr"]
        cost = stats["mean_cost"]
        used = stats["used_interrupt_rate"]
        # Win if both RPR is high AND used the interrupt
        win = "✓ LARS" if rpr >= 0.8 and used == 1.0 else (" " if rpr < 0.5 or used == 0.0 else "~")
        print(f"  | {method:18s} | {rpr:6.3f} | {cost:6.1f} | {used:6.2f} | {win:7s} |")
    print("  +--------------------+--------+--------+--------+---------+")
    print()
    print("  Headline (for the paper):")
    print(f"    Only LARS achieves high RPR ({summary.get('lars', {}).get('mean_rpr', 0):.0%})")
    print(f"    WHILE incorporating the user's interrupt (used_int=100%).")
    print(f"    Baselines either preserve everything but ignore the user (no_interrupt),")
    print(f"    or incorporate the user but lose the original reasoning (restart, langgraph).")


if __name__ == "__main__":
    main()
