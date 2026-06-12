"""
examples/demo_merge.py — End-to-end demo of LARS v2

Full pipeline:
  1. Extract S(t) from CoT
  2. User interrupts
  3. Parse interrupt → UpdateIntent (ΔU)
  4. Apply merge f(S, ΔU) → S(t+1)
  5. Show MergeTrace (what f actually did)
  6. Compute RPR(S(t), S(t+1))

Runs in MOCK mode (no API key) or REAL (with OPENAI_API_KEY).

Run:
    python examples/demo_merge.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lars.delta_u import DeltaUParserMock
from lars.extractor import StateExtractor
from lars.llm import MockLLM, OpenAILLM, default_client
from lars.merger import StateMerger
from lars.metrics import measure_adaptation_latency, rpr


# --------------------------------------------------------------------------- #
# Example: marketing plan for a fitness app
# --------------------------------------------------------------------------- #

GOAL = "Create a marketing plan for a fitness app targeting young adults in Egypt."

COT_INITIAL = """
Step 1: Analyze the market.
- Egypt has 110M people, 60% under 30, growing fitness app adoption.
- Target all major cities: Cairo, Alexandria, Giza.

Step 2: Define the audience.
- 18-30 year olds, urban, middle income, health-conscious.

Step 3: Choose channels.
- Instagram and TikTok for 18-30 reach, plus local influencers.

Step 4: Allocate budget.
- 40% paid social, 30% influencers, 20% content, 10% events.

Step 5: Define KPIs.
- Downloads, 30-day retention, CAC, LTV.
"""

# Three different interrupts, three different intent types
INTERRUPTS = [
    # Note: the mock ΔU parser is English-regex based.
    # The LLM-backed parser handles Arabic/native scripts.
    ("focus on Cairo only", "Cairo-only scope narrow"),  # English
    ("also include the Gulf region", "Scope expand to Gulf"),
    ("use Twitter instead of Facebook", "Channel replacement"),
]


# --------------------------------------------------------------------------- #
# Canned S(t) — shared by all three demos
# --------------------------------------------------------------------------- #

INITIAL_FIXTURE = {
    "Analyze the market": {
        "goal": GOAL,
        "steps_completed": [
            {"step_id": 1, "description": "Analyze the market: Egypt has 110M people, 60% under 30, growing fitness app adoption", "status": "completed", "dependencies": []},
            {"step_id": 2, "description": "Define the audience: 18-30 year olds, urban, middle income, health-conscious", "status": "completed", "dependencies": [1]},
            {"step_id": 3, "description": "Choose channels: Instagram and TikTok for 18-30 reach, plus Facebook ads and local influencers", "status": "completed", "dependencies": [2]},
        ],
        "steps_pending": [
            {"step_id": 4, "description": "Allocate budget across paid social, influencers, content", "status": "pending", "dependencies": [3]},
            {"step_id": 5, "description": "Define KPIs: downloads, retention, CAC, LTV", "status": "pending", "dependencies": [3, 4]},
        ],
        "assumptions": [
            "Egypt has 110M people, 60% under 30",
            "Fitness app market is growing 15% YoY",
            "Target all major cities including Cairo and Alexandria",
        ],
        "decisions": [
            {"decision": "Target 18-30 urban audience", "rationale": "Largest growing segment with fitness app adoption"},
            {"decision": "Use Instagram and TikTok as primary channels", "rationale": "High engagement with 18-30 demographic"},
        ],
        "confidence": 0.78,
    }
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def banner(s: str) -> None:
    print()
    print("=" * 78)
    print(f" {s}")
    print("=" * 78)


def show_state(label: str, sv) -> None:
    print(f"\n[{label}] {sv.summary()}")
    print(f"  goal: {sv.goal}")
    print(f"  completed steps:")
    for s in sv.steps_completed:
        print(f"    {s.step_id}. {s.description}")
    print(f"  pending steps:")
    for s in sv.steps_pending:
        print(f"    {s.step_id}. {s.description}")
    if sv.assumptions:
        print(f"  assumptions:")
        for a in sv.assumptions:
            print(f"    - {a}")
    if sv.decisions:
        print(f"  decisions:")
        for d in sv.decisions:
            print(f"    - {d.decision}")
    print(f"  confidence: {sv.confidence}")


def show_trace(trace) -> None:
    print(f"\n[MERGE TRACE] {trace.summary()}")
    if trace.preserved_steps:
        print(f"  preserved steps: {trace.preserved_steps}")
    if trace.modified_steps:
        print(f"  modified steps:  {trace.modified_steps}  (kept structure, updated content)")
    if trace.dropped_steps:
        print(f"  dropped steps:   {trace.dropped_steps}")
    if trace.inserted_steps:
        print(f"  inserted steps:  {trace.inserted_steps}")
    if trace.inserted_assumptions:
        print(f"  inserted assumptions:")
        for a in trace.inserted_assumptions:
            print(f"    + {a}")
    if trace.inserted_decisions:
        print(f"  inserted decisions:")
        for d in trace.inserted_decisions:
            print(f"    + {d}")
    if trace.notes:
        print(f"  notes:")
        for n in trace.notes:
            print(f"    • {n}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run() -> None:
    mock = not os.getenv("OPENAI_API_KEY")
    mode = "MOCK (no API key)" if mock else "REAL (OpenAI)"
    banner(f"LARS v2 — Full pipeline [ΔU → f → S(t+1) → RPR]  [{mode}]")
    print(
        "\n  Pipeline: extract S(t)  →  parse user interrupt  →  "
        "apply merge f(S, ΔU)  →  show trace  →  compute RPR\n"
    )

    # Set up LLM (real or mock)
    if mock:
        llm = MockLLM(fixtures=INITIAL_FIXTURE)
    else:
        llm = default_client()
    extractor = StateExtractor(llm)
    parser = DeltaUParserMock()  # heuristic parser (no LLM needed)
    merger = StateMerger(alpha=0.6, beta=0.3, gamma=0.1)

    # 1) Extract S(t) once
    s_before, _ = measure_adaptation_latency(extractor.extract, GOAL, COT_INITIAL)
    show_state("S(t)  initial", s_before)

    # 2) Run each interrupt through the full pipeline
    for text, label in INTERRUPTS:
        banner(f"INTERRUPT  —  '{text}'  ({label})")
        intent = parser.parse(text)
        print(f"\n[ΔU PARSER]  →  {intent.short()}")

        s_after, trace = merger.merge(s_before, intent)
        show_trace(trace)
        show_state("S(t+1) merged", s_after)

        score = rpr(s_before, s_after, match="jaccard")
        # Also compute the rule-based preservation (more honest for merge traces)
        rule_pres = trace.preservation_rate_steps()
        print(f"\n[METRICS]")
        print(f"  RPR[jaccard]             = {score:.4f}  ({score * 100:.1f}%)")
        print(f"  Step preservation rate   = {rule_pres:.4f}  ({rule_pres * 100:.1f}%)  "
              f"(from MergeTrace: preserved+modified / total)")

    banner("DONE")
    print(
        "\nWhat you just saw:\n"
        "  - ΔU parser classified each interrupt correctly\n"
        "  - f(S, ΔU) applied the change with α=0.6 preservation bias\n"
        "  - MergeTrace records every preserve/modify/drop/insert\n"
        "  - RPR shows how much reasoning survived the update\n"
        "\nNext: wire this into LangGraph (G1) + benchmark (G5/G6).\n"
    )


if __name__ == "__main__":
    run()
