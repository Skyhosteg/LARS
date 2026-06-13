"""
examples/demo.py — End-to-end demo of the S(t) Extractor

This is the simplest proof that LARS works:
  1. Take a real goal
  2. Generate two CoT snapshots (before and after a user interrupt)
  3. Extract StateVector for each
  4. Compute RPR — this is the headline metric

Runs in two modes:
  - MOCK (default, no API key needed)
  - REAL (if OPENAI_API_KEY is set in env)

Run:
    python examples/demo.py
"""

from __future__ import annotations

import json
import os
import sys

# Make `lars` importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lars.extractor import StateExtractor
from lars.llm import LLMClient, MockLLM, OpenAILLM, default_client
from lars.metrics import measure_adaptation_latency, rpr


# --------------------------------------------------------------------------- #
# Example: marketing plan for a fitness app, with a "focus on Cairo" pivot
# --------------------------------------------------------------------------- #

GOAL = "Create a marketing plan for a fitness app targeting young adults in Egypt."

COT_BEFORE_INTERRUPT = """
The user wants a marketing plan for a fitness app in Egypt.

Step 1: Analyze the market.
- Egypt has ~110M people, ~60% under 30.
- Fitness app market is growing ~15% YoY.
- Competitors: Gymondo (low penetration), local apps (fragmented).
- I will target all major cities: Cairo, Alexandria, Giza.

Step 2: Define the audience.
- Primary: 18-30 year olds, urban, middle income.
- Secondary: 30-40 year olds, health-conscious.

Step 3: Choose channels.
- Instagram (primary, high engagement with 18-30)
- TikTok (primary for short fitness content)
- Influencer partnerships (local fitness creators)

Step 4: Budget allocation.
- 40% paid social
- 30% influencers
- 20% content production
- 10% events

Step 5: KPIs.
- Downloads, 30-day retention, CAC, LTV.
"""

COT_AFTER_INTERRUPT = """
The user now wants to focus on Cairo only (not all of Egypt).

Step 1 (revised): Analyze the Cairo market only.
- Cairo metro: ~22M people, ~65% under 30.
- Fitness app penetration: still ~15% YoY growth.
- Drop the all-Egypt framing. Focus on Cairo districts: Maadi, Zamalek, New Cairo, 6th of October.

Step 2 (revised): Refine audience to Cairo.
- Primary: 18-30 year olds in Cairo, urban, middle-to-upper income.
- Drop the "all major cities" assumption.

Step 3: Keep channels (Instagram/TikTok work in Cairo).
- Same channel mix. Just tighten geo-targeting.

Step 4: Reallocate budget.
- Drop Alexandria spend, double down on Cairo social.
- 50% paid social (Cairo geo-targeting)
- 30% influencers (Cairo-based)
- 20% content production

Step 5: Same KPIs.
"""


# --------------------------------------------------------------------------- #
# Mock fixtures so the demo is deterministic and runs offline
# --------------------------------------------------------------------------- #

MOCK_FIXTURES = {
    "Analyze the market": {
        "goal": GOAL,
        "steps_completed": [
            {"step_id": 1, "description": "Analyze the market: Egypt has 110M people, 60% under 30, growing fitness app adoption", "status": "completed", "dependencies": []},
            {"step_id": 2, "description": "Define the audience: 18-30 year olds, urban, middle income, health-conscious", "status": "completed", "dependencies": [1]},
            {"step_id": 3, "description": "Choose channels: Instagram and TikTok for 18-30 reach, plus influencers", "status": "completed", "dependencies": [2]},
        ],
        "steps_pending": [
            {"step_id": 4, "description": "Allocate budget across paid social, influencers, content", "status": "pending", "dependencies": [3]},
            {"step_id": 5, "description": "Define KPIs: downloads, retention, CAC, LTV", "status": "pending", "dependencies": [3, 4]},
        ],
        "assumptions": [
            "Egypt has 110M people, 60% under 30",
            "Fitness app market is growing 15% YoY",
            "Target all major cities including Cairo",
        ],
        "decisions": [
            {"decision": "Target 18-30 urban audience", "rationale": "Largest growing segment with fitness app adoption"},
            {"decision": "Use Instagram and TikTok as primary channels", "rationale": "High engagement with 18-30 demographic"},
        ],
        "confidence": 0.78,
    },
    "Cairo market only": {
        "goal": GOAL,
        "steps_completed": [
            {"step_id": 1, "description": "Analyze the market: Cairo metro 22M people, 65% under 30, growing fitness app adoption", "status": "completed", "dependencies": []},
            {"step_id": 2, "description": "Define the audience: 18-30 year olds, urban, middle-to-upper income, health-conscious", "status": "completed", "dependencies": [1]},
            {"step_id": 3, "description": "Choose channels: Instagram and TikTok work well in Cairo, plus local influencers", "status": "completed", "dependencies": [2]},
        ],
        "steps_pending": [
            {"step_id": 4, "description": "Reallocate budget toward Cairo geo-targeting", "status": "pending", "dependencies": [3]},
            {"step_id": 5, "description": "Keep the same KPIs: downloads, retention, CAC, LTV", "status": "pending", "dependencies": [3, 4]},
        ],
        "assumptions": [
            "Cairo metro has 22M people, 65% under 30",
            "Fitness app adoption growing 15% YoY in Cairo",
            "Target Cairo districts: Maadi, Zamalek, New Cairo",
        ],
        "decisions": [
            {"decision": "Target 18-30 urban audience in Cairo only", "rationale": "User explicitly requested Cairo-only focus"},
            {"decision": "Keep Instagram and TikTok as primary channels", "rationale": "Channels still work for the narrower geo"},
        ],
        "confidence": 0.82,
    },
}


# --------------------------------------------------------------------------- #
# The demo
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
        print(f"    {s.step_id}. {s.description}  (deps={s.dependencies})")
    print(f"  pending steps:")
    for s in sv.steps_pending:
        print(f"    {s.step_id}. {s.description}  (deps={s.dependencies})")
    print(f"  assumptions: {sv.assumptions}")
    print(f"  decisions:   {[d.decision for d in sv.decisions]}")
    print(f"  confidence:  {sv.confidence}")


def run(llm: LLMClient) -> None:
    extractor = StateExtractor(llm)

    banner("STEP 1 — Extract S(t) BEFORE user interrupt")
    s_before, lat1 = measure_adaptation_latency(
        extractor.extract, GOAL, COT_BEFORE_INTERRUPT
    )
    show_state("S(t)  before", s_before)
    print(f"\n  extraction latency: {lat1.seconds * 1000:.1f} ms")

    banner("STEP 2 — User interrupts: 'focus on Cairo only'")
    print("\n[USER] ركّز على القاهرة بس، مش كل مصر")
    print("       (focus on Cairo only, not all of Egypt)")

    banner("STEP 3 — Extract S(t+1) AFTER interrupt")
    s_after, lat2 = measure_adaptation_latency(
        extractor.extract, GOAL, COT_AFTER_INTERRUPT
    )
    show_state("S(t+1) after", s_after)
    print(f"\n  extraction latency: {lat2.seconds * 1000:.1f} ms")

    banner("STEP 4 — Compute RPR (M1 metric from the survey)")
    print("\n  Two match modes are reported:")
    print("    - 'exact'  : strict string match (legacy, brittle to rewording)")
    print("    - 'jaccard': token overlap ≥ 0.3 (recommended for real CoT)")

    score_exact = rpr(s_before, s_after, match="exact")
    score_jaccard = rpr(s_before, s_after, match="jaccard")

    print(f"\n  RPR[exact]   = {score_exact:.4f}  ({score_exact * 100:.1f}%)")
    print(f"  RPR[jaccard] = {score_jaccard:.4f}  ({score_jaccard * 100:.1f}%)")
    print(
        f"\n  interpretation: of the {len(s_before.steps_completed) + len(s_before.assumptions) + len(s_before.decisions)} "
        f"reasoning elements in S(t), jaccard-mode finds {int(score_jaccard * (len(s_before.steps_completed) + len(s_before.assumptions) + len(s_before.decisions)))} "
        f"preserved in S(t+1)."
    )

    banner("DONE")
    print("\nNext steps for the prototype:")
    print("  1. Wire this into LangGraph as a node (G1: continuous interruption)")
    print("  2. Build ΔU parser — convert user input into structured update intent")
    print("  3. Build f(S, ΔU) — the merge function with α+β+γ weights")
    print("  4. Build the benchmark: 50 tasks × 3 interruption densities")
    print()


def main() -> None:
    mode = "MOCK (no API key)" if not os.getenv("OPENAI_API_KEY") else "REAL (OpenAI)"
    banner(f"LARS S(t) Extractor — Demo [{mode}]")
    print("\n  Goal: prove the extractor produces a usable StateVector")
    print("        and that RPR can be measured across an interrupt.\n")

    llm = MockLLM(fixtures=MOCK_FIXTURES) if not os.getenv("OPENAI_API_KEY") else default_client()
    run(llm)


if __name__ == "__main__":
    main()
