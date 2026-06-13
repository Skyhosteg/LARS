"""
examples/demo_live.py — Interactive live demo of LARS

Walks the user through a live session:
  1. They enter a goal
  2. LARS extracts an initial plan
  3. LARS executes each step, pausing after each one
  4. The user can interrupt at any pause with a free-form text
  5. ΔU parser + f + MergeTrace fire and display in real time
  6. RPR is computed after each step

This is the *proof* that LARS works as a live system — not just a
post-hoc analysis tool.

Run:
    python examples/demo_live.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lars.agent import LiveAgent
from lars.delta_u import DeltaUParserMock
from lars.executor import MockStepExecutor
from lars.extractor import StateExtractor
from lars.llm import MockLLM, OpenAILLM, default_client
from lars.merger import StateMerger


# --------------------------------------------------------------------------- #
# Canned CoT for the demo: each step's reasoning, in order
# --------------------------------------------------------------------------- #

INITIAL_COT = """
The user wants a marketing plan for a fitness app in Egypt.

Step 1: Analyze the market.
Step 2: Define the audience.
Step 3: Choose channels.
Step 4: Allocate budget.
Step 5: Define KPIs.
"""

CANNED_STEP_COTS = {
    1: (
        "Egypt has 110M people, 60% under 30, growing fitness app "
        "adoption at 15% YoY. Targeting all major cities: Cairo, "
        "Alexandria, Giza. Competitors: Gymondo, local fragmented apps."
    ),
    2: (
        "Primary audience: 18-30 year olds, urban, middle income, "
        "health-conscious. Secondary: 30-40 year olds with disposable "
        "income for premium plans."
    ),
    3: (
        "Channels: A multi-platform mix to maximize reach — primary "
        "social platforms (40% engagement with 18-30), short-form "
        "video (35% reach for short content), and local fitness "
        "influencers (25% trust). Paid ads supplement organic reach."
    ),
    4: (
        "Budget split: 40% paid social, 30% influencers, 20% content "
        "production, 10% events and sponsorships. Total Q1 budget: "
        "$50K."
    ),
    5: (
        "Primary KPIs: downloads, 30-day retention, CAC, LTV. "
        "Target for Q1: 100K downloads, 25% retention, CAC under $5, "
        "LTV over $40."
    ),
}


INITIAL_FIXTURE = {
    "marketing plan": {
        "goal": "Create a marketing plan for a fitness app in Egypt",
        "steps_completed": [],
        "steps_pending": [
            {"step_id": 1, "description": "Analyze the market for a fitness app in Egypt (Cairo, Alexandria, Giza)", "status": "pending", "dependencies": []},
            {"step_id": 2, "description": "Define the audience: 18-30 year olds, urban, middle income, in Egypt", "status": "pending", "dependencies": [1]},
            {"step_id": 3, "description": "Choose channels: Instagram, TikTok, and Facebook ads for 18-30 reach, plus local fitness influencers in Egypt", "status": "pending", "dependencies": [2]},
            {"step_id": 4, "description": "Allocate budget across paid social, influencers, content production in Egypt", "status": "pending", "dependencies": [3]},
            {"step_id": 5, "description": "Define KPIs: downloads, 30-day retention, CAC, LTV for the Egypt market", "status": "pending", "dependencies": [3, 4]},
        ],
        "assumptions": [
            "Egypt has 110M people, 60% under 30",
            "Fitness app adoption growing 15% YoY in Egypt",
            "Target all major cities including Cairo and Alexandria",
        ],
        "decisions": [
            {"decision": "Target 18-30 urban audience in Egypt", "rationale": "Largest growing segment with fitness app adoption"},
            {"decision": "Use Instagram and TikTok as primary channels", "rationale": "High engagement with 18-30 demographic"},
        ],
        "confidence": 0.5,
    }
}


# --------------------------------------------------------------------------- #
# Pre-baked interrupt scripts (the user can pick one of these for a guided demo)
# --------------------------------------------------------------------------- #

GUIDED_INTERRUPTS = {
    1: "focus on Cairo only",                    # SCOPE_NARROW at step 1
    2: "actually make the audience 25-40",       # CORRECTION at step 2
    3: "use Twitter instead of Facebook",        # REPLACE at step 3
    4: "drop the events line",                   # REMOVE at step 4
    5: "stop, restart from scratch",             # ABORT at step 5
}


# --------------------------------------------------------------------------- #
# The demo
# --------------------------------------------------------------------------- #


def banner(s: str) -> None:
    print()
    print("#" * 78)
    print(f"### {s}")
    print("#" * 78)


def main() -> None:
    banner("LARS — LIVE ADAPTIVE REASONING SYSTEM (interactive demo)")
    print()
    print("This is the live runtime. You enter a goal, LARS plans, then")
    print("executes step by step. At each pause you can interrupt freely.")
    print()
    print("Suggested interrupts (try them at any pause):")
    for step, text in GUIDED_INTERRUPTS.items():
        print(f"  after step {step}:  '{text}'")
    print()
    print("Or just press Enter to continue without interrupting.")
    print()

    # 1. Get the goal
    goal = input("> Enter your goal: ").strip()
    if not goal:
        goal = "Create a marketing plan for a fitness app targeting young adults in Egypt."
        print(f"  (using default goal: {goal})")

    # 2. Set up the LLM-backed components
    if os.getenv("OPENROUTER_API_KEY"):
        from lars.llm import OpenRouterLLM
        llm = OpenRouterLLM()
        backend = "OpenRouter"
        if os.getenv("OPENROUTER_MODEL"):
            print(f"\n[setup] Using {backend} LLM: {os.getenv('OPENROUTER_MODEL')}")
        else:
            print(f"\n[setup] Using {backend} LLM (default: openai/gpt-4o-mini)")
    elif os.getenv("OPENAI_API_KEY"):
        llm = OpenAILLM()
        backend = "OpenAI"
        print(f"\n[setup] Using {backend} LLM (real).")
    else:
        llm = MockLLM(fixtures=INITIAL_FIXTURE)
        backend = "Mock"
        print("\n[setup] Using MockLLM (no API key set).")

    extractor = StateExtractor(llm)
    # Use real LLM executor when API key is set, mock otherwise
    if backend in ("OpenAI", "OpenRouter"):
        from lars.executor import LLMStepExecutor
        executor = LLMStepExecutor(llm=llm)
    else:
        executor = MockStepExecutor(canned=CANNED_STEP_COTS)
    parser = DeltaUParserMock()  # heuristic; for Arabic/etc use DeltaUParserLLM
    merger = StateMerger(alpha=0.6, beta=0.3, gamma=0.1)

    # 3. Run the live agent
    banner("LIVE SESSION")
    agent = LiveAgent(
        extractor=extractor,
        executor=executor,
        parser=parser,
        merger=merger,
        initial_cot=INITIAL_COT,
    )
    final_state = agent.run(goal, max_steps=5)

    banner("FINAL STATE")
    print(f"\n  Goal: {final_state.goal}")
    print(f"  Version: {final_state.version}")
    print(f"  Confidence: {final_state.confidence:.2f}")
    print(f"  Completed steps: {len(final_state.steps_completed)}/{len(final_state.steps_completed) + len(final_state.steps_pending)}")
    print(f"\n  Completed:")
    for st in final_state.steps_completed:
        print(f"    {st.step_id}. {st.description}")
    print(f"\n  Pending:")
    for st in final_state.steps_pending:
        print(f"    {st.step_id}. {st.description}")
    if final_state.decisions:
        print(f"\n  Decisions made along the way:")
        for d in final_state.decisions:
            print(f"    - {d.decision}")

    banner("DONE")
    print("\nThis is what LARS does:")
    print("  - Plans a multi-step reasoning trajectory")
    print("  - Executes each step (with optional real-LLM backend)")
    print("  - Pauses after each step to listen for the user")
    print("  - On interrupt, parses ΔU, applies the 3-layer merge pipeline")
    print("  - Continues from the new state, not from scratch")
    print()
    print("Next steps for you:")
    print("  - LangGraph deployment:  pip install langgraph && python -m lars.langgraph_integration")
    print("  - Full 12-task benchmark: python examples/run_benchmark.py")
    print("  - Run all 33 tests:      python tests/test_merge.py && python tests/test_extractor.py && python tests/test_agent.py && python tests/test_benchmark.py")
    print("  - Read the v3 paper:     see lars_v3_paper.md")
    print()


if __name__ == "__main__":
    main()
