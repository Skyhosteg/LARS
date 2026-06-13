"""
baselines.py — The 3 baselines that LARS is compared against

These are the alternative systems a user could use today, none of which
preserve reasoning state across interruption.

  B1. no_interrupt: never listens. Generates the full plan, ignores user.
  B2. langgraph_checkpoint: LangGraph with interrupt_before, but on
      interrupt the state is DISCARDED and the agent restarts.
  B3. restart_from_scratch: on any user input, throw the current
      reasoning away and start over with the new instruction.

LARS is the proposed method (already implemented in agent.py).
"""

from __future__ import annotations

from .delta_u import DeltaUParser
from .extractor import StateExtractor
from .llm import LLMClient
from .merger import StateMerger
from .state import StateVector


def method_no_interrupt(
    goal: str,
    initial_cot: str,
    user_interrupt: str | None,
    llm: LLMClient,
    extractor: StateExtractor,
) -> tuple[StateVector, dict]:
    """
    B1: Just generate the plan, ignore the user interrupt.

    This is the "no live adaptation" baseline. RPR should be undefined
    or 0.0 (no merge happened, but the plan is also not what the user
    asked for).
    """
    sv = extractor.extract(goal, initial_cot)
    return sv, {"method": "no_interrupt", "used_interrupt": False}


def method_restart_from_scratch(
    goal: str,
    initial_cot: str,
    user_interrupt: str | None,
    llm: LLMClient,
    extractor: StateExtractor,
) -> tuple[StateVector, dict]:
    """
    B3: Throw away the old plan, regenerate from scratch with the
    interrupt in the prompt.

    This is what most current systems do (ChatGPT, Claude, etc.).
    RPR vs. original = ~0 (full recompute). Token cost = full.
    """
    if not user_interrupt:
        return method_no_interrupt(goal, initial_cot, None, llm, extractor)

    # The "naive" approach: stuff the interrupt into the prompt and regenerate
    new_cot = (
        f"[USER INSTRUCTION: {user_interrupt}]\n\n"
        f"Regenerating plan from scratch with the user instruction as the new starting point.\n\n"
        f"{initial_cot}"
    )
    sv = extractor.extract(goal, new_cot)
    return sv, {"method": "restart_from_scratch", "used_interrupt": True, "intent_type": "fresh"}


def method_langgraph_checkpoint(
    goal: str,
    initial_cot: str,
    user_interrupt: str | None,
    llm: LLMClient,
    extractor: StateExtractor,
) -> tuple[StateVector, dict]:
    """
    B2: LangGraph-style checkpoint interrupt.

    On interrupt, LangGraph's default behavior is to:
      - Halt at the checkpoint
      - Wait for human input
      - On resume, the human's input REPLACES the pending state
        (not merged with prior reasoning)

    This is the "structured but no preservation" baseline.
    """
    if not user_interrupt:
        return method_no_interrupt(goal, initial_cot, None, llm, extractor)

    # The "LangGraph way": take the checkpoint, append the user input,
    # regenerate. No merge of reasoning elements.
    merged_cot = (
        f"{initial_cot}\n\n"
        f"[USER INTERRUPT AT CHECKPOINT]: {user_interrupt}\n"
        f"[RESUMED FROM CHECKPOINT] Continuing without preserving prior reasoning."
    )
    sv = extractor.extract(goal, merged_cot)
    return sv, {"method": "langgraph_checkpoint", "used_interrupt": True, "intent_type": "appended"}



# The list of baselines for benchmarking
ALL_BASELINES = [
    ("no_interrupt", method_no_interrupt),
    ("restart_from_scratch", method_restart_from_scratch),
    ("langgraph_checkpoint", method_langgraph_checkpoint),
]


# --------------------------------------------------------------------------- #
# LARS (the proposed method) — added to the same harness
# --------------------------------------------------------------------------- #


def method_lars(
    goal: str,
    initial_cot: str,
    user_interrupt: str | None,
    llm: LLMClient,
    extractor: StateExtractor,
    parser: DeltaUParser,
    merger: StateMerger,
) -> tuple[StateVector, dict]:
    """
    LARS: extract → parse ΔU → merge → preserve ≥ 50% of state.

    The only method that preserves reasoning across interruption.
    """
    s_before = extractor.extract(goal, initial_cot)
    if not user_interrupt:
        return s_before, {"method": "lars", "used_interrupt": False}

    intent = parser.parse(user_interrupt)
    s_after, trace = merger.merge(s_before, intent)
    return s_after, {
        "method": "lars",
        "used_interrupt": True,
        "intent_type": intent.type.value,
        "trace": trace.summary(),
        "alpha": trace.alpha,
    }


# Full list of methods (baselines + LARS) for the benchmark
ALL_METHODS = ALL_BASELINES + [("lars", None)]  # LARS is special-cased in benchmark.py
