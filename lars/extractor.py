"""
extractor.py — The S(t) Extractor

Takes (goal, raw_cot) → StateVector.

This is the *first* component of LARS. It runs each time the system
needs a snapshot of where the model's reasoning currently stands.

Design notes:
  - The extractor is stateless. Given the same (goal, cot), it should
    produce the same StateVector (modulo timestamp/version).
  - The CoT may be partial (mid-generation) or full (post-hoc). The
    extractor handles both.
  - We deliberately keep the prompt simple. Sophistication belongs in
    the LLM, not the wrapper.
"""

from __future__ import annotations

from .llm import LLMClient
from .prompts import SYSTEM_EXTRACTOR
from .state import StateVector


class StateExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract(self, goal: str, cot: str) -> StateVector:
        """
        Parse a CoT trace into a StateVector.

        Args:
            goal: the original user goal
            cot:  raw chain-of-thought text (may be partial)

        Returns:
            StateVector
        """
        user_msg = (
            f"GOAL:\n{goal}\n\n"
            f"CHAIN-OF-THOUGHT TRACE:\n{cot}\n\n"
            f"Extract the structured state. Return JSON only."
        )
        sv = self.llm.complete_json(SYSTEM_EXTRACTOR, user_msg, StateVector)
        # Preserve the raw CoT for debugging / replay
        return sv.model_copy(update={"raw_cot": cot})
