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
  - v0.5.0 adds refresh_pending() so the merger can see fresh
    descriptions after a state change.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import List

from .llm import LLMClient
from .prompts import SYSTEM_EXTRACTOR
from .state import StateVector


class _RefreshUpdate(BaseModel):
    step_id: int
    new_description: str


class _RefreshResp(BaseModel):
    updates: List[_RefreshUpdate]


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

    def refresh_pending(
        self, s: StateVector, intent_type: str, intent_payload: str
    ) -> StateVector:
        """
        After a merge, re-state the descriptions of pending steps so
        they reflect the new state. This makes future merges accurate.

        Strict mode: only refreshes descriptions; never re-orders or
        adds steps. Uses a small, deterministic LLM call.

        Fails soft: if the LLM hiccups, returns s unchanged.
        """
        if not s.steps_pending:
            return s

        pending_lines = "\n".join(
            f"  - step_id={p.step_id}: {p.description}" for p in s.steps_pending
        )
        user_msg = (
            f"GOAL: {s.goal}\n\n"
            f"RECENT CHANGE: {intent_type} applied - {intent_payload}\n\n"
            f"NEW ASSUMPTIONS:\n  " + "\n  ".join(s.assumptions) + "\n\n"
            f"PENDING STEPS (rewrite each description to reflect the new state):\n"
            f"{pending_lines}\n\n"
            f"Return JSON with updates list."
        )

        system = (
            "You re-state pending step descriptions after a state change. "
            "Each new_description must be 1-2 sentences and include any keywords "
            "(channel names, geo, audiences, metrics) that the step will reason about. "
            "Do NOT reorder or add steps."
        )

        try:
            resp = self.llm.complete_json(system, user_msg, _RefreshResp)
        except Exception:
            return s

        updates = {u.step_id: u.new_description for u in resp.updates}
        new_pending = []
        for p in s.steps_pending:
            if p.step_id in updates:
                new_pending.append(p.model_copy(update={"description": updates[p.step_id]}))
            else:
                new_pending.append(p)
        return s.model_copy(update={
            "steps_pending": new_pending,
            "version": s.version + 1,
        })
