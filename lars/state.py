"""
state.py — The State Vector S(t)

This is the core data structure of LARS. It captures the *structured*
state of an LLM's reasoning at time t, so that reasoning can be
preserved, merged, and updated when the user interrupts.

Schema (from the gap survey, G3):

    S(t) = {
        goal:            the original goal G(t)
        steps_completed: list of ReasoningStep with status="completed"
        steps_pending:   list of ReasoningStep with status="pending"
        assumptions:     facts the model is taking for granted
        decisions:       (decision, rationale) pairs made so far
        confidence:      float in [0, 1]
        version:         monotonic int, bumped on every update
        timestamp:       wall clock at extraction
    }
"""

from __future__ import annotations

from time import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ReasoningStep(BaseModel):
    """A single step in the chain-of-thought."""

    step_id: int = Field(..., ge=1, description="1-based ordering within the trace")
    description: str = Field(..., min_length=1, description="What this step does")
    status: Literal["completed", "in_progress", "pending"] = "pending"
    dependencies: list[int] = Field(
        default_factory=list, description="step_ids this step depends on"
    )


class Decision(BaseModel):
    """A commitment the model has made during reasoning."""

    decision: str = Field(..., description="The choice that was made")
    rationale: str = Field(..., description="Why this choice was made")


class StateVector(BaseModel):
    """
    S(t) — the structured representation of an LLM's reasoning state.

    This is what gets:
      - Extracted from raw CoT (extractor.py)
      - Compared for preservation (metrics.py: rpr)
      - Merged with user input (future: f(S, ΔU))
    """

    goal: str = Field(..., min_length=1, description="The original user goal G(t)")
    steps_completed: list[ReasoningStep] = Field(default_factory=list)
    steps_pending: list[ReasoningStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    version: int = 1
    timestamp: float = Field(default_factory=time)
    raw_cot: Optional[str] = Field(
        default=None, description="Original CoT (kept for debugging / replay)"
    )

    @field_validator("confidence")
    @classmethod
    def _round_confidence(cls, v: float) -> float:
        return round(v, 3)

    # ---- Convenience helpers (no I/O) ----

    def all_step_ids(self) -> list[int]:
        return [s.step_id for s in self.steps_completed] + [
            s.step_id for s in self.steps_pending
        ]

    def step_descriptions(self) -> list[str]:
        return [s.description for s in self.steps_completed] + [
            s.description for s in self.steps_pending
        ]

    def summary(self) -> str:
        """One-line summary for logging / UI."""
        n_done = len(self.steps_completed)
        n_pend = len(self.steps_pending)
        return (
            f"S(v={self.version}) goal='{self.goal[:40]}...' "
            f"steps={n_done}✓/{n_pend}⏳ conf={self.confidence:.2f}"
        )

    def bumped(self) -> "StateVector":
        """Return a copy with version+1 and fresh timestamp."""
        return self.model_copy(update={"version": self.version + 1, "timestamp": time()})
