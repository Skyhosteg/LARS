"""
executor.py — The step executor (turns "plan" into "CoT")

The LiveAgent walks the reasoning plan one step at a time. For each
pending step, the executor produces the actual CoT text for that step.

Two backends:
  - MockStepExecutor  : returns canned CoT per step (for the demo, no LLM)
  - LLMStepExecutor   : calls the LLM to generate the step's CoT

In a full LARS system, this is where the real reasoning happens. For the
prototype, we keep it deterministic so the demo is reproducible.
"""

from __future__ import annotations

from .llm import LLMClient
from .state import ReasoningStep, StateVector


class StepExecutor:
    def execute(self, s: StateVector, step: ReasoningStep) -> str:
        raise NotImplementedError


class MockStepExecutor(StepExecutor):
    """
    Returns a canned CoT snippet for each step. The snippet describes
    what would happen if a real LLM was reasoning on this step.

    Steps not in `canned` get a generic stub so the demo can still
    progress.
    """

    def __init__(self, canned: dict[int, str] | None = None):
        self.canned = canned or {}

    def execute(self, s: StateVector, step: ReasoningStep) -> str:
        if step.step_id in self.canned:
            return self.canned[step.step_id]
        return (
            f"[mock CoT for step {step.step_id}]\n"
            f"  reasoning about: {step.description}\n"
            f"  in the context of: {s.goal[:60]}"
        )


class LLMStepExecutor(StepExecutor):
    """Real LLM-based executor. Stub for v2 (not used in v1 demo)."""

    def __init__(self, llm: LLMClient, model: str = "gpt-4o-mini"):
        self.llm = llm
        self.model = model

    def execute(self, s: StateVector, step: ReasoningStep) -> str:
        prompt = (
            f"GOAL: {s.goal}\n\n"
            f"CURRENT STATE: {s.summary()}\n\n"
            f"YOUR TASK: Execute step {step.step_id}: {step.description}\n"
            f"Write 2-3 sentences of reasoning. Be concrete and decisive."
        )
        # Raw completion (not JSON) — we just want text
        try:
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return f"[LLM call failed for step {step.step_id}]"
