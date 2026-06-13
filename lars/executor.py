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

    def _format_overrides(self, s: StateVector) -> str:
        """Format the active overrides into a prompt block. Empty string if none."""
        if not s.active_overrides:
            return ""
        lines = "\n".join(f"  - {o}" for o in s.active_overrides)
        return f"\n\nUSER OVERRIDES (apply these to your reasoning):\n{lines}\n"


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
    """Real LLM-based executor. Uses the supplied LLM client (OpenAI or OpenRouter)."""

    def __init__(self, llm: LLMClient, model: str | None = None):
        self.llm = llm
        # If the llm has a model attribute, use it; otherwise default.
        self.model = model or getattr(llm, "model", "gpt-4o-mini")

    def execute(self, s: StateVector, step: ReasoningStep) -> str:
        prompt = (
            f"GOAL: {s.goal}\n\n"
            f"CURRENT STATE: {s.summary()}\n\n"
            f"YOUR TASK: Execute step {step.step_id}: {step.description}\n"
            f"Write 2-3 sentences of reasoning. Be concrete and decisive."
            f"{self._format_overrides(s)}"
        )
        # We use a low-level call here (not the JSON-validated complete_json)
        # because we want free-form text, not structured output.
        try:
            # Delegate to whatever raw client the LLM has.
            from openai import OpenAI
            import os
            # The LLMClient stores its underlying OpenAI client privately;
            # we re-use the env-var path for simplicity.
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
            client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM call failed for step {step.step_id}: {e}]"
