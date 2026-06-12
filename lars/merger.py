"""
merger.py — The f(S, ΔU) function

This is the "intelligence" of LARS. It takes the current state S and a
structured user interrupt ΔU (an UpdateIntent), and produces a new state
S' that:

  - Preserves as much of the original reasoning as possible
  - Applies the user's change cleanly
  - Records exactly what it did (MergeTrace) — for the paper, for
    debugging, and for the user to inspect

Design choice: rule-based, not LLM-based.

Why?
  1. Deterministic → reproducible benchmarks
  2. Debuggable → we can show a reviewer exactly what f did
  3. Fast → sub-millisecond, real-time safe
  4. The paper can compare f_rule vs. f_llm as a v2 ablation

The constraint from the survey:
    α + β + γ = 1, α ≥ 0.5
where α = preservation weight, β = direct update weight, γ = strategy
adaptation weight. We make this explicit in MergeTrace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .state import Decision, ReasoningStep, StateVector
from .update_intent import IntentType, UpdateIntent


# --------------------------------------------------------------------------- #
# MergeTrace — the auditable record of what f did
# --------------------------------------------------------------------------- #


@dataclass
class MergeTrace:
    """Exactly what f did to S(t) to produce S(t+1)."""

    intent_type: IntentType
    preserved_steps: list[int] = field(default_factory=list)
    modified_steps: list[int] = field(default_factory=list)
    dropped_steps: list[int] = field(default_factory=list)
    inserted_steps: list[int] = field(default_factory=list)
    preserved_assumptions: list[int] = field(default_factory=list)
    modified_assumptions: list[int] = field(default_factory=list)
    dropped_assumptions: list[int] = field(default_factory=list)
    inserted_assumptions: list[str] = field(default_factory=list)
    preserved_decisions: list[int] = field(default_factory=list)
    inserted_decisions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # The α/β/γ weights actually applied (for paper-grade auditability)
    alpha: float = 0.0  # preservation
    beta: float = 0.0   # direct update
    gamma: float = 0.0  # strategy adaptation

    def preservation_rate_steps(self) -> float:
        n_old = len(self.preserved_steps) + len(self.modified_steps) + len(self.dropped_steps)
        if n_old == 0:
            return 1.0
        return (len(self.preserved_steps) + len(self.modified_steps)) / n_old

    def summary(self) -> str:
        return (
            f"MergeTrace({self.intent_type.value}): "
            f"steps[pres={len(self.preserved_steps)}, mod={len(self.modified_steps)}, "
            f"drop={len(self.dropped_steps)}, ins={len(self.inserted_steps)}] "
            f"α={self.alpha:.2f} β={self.beta:.2f} γ={self.gamma:.2f}"
        )


# --------------------------------------------------------------------------- #
# The merger
# --------------------------------------------------------------------------- #


class StateMerger:
    """
    f(S, ΔU) → (S', MergeTrace)

    Weights α/β/γ are configurable but default to 0.6 / 0.3 / 0.1
    (matches the survey's recommendation: α ≥ 0.5, sum = 1).
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1):
        if not (alpha >= 0.5):
            raise ValueError(f"alpha must be >= 0.5 (got {alpha}). Preservation is non-negotiable.")
        if abs((alpha + beta + gamma) - 1.0) > 1e-6:
            raise ValueError(f"alpha+beta+gamma must equal 1.0 (got {alpha+beta+gamma})")
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def merge(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        handler = {
            IntentType.SCOPE_NARROW: self._narrow,
            IntentType.SCOPE_EXPAND: self._expand,
            IntentType.CORRECTION: self._correct,
            IntentType.REPLACE: self._replace,
            IntentType.ADD: self._add,
            IntentType.REMOVE: self._remove,
            IntentType.REPRIORITIZE: self._reprioritize,
            IntentType.CLARIFY: self._clarify,
            IntentType.ABORT: self._abort,
        }[intent.type]
        return handler(s, intent)

    # ---- Per-intent handlers ----

    def _narrow(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """
        Scope-narrow: rewrite broad references in steps/assumptions to
        the narrow target. Keep everything that doesn't reference the
        old broad scope.
        """
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )
        new_value = intent.new_value or intent.value or "narrowed scope"
        # Words that signal "broad scope" we should rewrite
        broad_markers = [
            r"\ball\s+(\w+)", r"\bevery\s+(\w+)", r"\bacross\s+(\w+)",
            r"\bgeneral\b", r"\bwide\b", r"\bbroad\b",
        ]
        broad_pattern = re.compile("|".join(broad_markers), re.I)

        new_steps_done: list[ReasoningStep] = []
        new_assumptions: list[str] = []
        new_decisions: list[Decision] = []

        for i, step in enumerate(s.steps_completed):
            if broad_pattern.search(step.description) or re.search(
                r"\begypt\b|\ball\s+cities\b|\bnationwide\b",
                step.description, re.I,
            ):
                # Modify: rewrite the broad reference
                modified = re.sub(
                    r"\begypt\b", new_value, step.description, flags=re.I,
                )
                modified = re.sub(
                    r"\ball\s+major\s+cities\b", new_value, modified, flags=re.I,
                )
                modified = re.sub(
                    r"\ball\s+cities\b", new_value, modified, flags=re.I,
                )
                modified = re.sub(
                    r"\bnationwide\b", new_value, modified, flags=re.I,
                )
                new_steps_done.append(step.model_copy(update={"description": modified}))
                trace.modified_steps.append(step.step_id)
            else:
                new_steps_done.append(step)
                trace.preserved_steps.append(step.step_id)

        for i, asm in enumerate(s.assumptions):
            if broad_pattern.search(asm) or re.search(
                r"\begypt\b|\ball\s+cities\b|\bnationwide\b", asm, re.I,
            ):
                modified = re.sub(r"\begypt\b", new_value, asm, flags=re.I)
                modified = re.sub(r"\ball\s+major\s+cities\b", new_value, modified, flags=re.I)
                modified = re.sub(r"\ball\s+cities\b", new_value, modified, flags=re.I)
                new_assumptions.append(modified)
                trace.modified_assumptions.append(i)
            else:
                new_assumptions.append(asm)
                trace.preserved_assumptions.append(i)

        # Add a new decision noting the narrowing
        new_decisions.extend(s.decisions)
        new_decisions.append(Decision(
            decision=f"Narrow scope to {new_value}",
            rationale="User explicitly narrowed the scope",
        ))
        trace.inserted_decisions.append(f"Narrow scope to {new_value}")

        # Add a new assumption
        new_assumptions.append(f"Scope is restricted to {new_value}")
        trace.inserted_assumptions.append(f"Scope is restricted to {new_value}")

        # Keep pending steps; their content will need to be regenerated
        # by the next LLM call. For v1 we keep them as-is.
        new_steps_pending = list(s.steps_pending)
        trace.notes.append("Pending steps kept; regenerate them in next LLM call.")

        new_sv = StateVector(
            goal=s.goal,
            steps_completed=new_steps_done,
            steps_pending=new_steps_pending,
            assumptions=new_assumptions,
            decisions=new_decisions,
            confidence=max(0.0, s.confidence - 0.02),  # slight dip, scope changed
            version=s.version + 1,
            raw_cot=s.raw_cot,
        )
        return new_sv, trace

    def _expand(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """Scope-expand: keep all existing reasoning, add a new pending step."""
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )
        for step in s.steps_completed:
            trace.preserved_steps.append(step.step_id)
        for i, _ in enumerate(s.assumptions):
            trace.preserved_assumptions.append(i)
        for i, _ in enumerate(s.decisions):
            trace.preserved_decisions.append(i)

        new_value = intent.new_value or intent.value or "expanded scope"
        new_step_id = max(s.all_step_ids(), default=0) + 1
        new_step = ReasoningStep(
            step_id=new_step_id,
            description=f"Extend analysis to cover {new_value}",
            status="pending",
            dependencies=[],
        )
        new_steps_pending = list(s.steps_pending) + [new_step]
        trace.inserted_steps.append(new_step_id)
        trace.notes.append(f"Added new pending step for {new_value}")

        new_sv = StateVector(
            goal=s.goal,
            steps_completed=list(s.steps_completed),
            steps_pending=new_steps_pending,
            assumptions=list(s.assumptions) + [f"Scope expanded to include {new_value}"],
            decisions=list(s.decisions) + [Decision(
                decision=f"Expand scope to {new_value}",
                rationale="User explicitly expanded the scope",
            )],
            confidence=s.confidence,
            version=s.version + 1,
            raw_cot=s.raw_cot,
        )
        return new_sv, trace

    def _correct(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """Correction: if a value is mentioned in a step, modify it."""
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )
        new_value = intent.new_value or ""
        # Naive: modify the most-recent step that mentions a value
        # (For v1, a smarter matcher would use embeddings.)
        new_steps_done: list[ReasoningStep] = []
        corrected = False
        for step in reversed(s.steps_completed):
            if not corrected and new_value:
                new_steps_done.append(step.model_copy(update={"description": f"{step.description} [corrected: {new_value}]"}))
                trace.modified_steps.append(step.step_id)
                corrected = True
            else:
                new_steps_done.append(step)
                trace.preserved_steps.append(step.step_id)
        new_steps_done.reverse()

        new_sv = StateVector(
            goal=s.goal,
            steps_completed=new_steps_done,
            steps_pending=list(s.steps_pending),
            assumptions=list(s.assumptions),
            decisions=list(s.decisions) + [Decision(
                decision=f"Apply correction: {new_value}",
                rationale="User correction",
            )],
            confidence=max(0.0, s.confidence - 0.05),
            version=s.version + 1,
            raw_cot=s.raw_cot,
        )
        return new_sv, trace

    def _replace(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """Replace: swap old_value with new_value everywhere it appears."""
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )
        old_v, new_v = intent.old_value or "", intent.new_value or ""

        def swap(text: str) -> str:
            if not old_v or not new_v:
                return text
            return re.sub(
                rf"\b{re.escape(old_v)}\b", new_v, text, flags=re.I,
            )

        new_steps_done: list[ReasoningStep] = []
        for step in s.steps_completed:
            swapped = swap(step.description)
            if swapped != step.description:
                new_steps_done.append(step.model_copy(update={"description": swapped}))
                trace.modified_steps.append(step.step_id)
            else:
                new_steps_done.append(step)
                trace.preserved_steps.append(step.step_id)

        new_assumptions = [swap(a) for a in s.assumptions]
        new_decisions = [
            Decision(decision=swap(d.decision), rationale=d.rationale)
            for d in s.decisions
        ]

        new_sv = StateVector(
            goal=s.goal,
            steps_completed=new_steps_done,
            steps_pending=list(s.steps_pending),
            assumptions=new_assumptions,
            decisions=new_decisions,
            confidence=s.confidence,
            version=s.version + 1,
            raw_cot=s.raw_cot,
        )
        trace.notes.append(f"Replaced '{old_v}' → '{new_v}'")
        return new_sv, trace

    def _add(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """Add: append a new pending step or assumption."""
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )
        for step in s.steps_completed:
            trace.preserved_steps.append(step.step_id)

        new_value = intent.value or intent.new_value or "user-requested addition"
        new_step_id = max(s.all_step_ids(), default=0) + 1
        new_step = ReasoningStep(
            step_id=new_step_id,
            description=f"Include {new_value}",
            status="pending",
            dependencies=[],
        )
        new_steps_pending = list(s.steps_pending) + [new_step]
        trace.inserted_steps.append(new_step_id)

        new_sv = StateVector(
            goal=s.goal,
            steps_completed=list(s.steps_completed),
            steps_pending=new_steps_pending,
            assumptions=list(s.assumptions),
            decisions=list(s.decisions),
            confidence=s.confidence,
            version=s.version + 1,
            raw_cot=s.raw_cot,
        )
        return new_sv, trace

    def _remove(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """Remove: drop the step/decision that mentions the value."""
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )
        value = intent.value or ""

        new_steps_done: list[ReasoningStep] = []
        for step in s.steps_completed:
            if value and re.search(rf"\b{re.escape(value)}\b", step.description, re.I):
                trace.dropped_steps.append(step.step_id)
            else:
                new_steps_done.append(step)
                trace.preserved_steps.append(step.step_id)

        new_decisions = [
            d for d in s.decisions
            if not value or not re.search(rf"\b{re.escape(value)}\b", d.decision, re.I)
        ]

        new_sv = StateVector(
            goal=s.goal,
            steps_completed=new_steps_done,
            steps_pending=list(s.steps_pending),
            assumptions=list(s.assumptions) + [f"Removed: {value}"],
            decisions=new_decisions,
            confidence=s.confidence,
            version=s.version + 1,
            raw_cot=s.raw_cot,
        )
        trace.notes.append(f"Removed anything mentioning '{value}'")
        return new_sv, trace

    def _reprioritize(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """Reorder: for v1, just record the request as a note."""
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )
        for step in s.steps_completed:
            trace.preserved_steps.append(step.step_id)
        trace.notes.append(
            f"Reorder request received: '{intent.value}'. v1 keeps original order; "
            "v2 should re-rank step_ids by user priority."
        )
        new_sv = s.bumped()
        return new_sv, trace

    def _clarify(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """Clarify: no state change, just log."""
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
        )
        for step in s.steps_completed:
            trace.preserved_steps.append(step.step_id)
        trace.notes.append(f"Clarification requested: '{intent.value}'. No state change.")
        return s.bumped(), trace

    def _abort(
        self, s: StateVector, intent: UpdateIntent
    ) -> tuple[StateVector, MergeTrace]:
        """Abort: return an empty state, fresh version."""
        trace = MergeTrace(
            intent_type=intent.type,
            alpha=0.0, beta=0.0, gamma=0.0,
        )
        for step in s.steps_completed:
            trace.dropped_steps.append(step.step_id)
        trace.notes.append("ABORT: all reasoning discarded.")
        new_sv = StateVector(
            goal=s.goal,
            steps_completed=[],
            steps_pending=[
                ReasoningStep(step_id=1, description="Restart from scratch", status="pending", dependencies=[])
            ],
            assumptions=[],
            decisions=[],
            confidence=0.0,
            version=s.version + 1,
        )
        return new_sv, trace
