"""
tests/test_extractor.py — Quick smoke tests for the S(t) Extractor + RPR

Run:
    python -m pytest tests/
or just:
    python tests/test_extractor.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lars.llm import MockLLM
from lars.extractor import StateExtractor
from lars.metrics import rpr
from lars.state import StateVector, ReasoningStep, Decision


def make_state(**overrides) -> StateVector:
    """Build a StateVector with sensible defaults for testing."""
    base = dict(
        goal="test goal",
        steps_completed=[
            ReasoningStep(step_id=1, description="step 1", status="completed", dependencies=[]),
            ReasoningStep(step_id=2, description="step 2", status="completed", dependencies=[1]),
        ],
        steps_pending=[
            ReasoningStep(step_id=3, description="step 3", status="pending", dependencies=[2]),
        ],
        assumptions=["assumption A"],
        decisions=[Decision(decision="choice X", rationale="because")],
        confidence=0.7,
    )
    base.update(overrides)
    return StateVector(**base)


def test_extractor_with_mock():
    """Extractor produces a valid StateVector with the MockLLM."""
    llm = MockLLM(fixtures={
        "GOAL": {
            "goal": "do X",
            "steps_completed": [
                {"step_id": 1, "description": "first step", "status": "completed", "dependencies": []}
            ],
            "steps_pending": [
                {"step_id": 2, "description": "second step", "status": "pending", "dependencies": [1]}
            ],
            "assumptions": [],
            "decisions": [],
            "confidence": 0.6,
        }
    })
    extractor = StateExtractor(llm)
    sv = extractor.extract("do X", "Step 1: first step. Now starting step 2.")
    assert isinstance(sv, StateVector)
    assert sv.goal == "do X"
    assert len(sv.steps_completed) == 1
    assert len(sv.steps_pending) == 1
    assert sv.raw_cot is not None  # the extractor should preserve raw CoT
    print("✓ test_extractor_with_mock")


def test_rpr_identical_states():
    """RPR of a state with itself is 1.0."""
    s = make_state()
    assert rpr(s, s) == 1.0
    print("✓ test_rpr_identical_states")


def test_rpr_total_loss():
    """RPR is 0.0 when the new state shares nothing with the old."""
    s_old = StateVector(
        goal="old goal",
        steps_completed=[ReasoningStep(step_id=1, description="unique-old-content-xyz", status="completed", dependencies=[])],
        steps_pending=[],
        assumptions=["unique-old-assumption-xyz"],
        decisions=[Decision(decision="unique-old-decision-xyz", rationale="unique-rationale-xyz")],
        confidence=0.5,
    )
    s_new = StateVector(
        goal="new goal",
        steps_completed=[ReasoningStep(step_id=1, description="totally-different-text-abc", status="completed", dependencies=[])],
        steps_pending=[],
        assumptions=["unique-new-assumption-abc"],
        decisions=[Decision(decision="unique-new-decision-abc", rationale="unique-rationale-abc")],
        confidence=0.5,
    )
    # exact mode: 0.0
    assert rpr(s_old, s_new, match="exact") == 0.0
    print("✓ test_rpr_total_loss")


def test_rpr_jaccard_catches_paraphrase():
    """Jaccard mode is robust to minor rewording."""
    s_old = StateVector(
        goal="test",
        steps_completed=[ReasoningStep(step_id=1, description="Analyze the market for fitness apps in Egypt", status="completed", dependencies=[])],
        steps_pending=[],
        assumptions=[],
        decisions=[],
        confidence=0.5,
    )
    s_new = StateVector(
        goal="test",
        steps_completed=[ReasoningStep(step_id=1, description="Analyze the market for fitness apps in Cairo", status="completed", dependencies=[])],
        steps_pending=[],
        assumptions=[],
        decisions=[],
        confidence=0.5,
    )
    # exact: 0 (the only string differs)
    assert rpr(s_old, s_new, match="exact") == 0.0
    # jaccard: should be > 0 (heavy token overlap)
    score = rpr(s_old, s_new, match="jaccard", threshold=0.3)
    assert score > 0.5, f"expected high jaccard preservation, got {score}"
    print(f"✓ test_rpr_jaccard_catches_paraphrase  (RPR={score})")


def test_state_vector_bumped():
    """bumped() returns a copy with version+1."""
    s = make_state()
    s2 = s.bumped()
    assert s2.version == s.version + 1
    assert s2.goal == s.goal
    print("✓ test_state_vector_bumped")


def test_state_vector_summary():
    """summary() is a one-liner with key info."""
    s = make_state()
    line = s.summary()
    assert "goal=" in line
    assert "conf=" in line
    print(f"✓ test_state_vector_summary  ('{line[:60]}...')")


def test_extractor_refresh_pending_updates_descriptions():
    """
    v0.5.0: refresh_pending() should rewrite pending step descriptions
    to include the new keywords from the latest interrupt. Fails soft
    if the LLM doesn't return valid JSON (returns s unchanged).
    """
    s = StateVector(
        goal="Marketing plan for fitness app in Egypt",
        steps_completed=[
            ReasoningStep(step_id=1, description="Analyze market", status="completed", dependencies=[]),
        ],
        steps_pending=[
            ReasoningStep(step_id=2, description="Choose channels.", latest_cot=None, dependencies=[]),
            ReasoningStep(step_id=3, description="Allocate budget.", latest_cot=None, dependencies=[]),
        ],
        assumptions=["Egypt is the target market"],
        decisions=[],
        confidence=0.5,
    )

    # Use a mock LLM that returns a valid _RefreshResp
    from lars.llm import LLMClient
    from lars.extractor import _RefreshResp, _RefreshUpdate

    class _StubLLM(LLMClient):
        def complete_json(self, system, user, schema):
            return _RefreshResp(updates=[
                _RefreshUpdate(step_id=2, new_description="Choose marketing channels in Cairo."),
                _RefreshUpdate(step_id=3, new_description="Allocate budget for Cairo campaign."),
            ])

    e = StateExtractor(_StubLLM())
    s_new = e.refresh_pending(s, "scope_narrow", "Cairo")

    assert s_new.version == s.version + 1, "Version should increment after refresh"
    assert len(s_new.steps_pending) == len(s.steps_pending), "No steps should be added/dropped"
    step2 = next(p for p in s_new.steps_pending if p.step_id == 2)
    assert "Cairo" in step2.description, f"Step 2 should mention Cairo: {step2.description}"
    step3 = next(p for p in s_new.steps_pending if p.step_id == 3)
    assert "Cairo" in step3.description, f"Step 3 should mention Cairo: {step3.description}"
    print("✓ test_extractor_refresh_pending_updates_descriptions")


if __name__ == "__main__":
    test_extractor_with_mock()
    test_rpr_identical_states()
    test_rpr_total_loss()
    test_rpr_jaccard_catches_paraphrase()
    test_state_vector_bumped()
    test_state_vector_summary()
    test_extractor_refresh_pending_updates_descriptions()
    print("\nAll tests passed.")
