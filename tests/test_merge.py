"""
tests/test_merge.py — Tests for the ΔU parser and the merger f(S, ΔU)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lars.delta_u import DeltaUParserMock
from lars.merger import StateMerger
from lars.metrics import rpr
from lars.state import StateVector, ReasoningStep, Decision
from lars.update_intent import IntentType


def make_initial_state() -> StateVector:
    return StateVector(
        goal="Create a marketing plan for a fitness app in Egypt",
        steps_completed=[
            ReasoningStep(step_id=1, description="Analyze the market: Egypt has 110M people, 60% under 30", status="completed", dependencies=[]),
            ReasoningStep(step_id=2, description="Define the audience: 18-30 year olds in Egypt, urban, middle income", status="completed", dependencies=[1]),
            ReasoningStep(step_id=3, description="Choose channels: Instagram, TikTok, Facebook ads, influencers in Egypt", status="completed", dependencies=[2]),
        ],
        steps_pending=[
            ReasoningStep(step_id=4, description="Allocate budget across channels", status="pending", dependencies=[3]),
            ReasoningStep(step_id=5, description="Define KPIs", status="pending", dependencies=[3, 4]),
        ],
        assumptions=[
            "Egypt has 110M people, 60% under 30",
            "Target all major cities including Cairo and Alexandria",
        ],
        decisions=[
            Decision(decision="Target 18-30 urban audience", rationale="Largest growing segment"),
            Decision(decision="Use Instagram and TikTok", rationale="High engagement with 18-30"),
        ],
        confidence=0.78,
    )


def test_parser_scope_narrow():
    p = DeltaUParserMock()
    intent = p.parse("focus on Cairo only")
    assert intent.type == IntentType.SCOPE_NARROW
    assert "Cairo" in (intent.new_value or "")
    print(f"✓ test_parser_scope_narrow  →  {intent.short()}")


def test_parser_scope_expand():
    p = DeltaUParserMock()
    intent = p.parse("also include the Gulf region")
    assert intent.type == IntentType.SCOPE_EXPAND
    assert "Gulf" in (intent.new_value or "")
    print(f"✓ test_parser_scope_expand  →  {intent.short()}")


def test_parser_replace():
    p = DeltaUParserMock()
    intent = p.parse("use Twitter instead of Facebook")
    assert intent.type == IntentType.REPLACE
    assert intent.old_value.lower() == "facebook"
    assert intent.new_value.lower() == "twitter"
    print(f"✓ test_parser_replace  →  {intent.short()}")


def test_parser_remove():
    p = DeltaUParserMock()
    intent = p.parse("drop the influencer budget")
    assert intent.type == IntentType.REMOVE
    assert "influencer" in (intent.value or "").lower()
    print(f"✓ test_parser_remove  →  {intent.short()}")


def test_parser_abort():
    p = DeltaUParserMock()
    intent = p.parse("stop, restart from scratch")
    assert intent.type == IntentType.ABORT
    print(f"✓ test_parser_abort  →  {intent.short()}")


def test_merger_narrow_preserves_most():
    """Narrowing scope should keep ≥ 50% of steps preserved."""
    s = make_initial_state()
    p = DeltaUParserMock()
    intent = p.parse("focus on Cairo only")
    m = StateMerger()
    s_new, trace = m.merge(s, intent)
    assert trace.alpha == 0.6
    assert trace.beta == 0.3
    assert trace.gamma == 0.1
    # At least one step preserved, at least one modified
    assert len(trace.preserved_steps) + len(trace.modified_steps) >= 2
    # New decision was added
    assert any("Cairo" in d.decision for d in s_new.decisions)
    # RPR should be high (we modified, didn't drop)
    score = rpr(s, s_new, match="jaccard", threshold=0.2)
    assert score > 0.4, f"expected RPR > 0.4, got {score}"
    print(f"✓ test_merger_narrow_preserves_most  (RPR={score}, trace={trace.summary()})")


def test_merger_expand_inserts_step():
    """Expanding scope should insert a new pending step."""
    s = make_initial_state()
    p = DeltaUParserMock()
    intent = p.parse("also include the Gulf region")
    m = StateMerger()
    s_new, trace = m.merge(s, intent)
    # A new step was inserted
    assert len(trace.inserted_steps) == 1
    # All original steps were preserved
    assert len(trace.preserved_steps) == 3
    # The new pending step mentions "Gulf"
    new_pending = s_new.steps_pending[-1]
    assert "Gulf" in new_pending.description
    print(f"✓ test_merger_expand_inserts_step  (trace={trace.summary()})")


def test_merger_replace_swaps_value():
    """REPLACE should swap 'Facebook' → 'Twitter' everywhere."""
    s = make_initial_state()
    p = DeltaUParserMock()
    intent = p.parse("use Twitter instead of Facebook")
    m = StateMerger()
    s_new, trace = m.merge(s, intent)
    # The step mentioning Facebook should now mention Twitter (and not Facebook)
    modified_step = next(s for s in s_new.steps_completed if s.step_id == 3)
    assert "Twitter" in modified_step.description
    assert "Facebook" not in modified_step.description
    assert len(trace.modified_steps) >= 1
    print(f"✓ test_merger_replace_swaps_value  (modified_steps={trace.modified_steps})")


def test_merger_abort_drops_everything():
    """ABORT should drop all reasoning."""
    s = make_initial_state()
    p = DeltaUParserMock()
    intent = p.parse("stop, restart from scratch")
    m = StateMerger()
    s_new, trace = m.merge(s, intent)
    assert len(trace.dropped_steps) == 3
    assert len(s_new.steps_completed) == 0
    assert s_new.confidence == 0.0
    print(f"✓ test_merger_abort_drops_everything  (dropped={trace.dropped_steps})")


def test_merger_weight_validation():
    """α must be ≥ 0.5, α+β+γ must equal 1.0."""
    try:
        StateMerger(alpha=0.4, beta=0.3, gamma=0.3)
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        StateMerger(alpha=0.6, beta=0.3, gamma=0.2)
        assert False, "should have raised (sum != 1)"
    except ValueError:
        pass
    print("✓ test_merger_weight_validation")


def test_merger_version_increments():
    """Merge should always bump version."""
    s = make_initial_state()
    m = StateMerger()
    s_new, _ = m.merge(s, DeltaUParserMock().parse("focus on Cairo only"))
    assert s_new.version == s.version + 1
    print("✓ test_merger_version_increments")


if __name__ == "__main__":
    test_parser_scope_narrow()
    test_parser_scope_expand()
    test_parser_replace()
    test_parser_remove()
    test_parser_abort()
    test_merger_narrow_preserves_most()
    test_merger_expand_inserts_step()
    test_merger_replace_swaps_value()
    test_merger_abort_drops_everything()
    test_merger_weight_validation()
    test_merger_version_increments()
    print("\nAll ΔU + merger tests passed.")
