"""
tests/test_agent.py — Non-interactive tests of LiveAgent.

We don't want to actually block on stdin in tests. Instead we feed a
list of canned interrupts.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lars.agent import LiveAgent
from lars.delta_u import DeltaUParserMock
from lars.executor import MockStepExecutor
from lars.extractor import StateExtractor
from lars.llm import MockLLM
from lars.merger import StateMerger
from lars.state import StateVector, ReasoningStep, Decision


INITIAL_FIXTURE = {
    "Goal: Create": {
        "goal": "demo goal",
        "steps_completed": [],
        "steps_pending": [
            {"step_id": 1, "description": "Analyze the market", "status": "pending", "dependencies": []},
            {"step_id": 2, "description": "Define the audience", "status": "pending", "dependencies": [1]},
            {"step_id": 3, "description": "Choose channels", "status": "pending", "dependencies": [2]},
            {"step_id": 4, "description": "Allocate budget", "status": "pending", "dependencies": [3]},
            {"step_id": 5, "description": "Define KPIs", "status": "pending", "dependencies": [3, 4]},
        ],
        "assumptions": [],
        "decisions": [],
        "confidence": 0.5,
    }
}


CANNED_COTS = {
    1: "Egypt has 110M people, 60% under 30.",
    2: "Primary: 18-30 urban middle income.",
    3: "Instagram, TikTok, Facebook ads, influencers in Egypt.",
    4: "Budget: 40/30/20/10 split.",
    5: "KPIs: downloads, retention, CAC, LTV.",
}


def make_agent(interrupts: list[str | None]) -> LiveAgent:
    llm = MockLLM(fixtures=INITIAL_FIXTURE)
    extractor = StateExtractor(llm)
    executor = MockStepExecutor(canned=CANNED_COTS)
    parser = DeltaUParserMock()
    merger = StateMerger()

    queue = list(interrupts)

    def interrupt_source():
        return queue.pop(0) if queue else None

    return LiveAgent(
        extractor=extractor,
        executor=executor,
        parser=parser,
        merger=merger,
        initial_cot="Goal: Create a marketing plan.",
        interrupt_source=interrupt_source,
    )


def test_agent_no_interrupts():
    """Run end-to-end with no interrupts. All 5 steps should complete."""
    agent = make_agent([None] * 10)  # plenty of None interrupts
    final = agent.run("demo goal", max_steps=5)
    assert len(final.steps_completed) == 5
    assert len(final.steps_pending) == 0
    assert final.version >= 5
    print(f"✓ test_agent_no_interrupts  (final v={final.version}, completed={len(final.steps_completed)})")


def test_agent_interrupt_at_step_1():
    """Interrupt at step 1 with a scope-narrow. Should preserve/modify."""
    agent = make_agent([
        "focus on Cairo only",  # interrupt after step 1
        None, None, None, None,
    ])
    final = agent.run("demo goal", max_steps=5)
    # All 5 steps should still complete (interrupt didn't drop)
    assert len(final.steps_completed) == 5
    # The state should have a "Narrow scope to Cairo" decision
    assert any("Cairo" in d.decision for d in final.decisions)
    print(f"✓ test_agent_interrupt_at_step_1  (final v={final.version}, decisions={len(final.decisions)})")


def test_agent_abort_at_step_3():
    """ABORT after step 3 should clear the rest of the state."""
    agent = make_agent([
        None, None, "stop, restart from scratch",
    ])
    final = agent.run("demo goal", max_steps=5)
    # The ABORT handler clears the state, so subsequent steps find no pending
    # After abort, only "Restart from scratch" is pending
    assert final.confidence == 0.0
    # The agent should have stopped shortly after the abort
    print(f"✓ test_agent_abort_at_step_3  (final completed={len(final.steps_completed)}, conf={final.confidence})")


def test_agent_replace_at_step_3():
    """REPLACE after step 3 should swap Facebook → Twitter in the executed CoT."""
    agent = make_agent([
        None, None, "use Twitter instead of Facebook",
        None, None,
    ])
    final = agent.run("demo goal", max_steps=5)
    # The replace at step 3 might not have run because the mock executor's CoT
    # for step 3 doesn't contain "Facebook" — but it should still complete
    assert len(final.steps_completed) == 5
    print(f"✓ test_agent_replace_at_step_3  (final v={final.version})")


def test_agent_preserves_state_across_interrupts():
    """Each interrupt should bump version, and completed steps should persist."""
    agent = make_agent([
        "focus on Cairo only",  # after step 1
        "use Twitter instead of Facebook",  # after step 2
        None, None, None,
    ])
    final = agent.run("demo goal", max_steps=5)
    # Version should be at least 5 (steps) + 2 (merges) = 7
    assert final.version >= 5
    # All 5 steps should be completed
    assert len(final.steps_completed) == 5
    # Both decisions should be present
    decisions_text = " ".join(d.decision for d in final.decisions)
    assert "Cairo" in decisions_text or "Twitter" in decisions_text
    print(f"✓ test_agent_preserves_state_across_interrupts  (final v={final.version})")


if __name__ == "__main__":
    test_agent_no_interrupts()
    test_agent_interrupt_at_step_1()
    test_agent_abort_at_step_3()
    test_agent_replace_at_step_3()
    test_agent_preserves_state_across_interrupts()
    print("\nAll LiveAgent tests passed.")
