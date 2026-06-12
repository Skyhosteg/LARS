"""
agent.py — The LiveAgent

This is the *runtime* of LARS. It walks the reasoning plan step by step,
pausing after each step to listen for user interrupts. If an interrupt
arrives, it parses ΔU, applies f, prints the trace, and continues from
the new state.

The agent is what you'd embed in a product. Everything else (extractor,
ΔU parser, merger, metrics) is a library that the agent composes.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

from .delta_u import DeltaUParser
from .executor import StepExecutor
from .extractor import StateExtractor
from .merger import StateMerger
from .metrics import rpr
from .state import ReasoningStep, StateVector


def _cprint(role: str, msg: str, color: str = "") -> None:
    """Tiny coloured print. Falls back gracefully on dumb terminals."""
    colors = {"red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35, "cyan": 36}
    code = colors.get(color, 0)
    if code and sys.stdout.isatty():
        print(f"\033[{code}m[{role}]\033[0m {msg}")
    else:
        print(f"[{role}] {msg}")


class LiveAgent:
    """
    Walks S(t)'s pending steps one at a time, listening for interrupts.

    Lifecycle:
      1. extract S(t) from the goal (using the initial CoT)
      2. for each pending step:
           a. execute it (CoT)
           b. mark it completed in S(t)
           c. show S(t)
           d. listen for interrupt; if any, parse ΔU and merge
           e. print RPR
      3. end session, show final S(t)

    The `interrupt_source` is a Callable[[], Optional[str]]. By default
    it's a blocking input() prompt. Swap it for a WebSocket/SSE handler
    to get G1 (continuous interruption).
    """

    def __init__(
        self,
        extractor: StateExtractor,
        executor: StepExecutor,
        parser: DeltaUParser,
        merger: StateMerger,
        *,
        initial_cot: str = "",
        interrupt_source: Callable[[], Optional[str]] | None = None,
    ):
        self.extractor = extractor
        self.executor = executor
        self.parser = parser
        self.merger = merger
        self.initial_cot = initial_cot
        self.interrupt_source = interrupt_source or self._stdin_interrupt

    @staticmethod
    def _stdin_interrupt() -> Optional[str]:
        try:
            line = input().strip()
        except EOFError:
            return None
        return line or None

    def run(self, goal: str, max_steps: int = 5) -> StateVector:
        """Run the live session. Returns the final S(t)."""
        # 1. Initial state
        _cprint("LARS", f"Goal: {goal}", "cyan")
        _cprint("LARS", "Extracting initial state S(t)...", "cyan")
        s = self.extractor.extract(goal, self.initial_cot)
        _cprint("LARS", f"Plan: {len(s.steps_completed) + len(s.steps_pending)} steps total", "cyan")
        s_prev: Optional[StateVector] = None

        # 2. Walk the plan
        for step_num in range(max_steps):
            # Find next pending step
            next_step: Optional[ReasoningStep] = None
            for s_step in s.steps_pending:
                if s_step.status == "pending":
                    next_step = s_step
                    break
            if next_step is None:
                _cprint("LARS", "No more pending steps. Done.", "green")
                break

            _cprint("LARS", f"Step {next_step.step_id}: {next_step.description}", "blue")

            # a. Execute
            cot = self.executor.execute(s, next_step)
            _cprint("MODEL", cot, "yellow")

            # b. Mark completed: move from pending to completed
            s = s.model_copy(update={
                "steps_completed": s.steps_completed + [next_step.model_copy(update={"status": "completed"})],
                "steps_pending": [sp for sp in s.steps_pending if sp.step_id != next_step.step_id],
                "version": s.version + 1,
            })
            _cprint("STATE", s.summary(), "green")

            # c. RPR vs previous state
            if s_prev is not None:
                score = rpr(s_prev, s, match="jaccard", threshold=0.3)
                _cprint("METRIC", f"RPR[prev → now] = {score:.3f}", "magenta")

            # d. Listen for interrupt
            _cprint("LARS", "(press Enter to continue, or type to interrupt)", "cyan")
            user_text = self.interrupt_source()
            if user_text:
                _cprint("USER", user_text, "red")
                intent = self.parser.parse(user_text)
                _cprint("ΔU", intent.short(), "magenta")
                s_prev = s
                s, trace = self.merger.merge(s, intent)
                _cprint("F", trace.summary(), "magenta")
                _cprint("STATE", s.summary(), "green")
            else:
                s_prev = s

        _cprint("LARS", "Session ended.", "green")
        return s
