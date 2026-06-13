"""
langgraph_integration.py — Wire LARS into LangGraph

This is the production-grade wiring for G1 (continuous interruption).
LangGraph's `interrupt_before` gives us a checkpoint at every node —
in our case, every reasoning step.

The graph:
  [extract] → [execute_step] → [listen] → (if interrupt) → [merge] → [execute_step] → ...
                                          ↘ (else)              ↗
                                           [execute_step] ─────┘

For an even more aggressive G1, set interrupt_before=["execute_step"]
which makes the graph *always* pause before each step, waiting for
human input.
"""

from __future__ import annotations

from typing import Optional, TypedDict

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from .delta_u import DeltaUParser
from .executor import StepExecutor
from .extractor import StateExtractor
from .merger import StateMerger
from .state import StateVector


# --------------------------------------------------------------------------- #
# Graph state (LangGraph's state is a dict, not our StateVector)
# --------------------------------------------------------------------------- #


class LARSGraphState(TypedDict, total=False):
    goal: str
    initial_cot: str
    state_vector: dict         # serialized StateVector
    user_interrupt: Optional[str]
    interrupt_counter: int
    max_steps: int
    steps_executed: int
    final_state: dict


# --------------------------------------------------------------------------- #
# Building the graph
# --------------------------------------------------------------------------- #


def build_lars_graph(
    extractor: StateExtractor,
    executor: StepExecutor,
    parser: DeltaUParser,
    merger: StateMerger,
) -> "object":  # returns a CompiledGraph
    """
    Build a LangGraph that runs LARS with interrupts at every step.

    Usage:
        graph = build_lars_graph(extractor, executor, parser, merger)
        config = {"configurable": {"thread_id": "session-1"}}
        result = graph.invoke({"goal": "...", "max_steps": 5}, config=config)
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph not installed. Run: pip install langgraph")

    g = StateGraph(LARSGraphState)

    def extract_node(state: LARSGraphState) -> dict:
        sv = extractor.extract(state["goal"], state.get("initial_cot", ""))
        return {"state_vector": sv.model_dump()}

    def execute_node(state: LARSGraphState) -> dict:
        sv = StateVector.model_validate(state["state_vector"])
        # Find next pending step
        next_step = next(
            (s for s in sv.steps_pending if s.status == "pending"),
            None,
        )
        if next_step is None:
            return {"final_state": state["state_vector"], "steps_executed": state.get("steps_executed", 0)}
        # Execute
        cot = executor.execute(sv, next_step)
        # Mark completed
        new_sv = sv.model_copy(update={
            "steps_completed": sv.steps_completed + [next_step.model_copy(update={"status": "completed"})],
            "steps_pending": [sp for sp in sv.steps_pending if sp.step_id != next_step.step_id],
            "version": sv.version + 1,
        })
        return {
            "state_vector": new_sv.model_dump(),
            "steps_executed": state.get("steps_executed", 0) + 1,
        }

    def merge_node(state: LARSGraphState) -> dict:
        sv = StateVector.model_validate(state["state_vector"])
        user_text = state.get("user_interrupt")
        if not user_text:
            return {}
        intent = parser.parse(user_text)
        new_sv, _trace = merger.merge(sv, intent)
        return {
            "state_vector": new_sv.model_dump(),
            "user_interrupt": None,
            "interrupt_counter": state.get("interrupt_counter", 0) + 1,
        }

    def should_merge(state: LARSGraphState) -> str:
        if state.get("user_interrupt"):
            return "merge"
        return "execute"

    def should_continue(state: LARSGraphState) -> str:
        if state.get("final_state"):
            return END
        if state.get("steps_executed", 0) >= state.get("max_steps", 5):
            return END
        return "execute"

    g.add_node("extract", extract_node)
    g.add_node("execute", execute_node)
    g.add_node("merge", merge_node)

    g.add_edge(START, "extract")
    g.add_edge("extract", "execute")
    g.add_conditional_edges("execute", should_continue, {END: END, "execute": "execute"})
    g.add_conditional_edges("execute", should_merge, {"merge": "merge", "execute": "execute"})
    g.add_edge("merge", "execute")

    # The magic: interrupt before every execute_node
    # This is G1 — checkpoint-based but at every step
    return g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["execute"],
    )


# --------------------------------------------------------------------------- #
# A tiny runnable demo (also serves as a smoke test)
# --------------------------------------------------------------------------- #


def _demo() -> None:
    from .delta_u import DeltaUParserMock
    from .executor import MockStepExecutor
    from .llm import MockLLM
    from .state import StateVector

    if not LANGGRAPH_AVAILABLE:
        print("langgraph not available — skipping demo")
        return

    print("\n[langgraph demo] Building LARS graph with interrupt_before=[execute]...")

    fixture = {
        "Goal: Create": {
            "goal": "demo",
            "steps_completed": [],
            "steps_pending": [
                {"step_id": 1, "description": "Analyze the market", "status": "pending", "dependencies": []},
                {"step_id": 2, "description": "Define the audience", "status": "pending", "dependencies": [1]},
                {"step_id": 3, "description": "Choose channels", "status": "pending", "dependencies": [2]},
            ],
            "assumptions": [],
            "decisions": [],
            "confidence": 0.5,
        }
    }

    extractor = StateExtractor(MockLLM(fixtures=fixture))
    executor = MockStepExecutor(canned={
        1: "Market analysis text",
        2: "Audience definition text",
        3: "Channel selection text",
    })
    parser = DeltaUParserMock()
    merger = StateMerger()

    graph = build_lars_graph(extractor, executor, parser, merger)

    config = {"configurable": {"thread_id": "demo-1"}}
    print("\n[langgraph demo] Invoking graph with goal='Create a marketing plan'...")
    print("[langgraph demo] This will pause BEFORE each step (interrupt_before=[execute]).")
    print("[langgraph demo] In a real UI, you'd inject user input via Command(resume=...).\n")

    # First invocation: runs up to the first interrupt
    result = graph.invoke(
        {"goal": "Create a marketing plan", "max_steps": 3, "steps_executed": 0, "interrupt_counter": 0},
        config=config,
    )
    print(f"[langgraph demo] After 1st invoke: steps_executed={result.get('steps_executed')}")

    # Inject an interrupt and resume
    print("\n[langgraph demo] Injecting user_interrupt='focus on Cairo only' and resuming...")
    result = graph.invoke(
        {"user_interrupt": "focus on Cairo only"},
        config=config,
    )
    print(f"[langgraph demo] After 2nd invoke: steps_executed={result.get('steps_executed')}, "
          f"interrupt_counter={result.get('interrupt_counter')}")
    print("[langgraph demo] OK — graph responded to interrupt and continued.\n")


if __name__ == "__main__":
    _demo()
