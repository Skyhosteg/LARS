# LARS Architecture

This document goes deeper than the README. For the formal treatment,
see the [preprint on Zenodo](https://zenodo.org/records/20618761) (DOI:
10.5281/zenodo.20618761).

## Conceptual model

LARS reframes LLM interaction from a stateless request-response loop
to a **continuous state-transition process**:

```
S(t + 1) = f(S(t), ∆U(t))
```

- **S(t)** is the structured reasoning state at time t.
- **∆U(t)** is the user's interrupt, classified into a typed intent.
- **f** is the merge function that combines preservation, modification,
  and adaptation with weights α, β, γ (α + β + γ = 1, α ≥ 0.5).

## Module layout

```
lars/
├── state.py              # StateVector schema (Pydantic)
├── llm.py                # LLM client (OpenAI + Mock)
├── embeddings.py         # Pluggable embedder (Hash + OpenAI)
├── prompts.py            # System prompts for the LLM
│
├── extractor.py          # CoT → StateVector
├── update_intent.py      # 9 intent types
├── delta_u.py            # Raw text → UpdateIntent (LLM + Mock)
│
├── merger.py             # f(S, ∆U) + MergeTrace
├── metrics.py            # rpr, rpr_semantic, latency, cost
│
├── executor.py           # Step-by-step CoT generator
├── agent.py              # LiveAgent runtime
│
├── baselines.py          # 3 baselines + LARS method
├── tasks.py              # 12 benchmark tasks
├── benchmark.py          # Benchmark harness
│
└── langgraph_integration.py  # LangGraph wiring
```

## The 9 intent types

| Intent | Trigger phrase | Merge handler |
|---|---|---|
| `SCOPE_NARROW` | "focus on X only" | Rewrites broad-scope references |
| `SCOPE_EXPAND` | "also include Y" | Inserts new pending step |
| `CORRECTION` | "actually use X" | Modifies most recent step |
| `REPLACE` | "use X instead of Y" | Swaps token everywhere |
| `ADD` | "also include X" | Appends new pending step |
| `REMOVE` | "drop X" | Drops matching step |
| `REPRIORITIZE` | "do X first" | Records re-rank request (v1) |
| `CLARIFY` | "what do you mean by X?" | No-op + log |
| `ABORT` | "stop, restart" | Clears state |

## α/β/γ weights

The merge function combines three operations:

```
S(t+1) = α · Preserve(S(t))
       + β · Update(S(t), ∆U)
       + γ · Adapt(S(t), ∆U)
```

- **α (preservation)**: minimum 0.5. Default 0.6.
- **β (direct update)**: the user's intended change. Default 0.3.
- **γ (strategy adaptation)**: re-planning, re-prioritization. Default 0.1.

Every `MergeTrace` records the actual weights applied, so the paper
can show exactly how much of each merge was preservation vs. update.

## LiveAgent runtime

```
Goal + Initial CoT
        │
        ▼
   ┌────────┐
   │  Plan  │  (extractor)
   └────┬───┘
        │
        ▼
   ┌────────┐
   │  Loop  │  for each pending step:
   └────┬───┘
        │
        ├─► Execute step ──► Mark completed
        │                        │
        ▼                        ▼
   ┌────────┐              ┌──────────┐
   │Listen  │──interrupt──►│  Parse ∆U │
   └────────┘              └─────┬────┘
        │                        │
        │ no input               ▼
        │                  ┌──────────┐
        │                  │  Merge f │
        │                  └─────┬────┘
        │                        │
        ◄────────────────────────┘
        │
        ▼
   ┌────────┐
   │  RPR   │
   └────────┘
```

## LangGraph integration

The same runtime can be expressed as a LangGraph state machine:

```python
graph = build_lars_graph(extractor, executor, parser, merger)
graph = graph.compile(interrupt_before=["execute"])
```

LangGraph's `interrupt_before` provides checkpoint-based interruption
at every step — this is G1 from the gap survey.

## Metrics

### RPR (Reasoning Preservation Rate)

```python
rpr_semantic(s_old, s_new, embedder, threshold=0.7)
    = (# old items with semantic match ≥ threshold) / (# old items)
```

Available match modes: `exact`, `jaccard`, `semantic`.

### Adaptation Latency (M2)

Wall-clock time from ΔU arrival to S(t+1) stable. Target: < 500ms.

### Recompute Cost Ratio (M3)

```
ratio = tokens_used_by_lars / tokens_used_by_scratch_recompute
```

Target: ≤ 0.30 (i.e., 70% reduction).

## Benchmark

12 tasks × 4 methods. See `examples/run_benchmark.py`.

## Limitations (for paper discussion)

1. Rule-based merger is literal — f_llm ablation is v2 work
2. Mock ∆U parser is English-only
3. REPRIORITIZE is a no-op in v1
4. No real OpenAI token counts in the cost proxy
5. No user study yet
