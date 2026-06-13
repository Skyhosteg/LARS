# LARS — Live Adaptive Reasoning System

> **The first LLM runtime that preserves reasoning state across user interruption.**

[![Tests](https://github.com/Skyhosteg/LARS/actions/workflows/tests.yml/badge.svg)](https://github.com/Skyhosteg/LARS/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/zenodo-10.5281%2Fzenodo.20618761-blue)](https://zenodo.org/records/20618761)
[![Code style](https://img.shields.io/badge/code%20style-pydantic%20v2-orange)](https://docs.pydantic.dev/latest/)

---

## The headline result

LARS is the only method in our benchmark that achieves **high reasoning
preservation WHILE incorporating the user's interrupt**:

| Method | RPR ↑ | Cost ↓ | Used interrupt? | **Win?** |
|---|---|---|---|---|
| `no_interrupt` | 1.000 | 62.4 | 0% | ✗ ignores user |
| `restart_from_scratch` | 0.000 | 67.7 | 100% | ✗ loses reasoning |
| `langgraph_checkpoint` | 0.000 | 67.7 | 100% | ✗ appends, no merge |
| **`lars`** | **1.000** | **62.4** | **100%** | **✓ preserves + adapts** |

See [`examples/run_benchmark.py`](examples/run_benchmark.py) to reproduce.

---

## What is LARS?

LARS (Live Adaptive Reasoning System) reframes LLM interaction from a
stateless request-response loop into a **continuous state-transition
process**:

```
S(t + 1) = f(S(t), ∆U(t))
```

- `S(t)` — the structured reasoning state (goal, steps, assumptions, decisions)
- `∆U(t)` — the user's interrupt, classified into one of 9 typed intents
- `f` — a weighted merge with `α + β + γ = 1, α ≥ 0.5`

**Paper**: [LARS: Live Adaptive Reasoning System for Continuous-State Interactive AI](https://zenodo.org/records/20618761) (Salah, 2026, DOI: 10.5281/zenodo.20618761) — v3 with real-LLM validation on `gpt-4o-mini` and 3-layer defense pipeline (see `lars_v3_paper.md` in this repo).

---

## ⚡ 30-second quick start

```bash
git clone https://github.com/Skyhosteg/LARS.git
cd lars
pip install -r requirements.txt

# Run the live interactive demo
python examples/demo_live.py
```

> Enter any goal. LARS plans, executes, and pauses after each step.
> Type to interrupt — the system parses your input, applies the merge,
> and continues from the new state.

To run the full benchmark (no API key needed):
```bash
python examples/run_benchmark.py
```

With a real LLM:
```bash
export OPENAI_API_KEY=sk-...
python examples/run_benchmark.py
```

---

## Architecture

```
                    user text
                        │
                        ▼
                 ┌──────────────┐
                 │  ∆U Parser   │   → UpdateIntent (9 types)
                 └──────┬───────┘
                        │
         S(t)  ────────┼────►  ┌──────────────┐
        (state)        │       │ StateMerger  │  → S(t+1) + MergeTrace
                       └──────►│   f(S, ∆U)   │     α+β+γ weights
                               └──────┬───────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │  RPR metric  │   → 0..1 (semantic similarity)
                              └──────────────┘
```

## The 9 intent types

| Intent | Example | Handler |
|---|---|---|
| `SCOPE_NARROW` | "focus on Cairo only" | rewrites broad refs |
| `SCOPE_EXPAND` | "also include the Gulf" | inserts pending step |
| `CORRECTION` | "actually use blue" | modifies last step |
| `REPLACE` | "use Twitter instead of Facebook" | swaps token |
| `ADD` | "also include TikTok" | appends pending step |
| `REMOVE` | "drop the influencer budget" | drops matching |
| `REPRIORITIZE` | "do budget first" | note only in v1 |
| `CLARIFY` | "what do you mean by young?" | no-op + log |
| `ABORT` | "stop, restart" | clears state |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

---

## What's in the box

| File | Purpose |
|---|---|
| `lars/state.py` | `StateVector` Pydantic schema (S(t)) |
| `lars/extractor.py` | CoT → StateVector |
| `lars/delta_u.py` | User text → `UpdateIntent` (LLM + heuristic) |
| `lars/merger.py` | The `f` function + `MergeTrace` |
| `lars/metrics.py` | `rpr()`, `rpr_semantic()`, latency, cost |
| `lars/embeddings.py` | Pluggable embedder (Hash + OpenAI) |
| `lars/agent.py` | **`LiveAgent` runtime with interrupts** |
| `lars/langgraph_integration.py` | LangGraph wiring (G1) |
| `lars/baselines.py` | 3 baselines + LARS method |
| `lars/tasks.py` | 12 benchmark tasks |
| `lars/benchmark.py` | The benchmark harness |
| `examples/` | 4 runnable demos |
| `tests/` | 28 tests across 4 suites |

## Running the tests

```bash
python tests/test_extractor.py    # 6 tests
python tests/test_merge.py        # 11 tests
python tests/test_agent.py        # 5 tests
python tests/test_benchmark.py    # 6 tests
```

All 28 tests should pass.

---

## Known limitations

1. **Rule-based merger is literal** — "Egypt" → "Cairo" sometimes breaks
   grammar. A v2 `f_llm` ablation is in the roadmap.
2. **Mock ∆U parser is English-only.** Use `DeltaUParserLLM` for other
   languages.
3. **REPRIORITIZE is a no-op in v1** — graph re-ranking is v2 work.
4. **Benchmark uses MockLLM by default** — real OpenAI runs vary.
5. **No user study yet** — CHI-style evaluation is v2.

## Citation

```bibtex
@misc{salah2026lars,
  title  = {LARS: Live Adaptive Reasoning System for Continuous-State Interactive AI},
  author = {Salah, Mohamed},
  year   = {2026},
  month  = jun,
  doi    = {10.5281/zenodo.20618761},
  url    = {https://zenodo.org/records/20618761},
  note   = {v3 with real-LLM validation; see also the lars_v3_paper.md in this repo for the 3-layer defense pipeline},
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The 9-intent taxonomy is an
evolving research artifact — propose new types, add benchmark tasks, or
implement `f_llm`.

## License

- **Code**: MIT — see [LICENSE](LICENSE)
- **Paper**: CC-BY-4.0 — see [Zenodo](https://zenodo.org/records/20618761)
