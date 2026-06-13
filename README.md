# LARS — Live Adaptive Reasoning System

> The first LLM runtime that preserves reasoning state across user interruption.

[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-v3%20(Zenodo)-orange)](https://zenodo.org/records/20618761)
[![Code style](https://img.shields.io/badge/code%20style-black-black)](pyproject.toml)

## The headline result

LARS is the only method in our benchmark that achieves high reasoning preservation WHILE incorporating the user's interrupt:

| Method | RPR ↑ | Cost ↓ | Used interrupt? | Win? |
|---|---|---|---|---|
| `no_interrupt` | 1.000 | 62.4 | 0% | ✗ ignores user |
| `restart_from_scratch` | 0.000 | 67.7 | 100% | ✗ loses reasoning |
| `langgraph_checkpoint` | 0.000 | 67.7 | 100% | ✗ appends, no merge |
| **`lars`** | **1.000** | **62.4** | **100%** | **✓ preserves + adapts** |

See `examples/run_benchmark.py` to reproduce.

## What is LARS?

LARS (Live Adaptive Reasoning System) reframes LLM interaction from a
stateless request-response loop into a continuous state-transition process:

```
S(t + 1) = f(S(t), ∆U(t))
```

- **S(t)** — the structured reasoning state (goal, steps, assumptions, decisions)
- **∆U(t)** — the user's interrupt, classified into one of 9 typed intents
- **f** — a weighted merge with α + β + γ = 1, α ≥ 0.5

**Paper**: [LARS: Live Adaptive Reasoning System for Continuous-State Interactive AI](https://zenodo.org/records/20618761) (Salah, 2026, DOI: [10.5281/zenodo.20618761](https://doi.org/10.5281/zenodo.20618761)) — v3 with real-LLM validation on `gpt-4o-mini` and a 3-layer defense pipeline (see [`lars_v3_paper.md`](lars_v3_paper.md) in this repo).

## 🆕 What's new in v0.5.1 (3-layer defense pipeline)

The merger is now wrapped in a 3-layer defense pipeline that keeps the state
consistent with a *real* LLM (e.g., `gpt-4o-mini`), even when the LLM
produces generic step descriptions:

```
USER: "use Twitter instead of Facebook"
   │
   ├─ Layer 1: CoT-aware merger  ──►  rewrites the latest CoT of each step
   ├─ Layer 2: Pending-step refresh  ─►  re-states future steps with new keywords
   └─ Layer 3: Active override inject  ►  feeds user text into the next LLM prompt
```

Validated end-to-end on `openai/gpt-4o-mini` via OpenRouter. Both
``focus on Cairo only`` and ``use Twitter instead of Facebook`` produce
`mod=1` in the MergeTrace; in v0.4.9 they produced `mod=0`.

## ⚡ 30-second quick start

```bash
git clone https://github.com/Skyhosteg/LARS.git
cd LARS
pip install -r requirements.txt

# Optional: for LangGraph deployment
# pip install langgraph

# Run the live interactive demo (mock LLM)
python examples/demo_live.py
```

Enter any goal. LARS plans, executes, and pauses after each step. Type to
interrupt — the system parses your input, applies the merge, and continues
from the new state.

### With a real LLM (OpenRouter / OpenAI)

```powershell
# PowerShell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
$env:OPENROUTER_MODEL = "openai/gpt-4o-mini"
python examples/demo_live.py
```

```bash
# bash / zsh
export OPENAI_API_KEY=sk-...
python examples/demo_live.py
```

### With LangGraph (optional)

```bash
pip install langgraph
python -m lars.langgraph_integration
```

The LangGraph wiring is a 50-line graph (`lars/langgraph_integration.py`)
that exposes `interrupt_before=["execute_step"]` for production deployment
with persistence and time-travel.

```powershell
# PowerShell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
$env:OPENROUTER_MODEL = "openai/gpt-4o-mini"
python examples/demo_live.py
```

```bash
# bash / zsh
export OPENAI_API_KEY=sk-...
python examples/demo_live.py
```

To run the full benchmark (mock by default, real LLM with a key):

```bash
python examples/run_benchmark.py
```

## Architecture (v0.5.1)

```
                       user text
                           │
                           ▼
                    ┌──────────────┐
                    │  ∆U Parser   │   → UpdateIntent (9 types)
                    └──────┬───────┘
                           │
        S(t)  ─────────────┼────────►  ┌──────────────────┐
       (state)             │          │ StateMerger      │  → S(t+1) + MergeTrace
              ┌────────────┘          │  f_merge(S, ∆U)  │     α+β+γ weights
              │                       │  (CoT-aware)     │
              │                       └──────┬───────────┘
              │                              │
              ▼                              ▼
   ┌──────────────────┐            ┌──────────────────┐
   │ refresh_pending  │            │  active_overrides│  → injected into next LLM prompt
   │ (re-state future)│            │  Ω(t)            │
   └──────────────────┘            └──────────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │  RPR metric  │   → 0..1 (semantic similarity)
                   └──────────────┘
```

## The 9 intent types

| Intent | Example | Handler |
|---|---|---|
| `SCOPE_NARROW` | "focus on Cairo only" | rewrites broad refs (description + CoT) |
| `SCOPE_EXPAND` | "also include the Gulf" | inserts pending step |
| `CORRECTION` | "actually use blue" | modifies last step |
| `REPLACE` | "use Twitter instead of Facebook" | swaps token (description + CoT) |
| `ADD` | "also include TikTok" | appends pending step |
| `REMOVE` | "drop the influencer budget" | drops matching (description + CoT) |
| `REPRIORITIZE` | "do budget first" | note only in v1; design ready in v3 |
| `CLARIFY` | "what do you mean by young?" | no-op + log |
| `ABORT` | "stop, restart" | clears state |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## What's in the box

| File | Purpose |
|---|---|
| `lars/state.py` | `StateVector` Pydantic schema (S(t)) — incl. `latest_cot` and `active_overrides` |
| `lars/extractor.py` | CoT → `StateVector`; includes `refresh_pending()` |
| `lars/delta_u.py` | User text → `UpdateIntent` (LLM + heuristic) |
| `lars/merger.py` | The `f` function + `MergeTrace` (CoT-aware handlers) |
| `lars/metrics.py` | `rpr()`, `rpr_semantic()`, latency, cost |
| `lars/embeddings.py` | Pluggable embedder (Hash + OpenAI) |
| `lars/agent.py` | `LiveAgent` runtime with interrupts and 3-layer pipeline |
| `lars/llm.py` | `OpenAILLM`, `OpenRouterLLM`, `MockLLM` |
| `lars/langgraph_integration.py` | LangGraph wiring (G1) |
| `lars/baselines.py` | 3 baselines + LARS method |
| `lars/tasks.py` | 12 benchmark tasks |
| `lars/benchmark.py` | The benchmark harness |
| `examples/` | 4 runnable demos |
| `tests/` | **33 tests across 4 suites** |
| `lars_v3_paper.md` | v3 paper (real-LLM validated) |

## Running the tests

```bash
python tests/test_extractor.py    # 7 tests
python tests/test_merge.py        # 13 tests
python tests/test_agent.py        # 7 tests
python tests/test_benchmark.py    # 6 tests
```

All **33** tests should pass.

## Known limitations (v0.5.1)

1. **Rule-based merger is still literal** — but the 3-layer pipeline (CoT-aware
   merge, pending refresh, override injection) makes it robust to real LLM
   output. A learned $f_\theta$ remains the highest-priority extension.
2. **Mock ∆U parser is English-only.** Use `DeltaUParserLLM` for other
   languages.
3. **REPRIORITIZE is a no-op** — graph re-ranking is design-ready, pending impl.
4. **Single-LLM validation** — v0.5.1 validates on `gpt-4o-mini`. Cross-model
   replication (Claude, Llama, Gemini) is future work.
5. **No user study yet** — CHI-style evaluation is future work.

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

- Code: MIT — see [LICENSE](LICENSE)
- Paper: CC-BY-4.0 — see [Zenodo](https://zenodo.org/records/20618761)
