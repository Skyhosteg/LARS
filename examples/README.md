# Examples

This directory contains runnable demonstrations of LARS, ordered from
simplest to most complete.

## Quick start

```bash
# From the repo root:
python examples/demo.py            # v1: extract → RPR
python examples/demo_merge.py      # v2: full pipeline
python examples/demo_live.py       # v3: interactive REPL
python examples/run_benchmark.py   # v4: full benchmark
```

## What each demo shows

### `demo.py` — Extract + RPR
- Extracts `S(t)` from a CoT
- Compares before/after a user interrupt
- Computes RPR (jaccard + exact modes)

### `demo_merge.py` — Full pipeline
- Runs three different interrupts through the same S(t)
- Shows the `MergeTrace` for each
- Computes RPR for each

### `demo_live.py` — Interactive REPL
- You enter a goal
- LARS extracts an initial plan
- You can interrupt at each step
- RPR + MergeTrace fire in real time
- Type `quit` to exit

### `run_benchmark.py` — The 12-task benchmark
- Runs all 4 methods on all 12 tasks
- Prints the trade-off table
- Saves raw results to `benchmark_results.json`

## Customization

All demos use `MockLLM` by default (no API key needed). To use a real
LLM, set `OPENAI_API_KEY` in your environment.

To plug in a custom goal, edit the `GOAL` constant at the top of
each demo file.
