# Contributing to LARS

Thanks for your interest in LARS! This is an open research project and we
welcome contributions of all kinds: code, benchmarks, tasks, bug reports,
documentation, and ideas.

## How to contribute

### 1. Reporting bugs
Open an [issue](../../issues) with:
- A clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- Your Python version and OS

### 2. Suggesting new intent types
The 9 intent types in `lars/update_intent.py` are an evolving taxonomy.
If you find a user interrupt that doesn't fit, open an issue tagged
`new-intent` with examples.

### 3. Adding benchmark tasks
Add to `lars/tasks.py`. Each task needs:
- A clear `goal`
- A `initial_cot` with 4-7 reasoning steps
- A realistic `interrupt` at step 3
- Expected `intent` and `expected_rpr`

### 4. Improving the merger
The rule-based merger in `lars/merger.py` is intentionally simple. A
great contribution would be:
- The f_llm ablation (LLM-based merge)
- Better conflict detection
- Smarter REPRIORITIZE (graph re-ranking)

### 5. Adding a real LLM backend
We currently support OpenAI and a Mock. Anthropic, Together, vLLM, and
local models are all welcome. See `lars/llm.py` for the interface.

## Development setup

```bash
git clone https://github.com/Skyhosteg/LARS.git
cd lars
pip install -r requirements.txt
pip install -e .  # editable install

# Run the test suite
python tests/test_extractor.py
python tests/test_merge.py
python tests/test_agent.py
python tests/test_benchmark.py
```

All tests should pass and the benchmark should reproduce the headline result.

## Code style

- Type hints everywhere
- Pydantic for data models
- One class per concept
- Docstrings on public functions
- Tests alongside the code they test

## Pull request process

1. Open an issue first for non-trivial changes
2. Fork the repo and create a feature branch
3. Add tests for new functionality
4. Ensure all 28 tests pass
5. Update README if you change the public API
6. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.
