# Changelog

All notable changes to LARS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-06-12

### Added
- Full benchmark harness (`lars/benchmark.py`) with 12 tasks × 4 methods
- 3 baselines (no_interrupt, restart_from_scratch, langgraph_checkpoint)
- LARS method as the proposed approach
- Hash-based and OpenAI embedders (`lars/embeddings.py`)
- Semantic RPR (`rpr_semantic()`) with pluggable embedder
- LiveAgent runtime (`lars/agent.py`) with continuous interruption
- LangGraph integration (`lars/langgraph_integration.py`)
- 12 benchmark tasks (6 planning + 6 reasoning) in `lars/tasks.py`
- 6 new benchmark tests (28 total)
- Paper outline for v2 (`paper_outline.md`)
- This changelog and full repo metadata for GitHub

### Changed
- `MergeTrace` now records the actual α/β/γ weights applied (paper-grade auditability)
- `rpr()` now supports `match="exact"`, `match="jaccard"`, and `match="semantic"`

## [0.3.0] — 2026-06-10

### Added
- Preprint published to Zenodo: DOI [10.5281/zenodo.20618761](https://zenodo.org/records/20618761)
- 4-page formal paper with framework, RPR, and architecture

## [0.2.0] — 2026-06-09

### Added
- ΔU Parser (`lars/delta_u.py`) with 9 intent types
- StateMerger (`lars/merger.py`) with rule-based handlers for each intent
- MergeTrace dataclass for paper-grade auditability
- α/β/γ weighted merge with constraint validation
- 11 tests for the parser and merger

## [0.1.0] — 2026-06-09

### Added
- StateVector schema (`lars/state.py`) with Pydantic validation
- StateExtractor (`lars/extractor.py`) for CoT → StateVector
- LLM client (`lars/llm.py`) with OpenAI + Mock backends
- Initial RPR metric (jaccard mode)
- 6 smoke tests for the extractor
- README with quick start

[0.4.0]: https://github.com/Skyhosteg/LARS/compare/v0.3.0...v0.4.0
[0.3.0]: https://zenodo.org/records/20618761
[0.2.0]: https://github.com/Skyhosteg/LARS/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Skyhosteg/LARS/releases/tag/v0.1.0
