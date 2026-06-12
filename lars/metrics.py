"""
metrics.py — Quantifying LARS

This module implements the three metrics from the gap survey:

  M1. RPR(t)  — Reasoning Preservation Rate (G2)
  M2. Adaptation Latency  (G5)
  M3. Recompute Cost Ratio (G6)

These are what the paper's experiments will measure. They are also
useful for debugging the prototype.

RPR has two modes:
  - "exact"   : strict string match (legacy)
  - "jaccard" : token overlap (v1 demo)
  - "semantic": embedder-based (recommended for the paper)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .embeddings import Embedder, HashEmbedder, default_embedder
from .state import StateVector


# --------------------------------------------------------------------------- #
# M1 — Reasoning Preservation Rate
# --------------------------------------------------------------------------- #


def _tokenize(s: str) -> set[str]:
    """Lowercase, strip punctuation, drop short tokens."""
    import re
    return {t for t in re.findall(r"[a-z0-9\u0600-\u06ff]+", s.lower()) if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rpr(
    s_old: StateVector,
    s_new: StateVector,
    *,
    match: str = "jaccard",
    threshold: float = 0.3,
) -> float:
    """
    RPR — Reasoning Preservation Rate (M1 from the survey)

    Measures how much of s_old's reasoning survives in s_new.

    Two match modes:
      - "exact"   (default legacy): set intersection on lowercased strings.
                  Strict. Brittle to rewording.
      - "jaccard" (recommended):    token Jaccard >= threshold.
                  A step is "preserved" if its token overlap with any
                  step in the new state exceeds `threshold`.

    Returns a float in [0.0, 1.0]. Returns 1.0 if s_old had no reasoning
    to preserve (degenerate case).
    """
    if match == "exact":
        def fingerprint(s: StateVector) -> set[str]:
            fp: set[str] = set()
            for step in s.steps_completed:
                fp.add(f"step::{step.description.strip().lower()}")
            for a in s.assumptions:
                fp.add(f"asm::{a.strip().lower()}")
            for d in s.decisions:
                fp.add(f"dec::{d.decision.strip().lower()}")
            return fp

        old_fp = fingerprint(s_old)
        new_fp = fingerprint(s_new)
        if not old_fp:
            return 1.0
        return round(len(old_fp & new_fp) / len(old_fp), 4)

    if match == "jaccard":
        old_items: list[set[str]] = []
        for step in s_old.steps_completed:
            old_items.append(_tokenize(step.description))
        for a in s_old.assumptions:
            old_items.append(_tokenize(a))
        for d in s_old.decisions:
            old_items.append(_tokenize(d.decision) | _tokenize(d.rationale))

        new_items: list[set[str]] = []
        for step in s_new.steps_completed:
            new_items.append(_tokenize(step.description))
        for a in s_new.assumptions:
            new_items.append(_tokenize(a))
        for d in s_new.decisions:
            new_items.append(_tokenize(d.decision) | _tokenize(d.rationale))

        if not old_items:
            return 1.0

        preserved = 0
        for o in old_items:
            best = max((_jaccard(o, n) for n in new_items), default=0.0)
            if best >= threshold:
                preserved += 1

        return round(preserved / len(old_items), 4)

    raise ValueError(f"Unknown match mode: {match!r}. Use 'exact', 'jaccard', or 'semantic'.")


# --------------------------------------------------------------------------- #
# RPR with semantic (embedding-based) similarity
# --------------------------------------------------------------------------- #


def _gather_items(s: StateVector) -> list[str]:
    items: list[str] = []
    for step in s.steps_completed:
        items.append(step.description)
    for a in s.assumptions:
        items.append(a)
    for d in s.decisions:
        items.append(f"{d.decision} — {d.rationale}")
    return items


def rpr_semantic(
    s_old: StateVector,
    s_new: StateVector,
    embedder: Embedder | None = None,
    threshold: float = 0.7,
) -> float:
    """
    RPR via semantic similarity.

    An element of s_old is "preserved" in s_new if its best cosine
    similarity to any element in s_new exceeds `threshold`.

    Args:
        s_old: original state
        s_new: state after update
        embedder: pluggable; defaults to HashEmbedder (no API needed)
        threshold: similarity cutoff for "preserved" (default 0.7)

    Returns:
        float in [0.0, 1.0]
    """
    embedder = embedder or HashEmbedder()
    old_items = _gather_items(s_old)
    new_items = _gather_items(s_new)

    if not old_items:
        return 1.0

    preserved = 0
    for o in old_items:
        best = max((embedder.similarity(o, n) for n in new_items), default=0.0)
        if best >= threshold:
            preserved += 1

    return round(preserved / len(old_items), 4)


# Backwards-compatible alias
rpr_embedding = rpr_semantic


# --------------------------------------------------------------------------- #
# M2 — Adaptation Latency
# --------------------------------------------------------------------------- #


@dataclass
class LatencyResult:
    """Wall-clock latency for one update cycle."""

    seconds: float
    note: str = ""


def measure_adaptation_latency(fn, *args, **kwargs) -> tuple[StateVector, LatencyResult]:
    """
    Time how long `fn(*args, **kwargs)` takes to produce a new StateVector.

    Used to measure M2: time from ΔU arrival to S(t+1) stable.
    Target: < 500ms for conversational UX (per the survey).
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, LatencyResult(seconds=elapsed)


# --------------------------------------------------------------------------- #
# M3 — Recompute Cost Ratio
# --------------------------------------------------------------------------- #


def recompute_cost_ratio(
    lars_tokens: int, scratch_tokens: int
) -> float:
    """
    Ratio of LLM tokens consumed by LARS update vs. recompute-from-scratch.

    LARS = tokens used by the update path (ΔU parse + merge + verify)
    Scratch = tokens used by a fresh full CoT generation

    Target: ≤ 0.30 (i.e. 70% reduction claim from the user's framing).

    Returns:
        float — lars_tokens / scratch_tokens
    """
    if scratch_tokens <= 0:
        return 0.0
    return round(lars_tokens / scratch_tokens, 4)
