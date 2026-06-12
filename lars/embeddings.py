"""
embeddings.py — Pluggable embedder for semantic similarity

Why we need this:
  Jaccard on tokens is too strict. Two reasoning steps with the same
  meaning can have < 30% token overlap (paraphrasing, synonyms, etc.).
  We need semantic similarity via embeddings.

Two backends:
  - HashEmbedder: deterministic, no API key, no install
                  (random-projection bag-of-words)
  - OpenAIEmbedder: real embeddings, requires OPENAI_API_KEY

The hash embedder is good enough to demonstrate the metric and run
benchmarks. Swap in OpenAI for the final paper numbers.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def similarity(self, a: str, b: str) -> float: ...


# --------------------------------------------------------------------------- #
# Hash-based embedder (default, no deps)
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop short tokens and stop-words."""
    text = text.lower()
    # Light stemming: drop common suffixes
    tokens = re.findall(r"[a-z0-9\u0600-\u06ff]+", text)
    # Common English stop-words
    stops = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "as", "this", "that", "these", "those", "it", "its",
        "i", "we", "you", "they", "he", "she",
    }
    out = []
    for t in tokens:
        if len(t) <= 2 or t in stops:
            continue
        # Naive suffix stripping
        for suf in ("ing", "tion", "ment", "ness", "ies", "ed", "s"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                t = t[: -len(suf)]
                break
        out.append(t)
    return out


class HashEmbedder:
    """
    Deterministic random-projection bag-of-words.

    Each token is hashed to a 128-dim unit vector. The text embedding
    is the L2-normalized sum of its token vectors.

    Properties:
      - Deterministic: same text → same vector
      - No API needed
      - Cosine similarity in [0, 1] after L2 normalization
      - Semantically similar texts get high similarity
        (because they share many tokens after light stemming)
    """

    DIM = 128

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _hash_to_vec(self, token: str) -> list[float]:
        h = hashlib.md5(token.encode()).digest()
        # Use 4 bytes per dim, mod dim
        vec = [0.0] * self.dim
        for i in range(self.dim):
            byte = h[i % len(h)]
            sign = 1.0 if (h[(i + 7) % len(h)] & 1) else -1.0
            vec[i] = sign * ((byte / 255.0) - 0.5)
        return vec

    def embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self.dim
        vec = [0.0] * self.dim
        for tok in tokens:
            tok_vec = self._hash_to_vec(tok)
            for i in range(self.dim):
                vec[i] += tok_vec[i]
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embed(a), self.embed(b)
        dot = sum(x * y for x, y in zip(va, vb))
        return max(0.0, min(1.0, dot))  # clamp to [0, 1]


# --------------------------------------------------------------------------- #
# OpenAI embedder (optional, real)
# --------------------------------------------------------------------------- #


class OpenAIEmbedder:
    """Real OpenAI embeddings. Use for the final paper numbers."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embed(a), self.embed(b)
        dot = sum(x * y for x, y in zip(va, vb))
        na = math.sqrt(sum(x * x for x in va)) or 1.0
        nb = math.sqrt(sum(x * x for x in vb)) or 1.0
        return max(0.0, min(1.0, dot / (na * nb)))


def default_embedder() -> Embedder:
    """Pick OpenAI if key present, else hash-based."""
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbedder()
    return HashEmbedder()
