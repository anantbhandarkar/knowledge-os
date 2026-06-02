"""Semantic answer cache — return a prior answer when a new query is close enough.

Keyed by embedding cosine similarity (not exact string match), so paraphrases of a
previously answered question hit the cache. Entries expire after a TTL. The embedder
is whatever the caller passes (offline HashEmbedder by default), so this stays fully
deterministic and import-safe offline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class _Embedder(Protocol):
    async def embed(
        self, texts: Sequence[str], *, kind: str = ...
    ) -> list[list[float]]: ...


@dataclass
class _Entry:
    vector: np.ndarray
    query: str
    answer: str
    expires_at: float


class SemanticCache:
    """Cosine-similarity answer cache with per-entry TTL expiry."""

    def __init__(self, *, threshold: float = 0.9, ttl_s: float = 3600.0) -> None:
        self.threshold = threshold
        self.ttl_s = ttl_s
        self._entries: list[_Entry] = []

    async def get(self, query: str, embedder: _Embedder) -> str | None:
        vecs = await embedder.embed([query], kind="query")
        vec = np.asarray(vecs[0], dtype=np.float64)
        now = time.time()
        self._entries = [e for e in self._entries if now <= e.expires_at]
        best_answer: str | None = None
        best_score = self.threshold
        for entry in self._entries:
            score = self._cosine(vec, entry.vector)
            if score >= best_score:
                best_score = score
                best_answer = entry.answer
        return best_answer

    async def set(self, query: str, answer: str, embedder: _Embedder) -> None:
        vecs = await embedder.embed([query], kind="query")
        vec = np.asarray(vecs[0], dtype=np.float64)
        self._entries.append(
            _Entry(
                vector=vec,
                query=query,
                answer=answer,
                expires_at=time.time() + self.ttl_s,
            )
        )

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)
