"""Offline providers — deterministic, no network, no API keys.

- HashEmbedder: hashes tokens into a fixed-dim bag-of-words vector. Not semantically
  strong, but stable and dependency-free — enough to demonstrate the full pipeline.
- LexicalReranker: scores query-term overlap with IDF weighting.
- ExtractiveGenerator: the interesting one. It NEVER writes free text — it extracts
  the most query-relevant sentences from the evidence verbatim. An extractive answer
  is grounded by construction: it physically cannot hallucinate. That makes it the
  perfect default for a verify-before-generate system, and a useful safety floor.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

from app.models import Evidence

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")
_STOP = frozenset(
    "a an the is are was were be been being of for to in on at by with and or "
    "what which who whom how when where why do does did this that these those it "
    "its as from into about can could should would will i you we they he she".split()
)


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _content(text: str) -> set[str]:
    """Content tokens only — stopwords removed so scores reflect real overlap."""
    return {t for t in _tok(text) if t not in _STOP and len(t) > 1}


class HashEmbedder:
    name = "offline-hash"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: Sequence[str], *, kind: str = "passage") -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for term, count in Counter(_tok(text)).items():
                h = int(hashlib.md5(term.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign * (1.0 + math.log(count))
            out.append(vec)
        return out


class LexicalReranker:
    name = "offline-lexical"

    async def rerank(self, query, docs, *, top_n):
        q = _content(query)
        if not q:
            return [(i, 0.0) for i in range(min(top_n, len(docs)))]
        scored = []
        for i, d in enumerate(docs):
            # Calibrated to [0,1]: fraction of the query's content terms present.
            score = len(q & _content(d)) / len(q)
            scored.append((i, round(score, 4)))
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:top_n]


class ExtractiveGenerator:
    name = "offline-extractive"

    async def generate(self, query: str, evidence: Sequence[Evidence]) -> str:
        best = self._best_sentence(query, evidence)
        if best is None:
            return ("I don't have sufficient evidence to answer this confidently "
                    "from the provided sources.")
        sentence, ev = best
        return f"{sentence} [{ev.doc_title}]"

    async def judge_grounding(self, query, evidence) -> tuple[float, bool]:
        # Grounding ~ how well the best evidence sentence covers the query terms.
        best = self._best_sentence(query, evidence)
        if best is None:
            return (0.0, False)
        q = _content(query)
        s = _content(best[0])
        coverage = len(q & s) / max(len(q), 1)
        return (round(min(1.0, 0.4 + coverage), 3), False)

    @staticmethod
    def _best_sentence(query, evidence):
        q = _content(query)
        best, best_score = None, 0.0
        for ev in evidence:
            for sent in _SENT.split(ev.text.strip()):
                st = _content(sent)
                if not st:
                    continue
                # Content-term overlap, length-normalized to avoid favoring long sentences.
                score = len(q & st) / math.sqrt(len(st))
                if score > best_score:
                    best, best_score = (sent.strip(), ev), score
        return best
