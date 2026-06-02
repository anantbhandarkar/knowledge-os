"""InMemoryVectorStore — the zero-dependency default.

Numpy cosine similarity for vector search + a tiny BM25-ish lexical scorer over the
same chunks. ACL/tenant filtering happens in Python here (the production PgVectorStore
does it in SQL). Good enough to make the whole Express Lane run on `clone + run`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from app.models import Hit, RetrievalMode
from app.vectorstore.base import StoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, StoredChunk] = {}
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []

    # ---- write ----
    async def add(self, chunks: Sequence[StoredChunk]) -> None:
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self._reindex()

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
        self._reindex()

    def _reindex(self) -> None:
        self._ids = list(self._chunks)
        if not self._ids:
            self._matrix = None
            return
        mat = np.array([self._chunks[i].vector for i in self._ids], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.clip(norms, 1e-8, None)

    # ---- ACL (in Python here; SQL WHERE in production) ----
    def _visible(self, c: StoredChunk, tenant_id: str, acl_tags: Sequence[str]) -> bool:
        if c.tenant_id != tenant_id:
            return False
        if not c.acl_tags:
            return True  # public within tenant
        return bool(set(c.acl_tags) & set(acl_tags))

    # ---- vector search ----
    async def vector_search(self, query_vector, *, k, tenant_id, acl_tags) -> list[Hit]:
        if self._matrix is None:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = self._matrix @ q
        order = np.argsort(-sims)
        hits: list[Hit] = []
        for idx in order:
            c = self._chunks[self._ids[idx]]
            if not self._visible(c, tenant_id, acl_tags):
                continue
            hits.append(self._hit(c, float(sims[idx]), RetrievalMode.VECTOR))
            if len(hits) >= k:
                break
        return hits

    # ---- lexical (BM25-lite) ----
    async def lexical_search(self, query, *, k, tenant_id, acl_tags) -> list[Hit]:
        q_terms = set(_tokenize(query))
        if not q_terms:
            return []
        visible = [c for c in self._chunks.values()
                   if self._visible(c, tenant_id, acl_tags)]
        if not visible:
            return []
        N = len(visible)
        df = Counter()
        toks: dict[str, list[str]] = {}
        for c in visible:
            t = _tokenize(c.text)
            toks[c.chunk_id] = t
            for term in set(t) & q_terms:
                df[term] += 1
        scored: list[tuple[float, StoredChunk]] = []
        for c in visible:
            t = toks[c.chunk_id]
            tf = Counter(t)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * (tf[term] / (tf[term] + 1.5))
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [self._hit(c, sc, RetrievalMode.BM25) for sc, c in scored[:k]]

    async def count(self, *, tenant_id: str | None = None) -> int:
        if tenant_id is None:
            return len(self._chunks)
        return sum(1 for c in self._chunks.values() if c.tenant_id == tenant_id)

    @staticmethod
    def _hit(c: StoredChunk, score: float, mode: RetrievalMode) -> Hit:
        return Hit(
            chunk_id=c.chunk_id, doc_id=c.doc_id, doc_title=c.doc_title, text=c.text,
            section_path=c.section_path, page_numbers=c.page_numbers,
            source_mode=mode, score=score,
        )
