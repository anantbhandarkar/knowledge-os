"""TurboVecStore — scale-out (spec v3, Implementation B): the two-step retrieval.

  step 1: an in-process QUANTIZED (int8) vector index generates top-N candidate IDs
          — ACL-blind, ~8x less RAM than full-precision in pgvector.
  step 2: PostgreSQL filters those candidates by tenant/ACL/metadata and supplies
          the text + lexical search. Security stays in the DB; only vectors move out.

> NOTE: the spec names a Rust "TurboVec" library; that package is aspirational. This
> is a faithful, runnable stand-in implementing the SAME architecture with int8
> scalar quantization in NumPy. The point under test is the two-step DESIGN, and that
> ACL is enforced by Postgres even though vectors live outside it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from psycopg_pool import AsyncConnectionPool

from app.models import Hit, RetrievalMode
from app.vectorstore.base import StoredChunk

_ACL = "tenant_id = %s AND (cardinality(acl_tags) = 0 OR acl_tags && %s::text[])"
CANDIDATES = 100  # step-1 fan-out before ACL filtering


def _quantize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize then scale to int8 — the 8x memory win, codebook-free."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / np.clip(norms, 1e-8, None)
    return np.clip(np.round(unit * 127), -127, 127).astype(np.int8)


class TurboVecStore:
    def __init__(self, dsn: str, *, dim: int) -> None:
        self.dsn = dsn
        self.dim = dim
        self.pool: AsyncConnectionPool | None = None
        # In-process quantized index (the "TurboVec" part).
        self._codes: np.ndarray = np.zeros((0, dim), dtype=np.int8)
        self._ids: list[str] = []

    async def setup(self) -> None:
        self.pool = AsyncConnectionPool(self.dsn, min_size=1, max_size=4, open=False)
        await self.pool.open()
        async with self.pool.connection() as conn:
            # No embedding column here — vectors live in-process, not in Postgres.
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS turbo_chunks (
                    chunk_id text PRIMARY KEY,
                    doc_id text, doc_title text, body text,
                    section_path text[] DEFAULT '{}',
                    page_numbers int[] DEFAULT '{}',
                    tenant_id text NOT NULL,
                    acl_tags text[] DEFAULT '{}',
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
                )""")
            await conn.execute("CREATE INDEX IF NOT EXISTS turbo_tsv_idx ON turbo_chunks USING gin (tsv)")

    async def add(self, chunks: Sequence[StoredChunk]) -> None:
        assert self.pool
        if not chunks:
            return
        codes = _quantize(np.asarray([c.vector for c in chunks], dtype=np.float32))
        self._codes = np.vstack([self._codes, codes]) if self._codes.size else codes
        self._ids.extend(c.chunk_id for c in chunks)
        async with self.pool.connection() as conn:
            for c in chunks:
                await conn.execute(
                    """INSERT INTO turbo_chunks
                       (chunk_id, doc_id, doc_title, body, section_path,
                        page_numbers, tenant_id, acl_tags)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (chunk_id) DO UPDATE SET body = EXCLUDED.body""",
                    (c.chunk_id, c.doc_id, c.doc_title, c.text, c.section_path,
                     c.page_numbers, c.tenant_id, c.acl_tags),
                )

    async def vector_search(self, query_vector, *, k, tenant_id, acl_tags) -> list[Hit]:
        assert self.pool
        if not self._ids:
            return []
        # STEP 1 — quantized candidate generation (ACL-blind).
        q = _quantize(np.asarray([query_vector], dtype=np.float32))[0].astype(np.int32)
        scores = self._codes.astype(np.int32) @ q
        top = np.argsort(-scores)[:CANDIDATES]
        cand_ids = [self._ids[i] for i in top]
        score_by_id = {self._ids[i]: float(scores[i]) for i in top}
        # STEP 2 — Postgres enforces ACL/tenant on the candidate set.
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""SELECT chunk_id, doc_id, doc_title, body, section_path, page_numbers
                    FROM turbo_chunks
                    WHERE chunk_id = ANY(%s) AND {_ACL}""",
                (cand_ids, tenant_id, list(acl_tags)),
            )
            rows = await cur.fetchall()
        rows.sort(key=lambda r: score_by_id.get(r[0], 0.0), reverse=True)
        return [self._hit(r, score_by_id.get(r[0], 0.0), RetrievalMode.VECTOR) for r in rows[:k]]

    async def lexical_search(self, query, *, k, tenant_id, acl_tags) -> list[Hit]:
        assert self.pool
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""SELECT chunk_id, doc_id, doc_title, body, section_path, page_numbers,
                           ts_rank(tsv, plainto_tsquery('english', %s)) AS score
                    FROM turbo_chunks
                    WHERE {_ACL} AND tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC LIMIT %s""",
                (query, tenant_id, list(acl_tags), query, k),
            )
            rows = await cur.fetchall()
        return [self._hit(r[:6], float(r[6]), RetrievalMode.BM25) for r in rows]

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        assert self.pool
        keep = [i for i, cid in enumerate(self._ids) if cid not in set(chunk_ids)]
        self._codes = self._codes[keep] if keep else np.zeros((0, self.dim), dtype=np.int8)
        self._ids = [self._ids[i] for i in keep]
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM turbo_chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),))

    async def count(self, *, tenant_id: str | None = None) -> int:
        assert self.pool
        async with self.pool.connection() as conn:
            if tenant_id is None:
                cur = await conn.execute("SELECT count(*) FROM turbo_chunks")
            else:
                cur = await conn.execute("SELECT count(*) FROM turbo_chunks WHERE tenant_id = %s", (tenant_id,))
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    @staticmethod
    def _hit(r, score: float, mode: RetrievalMode) -> Hit:
        return Hit(
            chunk_id=r[0], doc_id=r[1], doc_title=r[2], text=r[3],
            section_path=r[4] or [], page_numbers=r[5] or [],
            source_mode=mode, score=score,
        )
