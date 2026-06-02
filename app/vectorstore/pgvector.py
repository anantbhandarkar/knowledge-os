"""PgVectorStore — the production default (spec v3, Implementation A).

One PostgreSQL table does it all: vector similarity (pgvector HNSW), BM25-style
full-text (tsvector + GIN), metadata, and ACL — and ACL is enforced IN the WHERE
clause, during the query, never after. That is the spec's security invariant made real.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from pgvector.psycopg import register_vector_async  # type: ignore[import-untyped]
from psycopg_pool import AsyncConnectionPool

from app.models import Hit, RetrievalMode
from app.vectorstore.base import StoredChunk

# ACL predicate shared by both searches: same tenant AND (public-in-tenant OR overlap).
_ACL = "tenant_id = %s AND (cardinality(acl_tags) = 0 OR acl_tags && %s::text[])"


class PgVectorStore:
    def __init__(self, dsn: str, *, dim: int) -> None:
        self.dsn = dsn
        self.dim = dim
        self.pool: AsyncConnectionPool | None = None

    @staticmethod
    async def _configure(conn) -> None:
        await register_vector_async(conn)

    async def setup(self) -> None:
        self.pool = AsyncConnectionPool(
            self.dsn, min_size=1, max_size=4, open=False, configure=self._configure
        )
        await self.pool.open()
        async with self.pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id text PRIMARY KEY,
                    doc_id text, doc_title text, body text,
                    section_path text[] DEFAULT '{{}}',
                    page_numbers int[] DEFAULT '{{}}',
                    tenant_id text NOT NULL,
                    acl_tags text[] DEFAULT '{{}}',
                    embedding vector({self.dim}),
                    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
                )""")
            await conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_vec_idx ON chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            )

    async def add(self, chunks: Sequence[StoredChunk]) -> None:
        assert self.pool
        async with self.pool.connection() as conn:
            for c in chunks:
                await conn.execute(
                    """INSERT INTO chunks
                       (chunk_id, doc_id, doc_title, body, section_path,
                        page_numbers, tenant_id, acl_tags, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (chunk_id) DO UPDATE SET
                         body = EXCLUDED.body, embedding = EXCLUDED.embedding""",
                    (c.chunk_id, c.doc_id, c.doc_title, c.text, c.section_path,
                     c.page_numbers, c.tenant_id, c.acl_tags,
                     np.asarray(c.vector, dtype=np.float32)),
                )

    async def vector_search(self, query_vector, *, k, tenant_id, acl_tags) -> list[Hit]:
        assert self.pool
        v = np.asarray(query_vector, dtype=np.float32)
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""SELECT chunk_id, doc_id, doc_title, body, section_path, page_numbers,
                           1 - (embedding <=> %s) AS score
                    FROM chunks WHERE {_ACL}
                    ORDER BY embedding <=> %s LIMIT %s""",
                (v, tenant_id, list(acl_tags), v, k),
            )
            rows = await cur.fetchall()
        return [self._hit(r, RetrievalMode.VECTOR) for r in rows]

    async def lexical_search(self, query, *, k, tenant_id, acl_tags) -> list[Hit]:
        assert self.pool
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                f"""SELECT chunk_id, doc_id, doc_title, body, section_path, page_numbers,
                           ts_rank(tsv, plainto_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE {_ACL} AND tsv @@ plainto_tsquery('english', %s)
                    ORDER BY score DESC LIMIT %s""",
                (query, tenant_id, list(acl_tags), query, k),
            )
            rows = await cur.fetchall()
        return [self._hit(r, RetrievalMode.BM25) for r in rows]

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        assert self.pool
        async with self.pool.connection() as conn:
            await conn.execute("DELETE FROM chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),))

    async def count(self, *, tenant_id: str | None = None) -> int:
        assert self.pool
        async with self.pool.connection() as conn:
            if tenant_id is None:
                cur = await conn.execute("SELECT count(*) FROM chunks")
            else:
                cur = await conn.execute("SELECT count(*) FROM chunks WHERE tenant_id = %s", (tenant_id,))
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    @staticmethod
    def _hit(r, mode: RetrievalMode) -> Hit:
        return Hit(
            chunk_id=r[0], doc_id=r[1], doc_title=r[2], text=r[3],
            section_path=r[4] or [], page_numbers=r[5] or [],
            source_mode=mode, score=float(r[6]),
        )
