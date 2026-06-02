"""Stage 3 — hybrid retrieval. ACL filters applied BEFORE the query, never after.

Phase-1 contract: run each planned mode against Postgres in parallel and collect
Hits. Vector + BM25 live in the same pgvector DB, so both filter on tenant_id +
acl_tags in the WHERE clause. Bodies are TODO — wire to your db.py session.
"""

from __future__ import annotations

import asyncio

from app.models import Hit, PipelineState, RetrievalMode


async def _vector_search(state: PipelineState) -> list[Hit]:
    # TODO: embed(rewritten_query, kind="query") then pgvector `<=>` ANN search,
    #       WHERE tenant_id = :tid AND acl_tags && :acl  (pre-filter is mandatory).
    return []


async def _bm25_search(state: PipelineState) -> list[Hit]:
    # TODO: tsvector / pg_search full-text, same ACL WHERE clause.
    return []


_MODE_FNS = {
    RetrievalMode.VECTOR: _vector_search,
    RetrievalMode.BM25: _bm25_search,
    # Phase 2: SUMMARY, QA, GRAPH, SQL.
}


async def retrieve(state: PipelineState) -> PipelineState:
    # Express Lane skips planning, so modes may be unset -> default to hybrid.
    modes = state.modes or [RetrievalMode.VECTOR, RetrievalMode.BM25]
    fns = [_MODE_FNS[m] for m in modes if m in _MODE_FNS]
    results = await asyncio.gather(*(fn(state) for fn in fns))
    state.hits = [h for batch in results for h in batch]
    return state
