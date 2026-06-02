"""Stage 3 — hybrid retrieval. ACL filters applied DURING the search, never after.

Runs the planned modes in parallel against the active VectorStore. Vector + BM25 are
both ACL/tenant-scoped at the store level (SQL WHERE in production, Python in the demo).
"""

from __future__ import annotations

import asyncio

from app import runtime
from app.models import PipelineState, RetrievalMode

CANDIDATES_PER_MODE = 20


async def _vector_search(state: PipelineState):
    q = state.rewritten_query or state.query
    (qvec,) = await runtime.EMBEDDER.embed([q], kind="query")
    return await runtime.STORE.vector_search(
        qvec, k=CANDIDATES_PER_MODE,
        tenant_id=state.tenant_id, acl_tags=state.acl_tags,
    )


async def _bm25_search(state: PipelineState):
    q = state.rewritten_query or state.query
    return await runtime.STORE.lexical_search(
        q, k=CANDIDATES_PER_MODE,
        tenant_id=state.tenant_id, acl_tags=state.acl_tags,
    )


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
