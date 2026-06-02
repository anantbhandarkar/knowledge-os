"""Stage 4 — Reciprocal Rank Fusion, then mandatory cross-encoder reranking.

RRF is implemented (it's pure math and worth seeing). The cross-encoder call is a
TODO — point it at Cohere Rerank 3.5 (API) or bge-reranker-v2-m3 (local ONNX).
Never skip reranking: it's the single highest-leverage quality step in the spec.
"""

from __future__ import annotations

from app import runtime
from app.models import Evidence, Hit, PipelineState

RRF_K = 60          # standard RRF constant
RERANK_TOP_N = 8    # evidence pieces handed to the generator


def reciprocal_rank_fusion(hits: list[Hit]) -> list[Hit]:
    """Fuse per-mode ranked lists. Score = sum 1/(k + rank) across modes."""
    by_mode: dict[str, list[Hit]] = {}
    for h in hits:
        by_mode.setdefault(h.source_mode.value, []).append(h)

    fused: dict[str, tuple[Hit, float]] = {}
    for mode_hits in by_mode.values():
        ranked = sorted(mode_hits, key=lambda h: h.score, reverse=True)
        for rank, h in enumerate(ranked):
            prev = fused.get(h.chunk_id)
            add = 1.0 / (RRF_K + rank)
            if prev:
                fused[h.chunk_id] = (prev[0], prev[1] + add)
            else:
                fused[h.chunk_id] = (h, add)

    return [h for h, _ in sorted(fused.values(), key=lambda t: t[1], reverse=True)]


async def _cross_encoder_rerank(query: str, hits: list[Hit]) -> list[Evidence]:
    if not hits:
        return []
    ranked = await runtime.RERANKER.rerank(
        query, [h.text for h in hits], top_n=RERANK_TOP_N
    )
    return [
        Evidence(
            chunk_id=hits[i].chunk_id, doc_title=hits[i].doc_title, text=hits[i].text,
            section_path=hits[i].section_path, page_numbers=hits[i].page_numbers,
            rerank_score=round(score, 4),
        )
        for i, score in ranked
    ]


async def fuse_and_rerank(state: PipelineState) -> PipelineState:
    fused = reciprocal_rank_fusion(state.hits)
    state.evidence = await _cross_encoder_rerank(
        state.rewritten_query or state.query, fused
    )
    return state
