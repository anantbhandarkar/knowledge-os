"""Stage 2 + 7 — query understanding, planning, and repair reformulation.

Phase-1 stubs: deterministic defaults so the graph runs end-to-end today. Swap the
bodies for LLM tool-calls (intent classification, decomposition) in Phase 2.
"""

from __future__ import annotations

from app.models import PipelineState, RetrievalMode


async def plan(state: PipelineState) -> PipelineState:
    # Phase 1: hybrid by default. Phase 2: LLM classifies intent + picks modes.
    state.rewritten_query = state.query
    state.modes = [RetrievalMode.VECTOR, RetrievalMode.BM25]
    return state


async def reformulate(state: PipelineState) -> str:
    # Phase 1: no-op passthrough. Phase 2: LLM rewrites for broader recall.
    return state.rewritten_query or state.query


def broaden_modes(modes: list[RetrievalMode]) -> list[RetrievalMode]:
    """On repair, add modes we haven't tried (summary, qa, graph)."""
    extra = [RetrievalMode.SUMMARY, RetrievalMode.QA, RetrievalMode.GRAPH]
    return list(dict.fromkeys([*modes, *extra]))
