"""Stage 8 — evidence-only generation + the grounding judge used by Stage 6.

Both delegate to the active Generator provider. The offline default is extractive
(grounded by construction); the api mode swaps in a real LLM with the closed-book
examiner system prompt. Citations are built from the evidence the answer drew on.
"""

from __future__ import annotations

from app import runtime
from app.models import Citation, Evidence, GateDecision, PipelineState


async def judge_grounding(query: str, evidence: list[Evidence]) -> tuple[float, bool]:
    """(grounding_confidence in [0,1], contradiction_detected)."""
    if not evidence:
        return (0.0, False)
    return await runtime.GENERATOR.judge_grounding(query, evidence)


def _citations(evidence: list[Evidence], limit: int = 3) -> list[Citation]:
    return [
        Citation(
            claim_text=e.text.strip().split(". ")[0][:160],
            source_doc_title=e.doc_title,
            section_path=e.section_path,
            page_numbers=e.page_numbers,
            evidence_span=e.text.strip()[:240],
            chunk_id=e.chunk_id,
            confidence=round(min(1.0, max(0.0, e.rerank_score)), 3),
        )
        for e in evidence[:limit]
    ]


async def generate_with_citations(state: PipelineState) -> PipelineState:
    state.answer = await runtime.GENERATOR.generate(state.query, state.evidence)
    # Align the primary citation with the evidence the answer actually drew from.
    core = (state.answer or "").rsplit(" [", 1)[0][:60]
    evidence = list(state.evidence)
    if core:
        evidence.sort(key=lambda e: core not in e.text)  # matching chunk first
    state.citations = _citations(evidence)
    return state


async def abstain(state: PipelineState) -> PipelineState:
    state.decision = GateDecision.ABSTAIN  # reflect the actual outcome in the response
    found = "; ".join(dict.fromkeys(e.doc_title for e in state.evidence[:3])) or "nothing relevant"
    state.answer = (
        "I don't have sufficient evidence to answer this confidently. "
        f"Here's what I found: {found}. You may want to check those sources directly."
    )
    state.citations = _citations(state.evidence)
    return state
