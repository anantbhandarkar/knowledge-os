"""Stage 8 — evidence-only generation + the grounding judge used by Stage 6.

`judge_grounding` is the LLM call the verify-gate leans on. `generate_with_citations`
and `abstain` produce the user-facing output. All three are TODO bodies — wire them
to your Generator provider (Claude / Gemini / OpenRouter behind one interface).
"""

from __future__ import annotations

from app.models import Evidence, PipelineState


async def judge_grounding(query: str, evidence: list[Evidence]) -> tuple[float, bool]:
    """Return (grounding_confidence in [0,1], contradiction_detected).

    TODO: structured LLM call. Prompt it to (a) rate how fully the evidence
    supports a confident answer to `query`, and (b) flag any contradictions
    between evidence pieces. Use tool-calling for a typed result.
    """
    return (0.0, False)


async def generate_with_citations(state: PipelineState) -> PipelineState:
    """Evidence-only answer with span-level citations. No claims beyond evidence."""
    # TODO: generator.stream(messages) with the evidence-only system prompt; map
    #       each sentence to its chunk_id; populate state.answer + state.citations.
    state.answer = "[generation not yet wired]"
    state.citations = [e.chunk_id for e in state.evidence]
    return state


async def abstain(state: PipelineState) -> PipelineState:
    """No-answer policy — show partial evidence, suggest sources, never guess."""
    found = "; ".join(f"{e.doc_title}" for e in state.evidence[:3]) or "nothing relevant"
    state.answer = (
        "I don't have sufficient evidence to answer this confidently. "
        f"Here's what I found: {found}."
    )
    state.citations = [e.chunk_id for e in state.evidence]
    return state
