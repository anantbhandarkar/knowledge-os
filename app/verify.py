"""Stage 6 — the verification gate. The single most important file in the system.

`score_evidence` produces three signals + a contradiction flag (implemented for you
below using the reranker scores you already have, plus an LLM contradiction check).

`decide_gate` turns those signals into one of three actions: PASS / REPAIR / ABSTAIN.
THIS is the function that defines whether the system hallucinates, over-refuses, or
loops forever. It is yours to write — see the TODO.
"""

from __future__ import annotations

from app.models import GateDecision, PipelineState, VerificationScores
from app import generation

# Tunable thresholds. Start here; move them once your golden-set eval tells you to.
PASS_THRESHOLD = 0.70          # spec's default confidence_threshold
REPAIR_FLOOR = 0.35            # below this even repair is unlikely to help
MAX_REPAIR_ITERATIONS = 3


async def score_evidence(state: PipelineState) -> VerificationScores:
    """Derive the three confidence signals from the reranked evidence.

    - retrieval_confidence: do we have *any* strong candidate? (top rerank score)
    - grounding_confidence: LLM judges whether evidence supports a clear answer.
    - citation_confidence:  share of evidence with usable source spans.
    - contradiction:        LLM flags conflicting claims across evidence.
    """
    if not state.evidence:
        return VerificationScores(
            retrieval_confidence=0.0,
            grounding_confidence=0.0,
            citation_confidence=0.0,
        )

    top = max(e.rerank_score for e in state.evidence)
    retrieval_confidence = min(1.0, max(0.0, top))

    grounding, contradiction = await generation.judge_grounding(
        state.query, state.evidence
    )
    cited = sum(1 for e in state.evidence if e.page_numbers or e.section_path)
    citation_confidence = cited / len(state.evidence)

    return VerificationScores(
        retrieval_confidence=retrieval_confidence,
        grounding_confidence=grounding,
        citation_confidence=citation_confidence,
        contradiction_detected=contradiction,
    )


def decide_gate(scores: VerificationScores, repair_count: int) -> GateDecision:
    """Decide PASS / REPAIR / ABSTAIN from the verification scores.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  TODO(you): implement this. ~8-12 lines. This is the behavioral       │
    │  heart of the whole system — it shapes hallucination rate, refusal    │
    │  rate, and latency/cost all at once.                                  │
    └─────────────────────────────────────────────────────────────────────┘

    Inputs available on `scores`:
      - retrieval_confidence   float [0,1]
      - grounding_confidence   float [0,1]
      - citation_confidence    float [0,1]
      - contradiction_detected bool
    And `repair_count` (how many repair loops we've already burned).

    Constants you can use: PASS_THRESHOLD, REPAIR_FLOOR, MAX_REPAIR_ITERATIONS.

    Design questions whose answers ARE this function — there is no single right
    answer, which is exactly why it's yours:

      1. Which signal is the veto? The spec says "every claim must map to an
         evidence span" — so should LOW citation_confidence block a PASS even
         when grounding is high? (Strict = safer, but refuses more.)

      2. What do you do with a contradiction? Surface it to the user as a PASS
         ("sources disagree: ..."), or REPAIR to find a tie-breaker first?

      3. When repair_count has hit MAX_REPAIR_ITERATIONS but scores are still
         middling — ABSTAIN, or PASS with explicit uncertainty language?
         (Looping forever is the bug REPAIR_FLOOR / the cap exists to prevent.)

      4. Should a high grounding_confidence rescue a mediocre retrieval_confidence,
         or must BOTH clear the bar? (Weighted blend vs. hard AND-gate.)

    Replace the line below with your logic.
    """
    raise NotImplementedError(
        "Implement decide_gate — see the design questions above. "
        "Return GateDecision.PASS / .REPAIR / .ABSTAIN."
    )
