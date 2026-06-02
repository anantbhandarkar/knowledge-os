"""Async eval runner: drive the compiled graph over the golden set and score hits.

Assumes the caller has already seeded the vector store (this module never ingests).
Imports of the graph and models are lazy so the package stays import-safe offline.
"""

from __future__ import annotations

from typing import Any


async def run_eval() -> dict[str, Any]:
    """Run every GOLDEN item through the graph and return per-item hit scores.

    A hit requires both: the expected substring appears in the answer (case-
    insensitive) AND the gate decision matches the expected decision.
    """
    from app.graph import COMPILED_GRAPH
    from app.models import PipelineState

    from app.evals.golden import GOLDEN

    items: list[dict[str, Any]] = []
    hits = 0

    for entry in GOLDEN:
        query = entry["query"]
        expect_substring = entry["expect_substring"]
        expect_decision = entry["expect_decision"]

        state = PipelineState(query=query, tenant_id="default")
        final = await COMPILED_GRAPH.ainvoke(state)
        result = PipelineState.model_validate(final)

        answer = result.answer or ""
        decision = result.decision.value if result.decision is not None else None

        substring_ok = expect_substring.lower() in answer.lower()
        decision_ok = decision == expect_decision
        hit = substring_ok and decision_ok
        if hit:
            hits += 1

        items.append(
            {
                "query": query,
                "hit": hit,
                "decision": decision,
                "answer": answer,
            }
        )

    return {"n": len(GOLDEN), "hits": hits, "items": items}
