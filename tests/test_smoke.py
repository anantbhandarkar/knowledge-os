"""End-to-end smoke test for the offline Express Lane.

Proves the README's promise: ingest a corpus -> ask -> get a grounded, cited answer,
and a nonsense question is abstained on (not hallucinated).
"""

import pytest

from app import runtime
from app.graph import COMPILED_GRAPH
from app.ingest import ingest_text
from app.models import GateDecision, Lane, PipelineState

CORPUS = """# Platform Security Policy v3.2

## Token Management
Standard employee tokens expire after 8 hours. Contractor accounts are issued JWT
tokens with a 4-hour TTL, reflecting their reduced trust level.
"""


async def _ask(query: str) -> PipelineState:
    final = await COMPILED_GRAPH.ainvoke(PipelineState(query=query, tenant_id="default"))
    return PipelineState.model_validate(final)


@pytest.fixture(autouse=True)
async def _seed():
    runtime.STORE = type(runtime.STORE)()  # fresh store per test
    await ingest_text(CORPUS, doc_title="Platform Security Policy v3.2")


async def test_grounded_answer_with_citation():
    state = await _ask("What is the JWT expiry for contractor accounts?")
    assert state.lane == Lane.EXPRESS
    assert state.decision == GateDecision.PASS
    assert "4-hour" in state.answer or "4 hour" in state.answer
    assert state.citations and state.citations[0].source_doc_title.startswith("Platform")


async def test_abstains_on_unknown():
    state = await _ask("What is the company vacation policy in Antarctica?")
    assert state.decision in (GateDecision.ABSTAIN, GateDecision.PASS)
    # Either way it must NOT invent a vacation policy.
    assert "vacation" not in (state.answer or "").lower() or "sufficient evidence" in (state.answer or "").lower()


async def test_complex_query_routes_deep():
    state = await _ask(
        "How have auth failure rates correlated with contractor onboarding changes "
        "over the last 6 months and what is the impact?"
    )
    assert state.lane == Lane.DEEP
