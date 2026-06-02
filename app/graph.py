"""The LangGraph spine (v2: Express/Deep lane fork).

v1 routed EVERY query through plan->retrieve->...->repair. v2's core insight is
that 60-70% of enterprise queries are simple lookups that should never pay for an
LLM planner or a repair loop. So the first node now CLASSIFIES and forks:

    classify ─┬─ EXPRESS ─▶ retrieve ─▶ rerank ─▶ assemble ─▶ verify ─pass─▶ generate ─▶ END
              │                            ▲                    │
              └─ DEEP ─▶ plan ─────────────┘                    ├─repair (DEEP only)─┐
                                                                └─abstain ─▶ END      │
                          retrieve ◀───────────────────────────────────────(loop)────┘

Both lanes share retrieve / rerank / assemble / verify / generate. The ONLY
structural differences: Deep runs `plan` first, and only Deep may take the repair
edge. Express trades completeness for latency — it never loops.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.models import GateDecision, Lane, PipelineState
from app.verify import decide_gate, score_evidence
from app import retrieval, rerank, generation, planning, routing, compression

MAX_REPAIR_ITERATIONS = 3


# ---- nodes -----------------------------------------------------------------

async def classify_node(state: PipelineState) -> PipelineState:
    """Stage 2: intent classification + lane routing (the v2 fork point)."""
    state.intent = await routing.classify_intent(state)
    state.lane = routing.classify_lane(state.intent, state.query)
    return state


async def plan_node(state: PipelineState) -> PipelineState:
    """Deep Lane only: decompose, choose modes, rewrite query."""
    return await planning.plan(state)


async def retrieve_node(state: PipelineState) -> PipelineState:
    return await retrieval.retrieve(state)


async def rerank_node(state: PipelineState) -> PipelineState:
    return await rerank.fuse_and_rerank(state)


async def assemble_node(state: PipelineState) -> PipelineState:
    """Stage 5: choose chunk-assembly vs full-document stuffing, build context."""
    return await compression.assemble_context(state)


async def verify_node(state: PipelineState) -> PipelineState:
    state.scores = await score_evidence(state)
    state.decision = decide_gate(state.scores, state.repair_count)
    return state


async def repair_node(state: PipelineState) -> PipelineState:
    state.repair_count += 1
    state.rewritten_query = await planning.reformulate(state)
    state.modes = planning.broaden_modes(state.modes)
    return state


async def generate_node(state: PipelineState) -> PipelineState:
    return await generation.generate_with_citations(state)


async def abstain_node(state: PipelineState) -> PipelineState:
    return await generation.abstain(state)


# ---- conditional edges -----------------------------------------------------

def route_by_lane(state: PipelineState) -> str:
    """Express skips planning entirely; Deep plans first."""
    return "plan" if state.lane == Lane.DEEP else "retrieve"


def route_after_verify(state: PipelineState) -> str:
    """Branch out of verify. KEY v2 rule: only the Deep Lane may repair.

    On the Express Lane a REPAIR verdict is downgraded — we generate with explicit
    uncertainty rather than loop, because Express's whole contract is low latency.
    """
    if state.decision == GateDecision.PASS:
        return "generate"
    if state.decision == GateDecision.ABSTAIN:
        return "abstain"
    # decision == REPAIR
    if state.lane == Lane.DEEP and state.repair_count < MAX_REPAIR_ITERATIONS:
        return "repair"
    # Express can't repair, and we won't emit weakly-grounded text -> abstain.
    # (Deep that exhausted its repair budget also abstains.)
    return "abstain"


# ---- assembly --------------------------------------------------------------

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("classify", classify_node)
    g.add_node("plan", plan_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("assemble", assemble_node)
    g.add_node("verify", verify_node)
    g.add_node("repair", repair_node)
    g.add_node("generate", generate_node)
    g.add_node("abstain", abstain_node)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_by_lane, {
        "plan": "plan",
        "retrieve": "retrieve",
    })
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "assemble")
    g.add_edge("assemble", "verify")
    g.add_conditional_edges("verify", route_after_verify, {
        "generate": "generate",
        "repair": "repair",
        "abstain": "abstain",
    })
    g.add_edge("repair", "retrieve")  # the Deep-Lane loop
    g.add_edge("generate", END)
    g.add_edge("abstain", END)
    return g.compile()


COMPILED_GRAPH = build_graph()
