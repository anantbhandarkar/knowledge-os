"""Stage 2 — intent classification + Express/Deep lane routing (the v2 fork).

`classify_intent` is a Phase-1 rule-based stub (the spec explicitly allows this:
"can be rule-based initially"). In Phase 2 it becomes an LLM/ML classifier.

`classify_lane` is the decision that DEFINES v2's economics and is yours to write.
"""

from __future__ import annotations

from app.models import Lane

# v2's intent taxonomy (Stage 2). Each maps to a lane in classify_lane.
EXPRESS_INTENTS = {"factual_lookup", "policy_lookup", "entity_lookup", "faq"}
DEEP_INTENTS = {"analytical", "multi_hop", "troubleshooting", "comparative", "temporal"}

# Cheap lexical signals that a query is structurally complex (multi-step).
_COMPLEX_SIGNALS = (
    "correlat", "compare", "trend", "over the last", "why", "root cause",
    "impact of", "how have", "relationship between", " vs ", " versus ",
)


async def classify_intent(state) -> str:
    """Phase-1 rule-based intent guess. Returns one of the v2 intent labels."""
    q = state.query.lower()
    if any(sig in q for sig in _COMPLEX_SIGNALS):
        return "analytical"
    if q.startswith(("what is", "what's", "define", "who owns")):
        return "factual_lookup"
    if "policy" in q or "allowed" in q or "can i" in q:
        return "policy_lookup"
    return "factual_lookup"


def classify_lane(intent: str, query: str) -> Lane:
    """Route a query to the Express Lane (fast, deterministic, no repair) or the
    Deep Lane (constrained-agentic, decomposition + repair).

    ┌─────────────────────────────────────────────────────────────────────┐
    │  TODO(you): implement this. ~8-12 lines. This is THE v2 decision —    │
    │  it sets the latency/cost/quality balance of the entire system.       │
    └─────────────────────────────────────────────────────────────────────┘

    Inputs:
      - intent: a label from classify_intent (see EXPRESS_INTENTS / DEEP_INTENTS)
      - query:  the raw query string (use for length / signal heuristics)

    The trade-off you are encoding — there is no universally correct answer:

      1. ASYMMETRIC COST OF ERROR. Misrouting a complex query to Express yields a
         confidently wrong, un-repaired answer (bad). Misrouting a simple query to
         Deep just wastes latency/tokens (annoying, not dangerous). Which way
         should an UNCERTAIN query lean — Express (fast) or Deep (safe)?

      2. INTENT vs. SIGNALS. Trust the intent label alone, or also gate on query
         shape (length, the _COMPLEX_SIGNALS above, presence of multiple clauses)?
         A query labeled "factual_lookup" that also says "...and compare to last
         year" is not actually simple.

      3. THE TARGET MIX. v2 says Express should carry 60-70% of traffic. If your
         rule sends 95% to Deep, P95 latency collapses; if it sends 95% to
         Express, complex queries silently degrade. You're tuning toward that mix.

    Constants available: EXPRESS_INTENTS, DEEP_INTENTS, _COMPLEX_SIGNALS, Lane.

    Replace the line below with your routing logic.

    --- STARTER DEFAULT (tune me) ---
    Leans SAFE: any complexity signal OR a deep intent OR a long/multi-clause query
    goes Deep; only clearly-simple lookups take the Express fast path.
    """
    q = query.lower()
    if intent in DEEP_INTENTS:
        return Lane.DEEP
    if any(sig in q for sig in _COMPLEX_SIGNALS):
        return Lane.DEEP
    if len(q.split()) > 25 or " and " in q:  # multi-clause / long → likely multi-hop
        return Lane.DEEP
    return Lane.EXPRESS
