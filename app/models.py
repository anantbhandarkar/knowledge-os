"""Domain types for the grounded-RAG spine.

These Pydantic models are the data contracts that flow between LangGraph nodes
(plan -> retrieve -> fuse+rerank -> verify -> repair? -> generate). Every node
takes and returns a `PipelineState`, so the graph stays inspectable and testable.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    SUMMARY = "summary"
    QA = "qa"
    GRAPH = "graph"
    SQL = "sql"


class Lane(str, Enum):
    """v2 routing fork. Express = deterministic & fast (60-70% of traffic);
    Deep = constrained-agentic, repair-capable (only when one pass can't answer)."""
    EXPRESS = "express"
    DEEP = "deep"


class AssemblyMode(str, Enum):
    """Stage 5 context-assembly routing (v2)."""
    CHUNKS = "chunks"            # default: dense evidence from many docs
    FULL_DOCUMENT = "full_doc"   # stuff whole docs when retrieval concentrates


class Hit(BaseModel):
    """One retrieved candidate, before reranking."""
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_mode: RetrievalMode
    score: float  # raw per-retriever score (not comparable across modes)


class Evidence(BaseModel):
    """A reranked, citation-ready piece of context handed to the generator."""
    chunk_id: str
    doc_title: str
    text: str
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    rerank_score: float  # cross-encoder score, comparable across all evidence


class VerificationScores(BaseModel):
    """The three signals the verify-gate weighs. Each in [0.0, 1.0]."""
    retrieval_confidence: float  # did we find relevant material at all?
    grounding_confidence: float  # does the evidence support a clear answer?
    citation_confidence: float   # can every claim trace to a specific source?
    contradiction_detected: bool = False  # do evidence pieces conflict?


class GateDecision(str, Enum):
    PASS = "pass"        # evidence is sufficient -> generate
    REPAIR = "repair"    # weak but recoverable -> reformulate & retry
    ABSTAIN = "abstain"  # insufficient after repairs -> no-answer response


class PipelineState(BaseModel):
    """The single object threaded through every LangGraph node."""
    # --- input ---
    query: str
    tenant_id: str
    acl_tags: list[str] = Field(default_factory=list)

    # --- stage 2: classify + route (v2) ---
    intent: str | None = None
    lane: Lane | None = None  # set by the classifier; decides the whole downstream path

    # --- stage 2: plan (Deep Lane only) ---
    rewritten_query: str | None = None
    modes: list[RetrievalMode] = Field(default_factory=list)

    # --- stage 3-4: retrieve + rerank ---
    hits: list[Hit] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    # --- stage 5: context assembly (v2) ---
    assembly_mode: AssemblyMode | None = None
    context: str | None = None

    # --- stage 6-7: verify + repair ---
    scores: VerificationScores | None = None
    decision: GateDecision | None = None
    repair_count: int = 0

    # --- stage 8: output ---
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)
