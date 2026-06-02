"""Stage 5 — context assembly (v2 promotes this to a routing decision).

Two modes, chosen from how concentrated the retrieval was:
- CHUNKS:        evidence spread across many docs -> assemble top reranked chunks.
- FULL_DOCUMENT: evidence clusters into <=5 docs that fit the window -> stuff the
                 whole parsed docs (NotebookLM pattern) to preserve cross-references.

The decision logic is given verbatim by the spec; it's mechanical, so it's
implemented here rather than handed to you.
"""

from __future__ import annotations

from app.models import AssemblyMode, PipelineState

MAX_FULL_DOCS = 5
CONTEXT_BUDGET_TOKENS = 180_000   # target model window
USABLE_FRACTION = 0.7             # reserve 30% for prompt + history + headroom


def _approx_tokens(text: str) -> int:
    return len(text) // 4  # ~4 chars/token heuristic; swap for a real tokenizer


async def assemble_context(state: PipelineState) -> PipelineState:
    source_docs = {e.chunk_id: e for e in state.evidence}
    unique_doc_titles = {e.doc_title for e in state.evidence}

    # TODO: replace with a real per-doc token lookup from the documents table.
    est_full_doc_tokens = sum(_approx_tokens(e.text) for e in state.evidence) * 10

    if len(unique_doc_titles) <= MAX_FULL_DOCS and \
       est_full_doc_tokens <= CONTEXT_BUDGET_TOKENS * USABLE_FRACTION:
        state.assembly_mode = AssemblyMode.FULL_DOCUMENT
        # TODO: load_full_documents(top_source_doc_ids) -> joined markdown
        state.context = "\n\n".join(e.text for e in state.evidence)
    else:
        state.assembly_mode = AssemblyMode.CHUNKS
        ordered = sorted(state.evidence, key=lambda e: e.rerank_score, reverse=True)
        state.context = "\n\n".join(
            f"[{e.doc_title} §{'/'.join(e.section_path)}]\n{e.text}" for e in ordered
        )
    return state
