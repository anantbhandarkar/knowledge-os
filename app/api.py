"""FastAPI surface for Knowledge OS. Runs the compiled LangGraph pipeline per query.

On startup it auto-ingests the bundled demo corpus so `docker compose up` + a single
/chat call returns a real, cited answer with zero setup.
"""

from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from app import runtime
from app.graph import COMPILED_GRAPH
from app.ingest import ingest_text
from app.models import PipelineState

DEMO_DIR = pathlib.Path(__file__).resolve().parent.parent / "demo" / "corpus"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if hasattr(runtime.STORE, "setup"):
        await runtime.STORE.setup()  # pgvector/turbovec async init (no-op for memory)
    if DEMO_DIR.exists() and await runtime.STORE.count() == 0:
        for md in sorted(DEMO_DIR.glob("*.md")):
            await ingest_text(md.read_text(), doc_title=md.stem, tenant_id="default")
    yield
    if hasattr(runtime.STORE, "close"):
        await runtime.STORE.close()


app = FastAPI(title="Knowledge OS", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    tenant_id: str = "default"
    acl_tags: list[str] = []


class IngestRequest(BaseModel):
    text: str
    doc_title: str
    tenant_id: str = "default"
    acl_tags: list[str] = []


def _run(req: ChatRequest) -> PipelineState:
    return PipelineState(query=req.query, tenant_id=req.tenant_id, acl_tags=req.acl_tags)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": runtime.MODE, "chunks": await runtime.STORE.count()}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    n = await ingest_text(req.text, doc_title=req.doc_title,
                          tenant_id=req.tenant_id, acl_tags=req.acl_tags)
    return {"ingested_chunks": n, "doc_title": req.doc_title}


@app.post("/ingest/file")
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    tenant_id: str = Form("default"),
    acl_tags: str = Form(""),
    refine: bool = Form(False),
):
    """Upload a PDF/DOCX/PPTX/HTML/MD/TXT file -> CanonicalDoc JSON -> indexed."""
    import tempfile
    from app.ingest import ingest_file

    suffix = pathlib.Path(file.filename or "upload.txt").suffix or ".txt"
    tags = [t.strip() for t in acl_tags.split(",") if t.strip()]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        n = await ingest_file(
            tmp_path, tenant_id=tenant_id, acl_tags=tags,
            doc_title=pathlib.Path(file.filename or "upload").stem, refine=refine,
        )
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
    return {"ingested_chunks": n, "doc_title": pathlib.Path(file.filename or "upload").stem,
            "format": suffix.lstrip(".")}


@app.post("/chat")
async def chat(req: ChatRequest):
    final = await COMPILED_GRAPH.ainvoke(_run(req))
    state = PipelineState.model_validate(final)
    return {
        "answer": state.answer,
        "lane": state.lane,
        "verification_result": state.decision,
        "retrieval_confidence": state.scores.retrieval_confidence if state.scores else None,
        "citations": [c.model_dump() for c in state.citations],
    }


@app.post("/search")
async def search(req: ChatRequest):
    from app import retrieval, rerank
    state = await retrieval.retrieve(_run(req))
    state = await rerank.fuse_and_rerank(state)
    return {"evidence": [e.model_dump() for e in state.evidence]}
