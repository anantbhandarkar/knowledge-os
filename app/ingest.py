"""Stage 0 — ingestion. Raw file/text -> CanonicalDoc JSON -> chunks -> embeddings.

The pipeline NEVER consumes raw document bytes. Every input is first turned into a
CanonicalDoc (app/parsing), optionally LLM-refined, persisted as JSON, and only then
chunked + embedded + stored. Re-chunking/re-embedding later needs only the JSON.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Sequence

from app import runtime
from app.parsing import CanonicalDoc, parse_file
from app.parsing.parsers import markdown_to_canonical
from app.parsing.refine import refine_canonical
from app.vectorstore.base import StoredChunk

CANONICAL_DIR = Path("data/canonical")


def _doc_id_for(title: str) -> str:
    return hashlib.sha256(title.encode()).hexdigest()[:16]


def _chunk(body: str, max_chars: int = 700) -> list[str]:
    """Paragraph-greedy chunking with a char cap (fallback for unstructured text)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def persist_canonical(doc: CanonicalDoc) -> Path:
    """Write the canonical JSON to disk so raw docs are never needed again."""
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", doc.doc_title)[:80] or "doc"
    out = CANONICAL_DIR / f"{safe}.json"
    out.write_text(doc.model_dump_json(indent=2))
    return out


async def _ingest_canonical(
    doc: CanonicalDoc, *, tenant_id: str, acl_tags: Sequence[str] | None,
) -> int:
    """Chunk a CanonicalDoc, embed with contextual prefixes, store. Returns chunk count."""
    doc_id = _doc_id_for(doc.doc_title)
    summary = doc.summary_hint()
    stored: list[StoredChunk] = []
    raw_texts: list[str] = []

    for section_path, body in doc.iter_section_texts():
        section = " > ".join(section_path) if section_path else doc.doc_title
        for piece in _chunk(body):
            # Spec's contextual-embedding prefix — improves recall for thin chunks.
            context_prefix = f"Context: from '{doc.doc_title}' § {section}. {summary}\n---\n"
            raw_texts.append(context_prefix + piece)
            stored.append(StoredChunk(
                chunk_id=str(uuid.uuid4()), doc_id=doc_id, doc_title=doc.doc_title,
                text=piece, section_path=list(section_path),
                tenant_id=tenant_id, acl_tags=list(acl_tags or []), vector=[],
            ))

    if not stored:
        return 0
    vectors = await runtime.EMBEDDER.embed(raw_texts, kind="passage")
    for chunk, vec in zip(stored, vectors):
        chunk.vector = vec
    await runtime.STORE.add(stored)
    return len(stored)


async def ingest_text(
    text: str, *, doc_title: str, tenant_id: str = "default",
    acl_tags: Sequence[str] | None = None,
) -> int:
    """Ingest raw markdown/text (parsed into a CanonicalDoc first)."""
    doc = markdown_to_canonical(text, doc_title=doc_title)
    return await _ingest_canonical(doc, tenant_id=tenant_id, acl_tags=acl_tags)


async def ingest_file(
    path: str | Path, *, tenant_id: str = "default",
    acl_tags: Sequence[str] | None = None, doc_title: str | None = None,
    refine: bool = False, persist: bool = True,
) -> int:
    """Ingest any supported file: PDF/DOCX/PPTX/HTML/MD/TXT -> CanonicalDoc -> store.

    `refine=True` runs an optional LLM cleanup pass (api mode only). The canonical
    JSON is persisted to data/canonical/ by default.
    """
    doc = parse_file(path, doc_title=doc_title)
    doc = await refine_canonical(doc, enabled=refine)
    if persist:
        persist_canonical(doc)
    return await _ingest_canonical(doc, tenant_id=tenant_id, acl_tags=acl_tags)
