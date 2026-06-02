"""Stage 0 (Phase-1 slice) — Markdown/text ingestion.

Heading-aware chunking + contextual-embedding prefix + embed + store. Kept small and
real: it actually parses headings, builds section paths, prepends the spec's context
prefix before embedding, and writes to the active VectorStore.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Sequence

from app import runtime
from app.vectorstore.base import StoredChunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _doc_id_for(title: str) -> str:
    return hashlib.sha256(title.encode()).hexdigest()[:16]


def _split_sections(markdown: str) -> list[tuple[list[str], str]]:
    """Return (section_path, body) pairs from markdown headings."""
    sections: list[tuple[list[str], str]] = []
    stack: list[str] = []
    buf: list[str] = []
    path: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((list(path), body))

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            buf.clear()
            level = len(m.group(1))
            stack[:] = stack[: level - 1]
            stack.append(m.group(2).strip())
            path = list(stack)
        else:
            buf.append(line)
    flush()
    return sections or [([], markdown.strip())]


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


async def ingest_text(
    text: str, *, doc_title: str, tenant_id: str = "default",
    acl_tags: Sequence[str] | None = None,
) -> int:
    """Ingest one document. Returns the number of chunks stored."""
    doc_id = _doc_id_for(doc_title)
    summary = text.strip().split("\n\n", 1)[0][:160]
    stored: list[StoredChunk] = []
    raw_texts: list[str] = []

    for section_path, body in _split_sections(text):
        for piece in _chunk(body):
            section = " > ".join(section_path) if section_path else doc_title
            # Spec's contextual-embedding prefix — improves recall for thin chunks.
            context_prefix = f"Context: from '{doc_title}' § {section}. {summary}\n---\n"
            raw_texts.append(context_prefix + piece)
            stored.append(StoredChunk(
                chunk_id=str(uuid.uuid4()), doc_id=doc_id, doc_title=doc_title,
                text=piece, section_path=section_path,
                tenant_id=tenant_id, acl_tags=list(acl_tags or []), vector=[],
            ))

    vectors = await runtime.EMBEDDER.embed(raw_texts, kind="passage")
    for chunk, vec in zip(stored, vectors):
        chunk.vector = vec
    await runtime.STORE.add(stored)
    return len(stored)
