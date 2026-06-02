"""VectorStore interface + the stored-chunk record.

ACL filtering is part of the search signature on purpose: tenant_id + acl_tags are
applied DURING the search (in the DB WHERE clause for pg/turbo, in Python for the
in-memory demo), never after. That is the spec's non-negotiable security invariant.
"""

from __future__ import annotations

from typing import Protocol, Sequence
from pydantic import BaseModel, Field

from app.models import Hit


class StoredChunk(BaseModel):
    """One indexed chunk: the embedding vector + everything needed to build a Hit."""
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    vector: list[float]
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    tenant_id: str = "default"
    acl_tags: list[str] = Field(default_factory=list)


class VectorStore(Protocol):
    """The v3 storage seam. Implementations: InMemory (default), PgVector, TurboVec."""

    async def add(self, chunks: Sequence[StoredChunk]) -> None: ...

    async def vector_search(
        self, query_vector: list[float], *, k: int,
        tenant_id: str, acl_tags: Sequence[str],
    ) -> list[Hit]: ...

    async def lexical_search(
        self, query: str, *, k: int,
        tenant_id: str, acl_tags: Sequence[str],
    ) -> list[Hit]: ...

    async def delete(self, chunk_ids: Sequence[str]) -> None: ...

    async def count(self, *, tenant_id: str | None = None) -> int: ...
