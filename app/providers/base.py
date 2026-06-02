"""Provider Protocols. Real backends (OpenAI/Cohere/Anthropic/local) implement these."""

from __future__ import annotations

from typing import Protocol, Sequence

from app.models import Evidence


class Embedder(Protocol):
    name: str
    dim: int
    async def embed(self, texts: Sequence[str], *, kind: str = "passage") -> list[list[float]]: ...
    # kind = "passage" | "query" — asymmetric models (BGE, Jina) need this


class Reranker(Protocol):
    name: str
    async def rerank(self, query: str, docs: Sequence[str], *, top_n: int) -> list[tuple[int, float]]: ...
    # returns (original_index, score), sorted desc


class Generator(Protocol):
    name: str
    async def generate(self, query: str, evidence: Sequence[Evidence]) -> str: ...
    async def judge_grounding(self, query: str, evidence: Sequence[Evidence]) -> tuple[float, bool]: ...
