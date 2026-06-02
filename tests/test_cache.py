"""Semantic answer-cache tests (offline, deterministic).

Uses the offline HashEmbedder so cosine similarity is stable: a repeated query
hits the cache, while an unrelated query falls below the similarity threshold.
"""

from __future__ import annotations

from app.cache import SemanticCache
from app.providers.offline import HashEmbedder


async def test_cache_hit_on_same_query() -> None:
    cache = SemanticCache(threshold=0.6)
    embedder = HashEmbedder()
    await cache.set("jwt token expiry for contractors", "4 hours", embedder)
    assert await cache.get("jwt token expiry for contractors", embedder) == "4 hours"


async def test_cache_miss_on_unrelated_query() -> None:
    cache = SemanticCache(threshold=0.6)
    embedder = HashEmbedder()
    await cache.set("jwt token expiry for contractors", "4 hours", embedder)
    assert await cache.get("unrelated lunch menu pizza", embedder) is None
