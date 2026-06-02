"""VectorStore abstraction (spec v3).

The retrieval pipeline calls the `VectorStore` interface, never an implementation.
Switching engines is a config change, not a rewrite.

- InMemoryVectorStore: zero-dependency default — runs anywhere, no Postgres, no API
  keys. Numpy cosine + a lightweight lexical index. Perfect for the demo / tests.
- PgVectorStore (Phase 3): pgvector HNSW + tsvector BM25 + ACL in one SQL query.
- TurboVecStore (Phase 3): quantized scale-out for 5M+ chunks.
"""

from app.vectorstore.base import StoredChunk, VectorStore
from app.vectorstore.memory import InMemoryVectorStore

__all__ = ["StoredChunk", "VectorStore", "InMemoryVectorStore"]
