"""Runtime registry — the active store + providers, wired from config.

Phase-1 default is fully offline. Swapping to real providers/pgvector is a matter of
constructing different objects here based on env (KOS_MODE, VECTOR_STORE, ...).
"""

from __future__ import annotations

import os

from app.vectorstore.memory import InMemoryVectorStore
from app.providers.offline import HashEmbedder, LexicalReranker, ExtractiveGenerator

MODE = os.getenv("KOS_MODE", "offline")

# Offline default: runs with zero external dependencies.
EMBEDDER = HashEmbedder()
RERANKER = LexicalReranker()
GENERATOR = ExtractiveGenerator()
STORE = InMemoryVectorStore()

# NOTE: when MODE == "api", construct OpenAI/Cohere/Anthropic-backed providers and
# (optionally) PgVectorStore here. The pipeline code below never changes.
