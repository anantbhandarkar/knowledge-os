"""Runtime registry — the active store + providers, wired from env.

Defaults are fully offline (no keys, no DB). Flip with env, no pipeline changes:
    KOS_MODE=api          -> real LLM generation via OpenRouter
    VECTOR_STORE=pgvector -> PostgreSQL + pgvector (production default)
    VECTOR_STORE=turbovec -> two-step quantized scale-out
Stores that need async init expose .setup()/.close(), called by the API lifespan.
"""

from __future__ import annotations

import os

from app.providers.base import Generator
from app.providers.offline import HashEmbedder, LexicalReranker, ExtractiveGenerator
from app.vectorstore.base import VectorStore
from app.vectorstore.memory import InMemoryVectorStore

MODE = os.getenv("KOS_MODE", "offline")
VECTOR_STORE = os.getenv("VECTOR_STORE", "memory")
MODEL = os.getenv("KOS_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
DSN = os.getenv("DATABASE_URL", "postgresql://kos:kos@localhost:5432/knowledge_os")

# Embedding + reranking stay offline/deterministic by default (swap as needed).
EMBEDDER = HashEmbedder()
RERANKER = LexicalReranker()

GENERATOR: Generator
if MODE == "api":
    from app.providers.openrouter import OpenRouterGenerator
    GENERATOR = OpenRouterGenerator(MODEL)
else:
    GENERATOR = ExtractiveGenerator()

STORE: VectorStore
if VECTOR_STORE == "pgvector":
    from app.vectorstore.pgvector import PgVectorStore
    STORE = PgVectorStore(DSN, dim=EMBEDDER.dim)
elif VECTOR_STORE == "turbovec":
    from app.vectorstore.turbovec import TurboVecStore
    STORE = TurboVecStore(DSN, dim=EMBEDDER.dim)
else:
    STORE = InMemoryVectorStore()
