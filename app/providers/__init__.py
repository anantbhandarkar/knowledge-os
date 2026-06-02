"""Provider layer — Embedder / Reranker / Generator behind Protocols.

Default mode is OFFLINE (deterministic, no API keys, no network) so the repo runs
on `clone + run`. Set KOS_MODE=api and provider keys to swap in real models.
"""
