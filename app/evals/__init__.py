"""Offline, deterministic evaluation harness for the grounded-RAG spine.

Holds a tiny golden set grounded in the demo corpus, a pure faithfulness metric,
and an async runner that drives the compiled graph and scores each item.
"""

from __future__ import annotations
