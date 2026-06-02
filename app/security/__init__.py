"""Security utilities for knowledge-os — offline, deterministic guardrails.

Currently exposes :mod:`app.security.guardrails`, which provides pure functions
for PII detection/redaction, prompt-injection heuristics, and role-to-ACL mapping.
No network or LLM calls; safe to import anywhere.
"""

from __future__ import annotations
