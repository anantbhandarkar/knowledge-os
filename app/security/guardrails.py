"""Offline, deterministic security guardrails — no network, no LLM.

Pure functions over plain strings:
- ``detect_pii`` / ``redact_pii``: regex-based PII spotting and masking.
- ``detect_prompt_injection``: phrase-overlap heuristic returning (flag, score).
- ``map_roles_to_acl``: expand caller roles into ACL tags via a mapping.

All regex patterns are compiled once at module import for cheap reuse, and the
module is import-safe with zero side effects.
"""

from __future__ import annotations

import re

__all__ = [
    "detect_pii",
    "redact_pii",
    "detect_prompt_injection",
    "map_roles_to_acl",
]

# --- PII patterns (order matters: more specific patterns first) ----------------
# SSN must be tried before the bare phone pattern so it is not swallowed.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# 16-digit card-like number: groups of 4 separated by space/hyphen, or 16 solid.
_CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){15}\d\b")
# US phone: optional country code, separators of space/dot/hyphen, optional parens.
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[ .\-]?)?"
    r"(?:\(\d{3}\)|\d{3})"
    r"[ .\-]?\d{3}[ .\-]?\d{4}"
    r"(?!\d)"
)

# Evaluated in this order; earlier matches claim their spans first.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", _EMAIL_RE),
    ("SSN", _SSN_RE),
    ("CREDIT_CARD", _CREDIT_CARD_RE),
    ("PHONE", _PHONE_RE),
)

# --- Prompt-injection heuristic ------------------------------------------------
_INJECTION_PHRASES: tuple[str, ...] = (
    "ignore previous instructions",
    "disregard the above",
    "you are now",
    "system prompt",
    "reveal your instructions",
    "ignore all prior",
)
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(re.escape(phrase), re.IGNORECASE) for phrase in _INJECTION_PHRASES
)

# --- Role -> ACL mapping -------------------------------------------------------
_DEFAULT_ROLE_MAPPING: dict[str, list[str]] = {"admin": ["*"]}


def detect_pii(text: str) -> list[dict]:
    """Find PII spans in ``text``.

    Returns a list of ``{"type", "value", "start", "end"}`` dicts, sorted by
    ``start``. Overlapping matches are resolved by pattern priority (email, SSN,
    credit card, phone): once a character span is claimed it is not re-reported.
    """
    claimed: list[tuple[int, int]] = []
    results: list[dict] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < c_end and end > c_start for c_start, c_end in claimed)

    for pii_type, pattern in _PII_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if _overlaps(start, end):
                continue
            claimed.append((start, end))
            results.append(
                {
                    "type": pii_type,
                    "value": match.group(0),
                    "start": start,
                    "end": end,
                }
            )

    results.sort(key=lambda item: item["start"])
    return results


def redact_pii(text: str) -> str:
    """Replace every detected PII span with ``[REDACTED_<TYPE>]``."""
    spans = detect_pii(text)
    if not spans:
        return text
    # Rebuild left-to-right; spans are non-overlapping and sorted by start.
    out: list[str] = []
    cursor = 0
    for span in spans:
        start, end = span["start"], span["end"]
        out.append(text[cursor:start])
        out.append(f"[REDACTED_{span['type']}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def detect_prompt_injection(text: str) -> tuple[bool, float]:
    """Heuristically score ``text`` for prompt-injection intent.

    ``score = min(1.0, matches * 0.34)`` where ``matches`` is the number of
    known injection phrases present. ``is_injection`` is ``score >= 0.34``.
    """
    matches = sum(1 for pattern in _INJECTION_PATTERNS if pattern.search(text))
    score = min(1.0, matches * 0.34)
    return (score >= 0.34, score)


def map_roles_to_acl(
    roles: list[str],
    mapping: dict[str, list[str]] | None = None,
) -> list[str]:
    """Expand ``roles`` into ACL tags.

    Each role is replaced by its mapped tags; roles absent from ``mapping`` pass
    through as their own tag. The result is de-duplicated and sorted.
    """
    table = _DEFAULT_ROLE_MAPPING if mapping is None else mapping
    tags: set[str] = set()
    for role in roles:
        expanded = table.get(role)
        if expanded is None:
            tags.add(role)
        else:
            tags.update(expanded)
    return sorted(tags)
