"""Phase-2 multi-representation generation from a CanonicalDoc.

A single section can be retrieved many ways. Beyond the raw chunk, we precompute
alternate "views" of each section so retrieval can match on the shape that fits the
query: a condensed summary, synthetic Q/A pairs, deterministic keyword anchors, and
extracted entities. All offline paths are deterministic and import-safe — heavy/LLM
work is opt-in via runtime.GENERATOR and never required.
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, Field

from app import runtime
from app.parsing.canonical import CanonicalDoc

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_ERRCODE = re.compile(r"\b[A-Za-z]+[-_]?\d{2,}\b|\b[A-Z]{1,}\d+\b")

_STOP = frozenset(
    "a an the is are was were be been being of for to in on at by with and or "
    "what which who whom how when where why do does did this that these those it "
    "its as from into about can could should would will i you we they he she "
    "not but if then than so such have has had no nor only own same too very "
    "section describe describes our your their".split()
)


class Summary(BaseModel):
    """A condensed view of one section."""

    level: str
    text: str
    section_path: list[str] = Field(default_factory=list)


class QAPair(BaseModel):
    """A synthetic question/answer derived from one section."""

    question: str
    answer: str
    source_section: list[str] = Field(default_factory=list)


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def _first_sentences(text: str, n: int) -> str:
    sents = _sentences(text)
    return " ".join(sents[:n]).strip()


def _path_label(path: list[str]) -> str:
    return path[-1] if path else "this"


async def generate_summaries(doc: CanonicalDoc) -> list[Summary]:
    """One summary per non-empty section.

    Uses runtime.GENERATOR.complete when available; otherwise falls back to a
    deterministic extractive summary (first 1-2 sentences of the section text).
    """
    complete = getattr(runtime.GENERATOR, "complete", None)
    out: list[Summary] = []
    for section in doc.sections:
        text = section.text()
        if not text:
            continue
        if complete is not None:
            prompt = (
                "Summarize the following section in 1-2 sentences. "
                f"Section: {_path_label(section.path)}\n\n{text}"
            )
            summary_text = (await complete(prompt)).strip()
        else:
            summary_text = _first_sentences(text, 2)
        if not summary_text:
            summary_text = _first_sentences(text, 1)
        out.append(
            Summary(level="section", text=summary_text, section_path=list(section.path))
        )
    return out


async def generate_qa_pairs(doc: CanonicalDoc, *, max_per_section: int = 1) -> list[QAPair]:
    """Synthetic Q/A pairs per non-empty section.

    Uses runtime.GENERATOR.complete when available; otherwise falls back to a
    deterministic templated question with an extractive first-sentence answer.
    """
    complete = getattr(runtime.GENERATOR, "complete", None)
    out: list[QAPair] = []
    for section in doc.sections:
        text = section.text()
        if not text:
            continue
        label = _path_label(section.path)
        if complete is not None:
            prompt = (
                f"Write up to {max_per_section} question/answer pair(s) about the "
                f"'{label}' section, one per line as 'Q: ... | A: ...'.\n\n{text}"
            )
            raw = (await complete(prompt)).strip()
            pairs = _parse_qa_lines(raw, max_per_section)
            if not pairs:
                pairs = [_fallback_qa(label, text)]
        else:
            pairs = [_fallback_qa(label, text)]
        for question, answer in pairs[:max_per_section]:
            out.append(
                QAPair(
                    question=question,
                    answer=answer,
                    source_section=list(section.path),
                )
            )
    return out


def _fallback_qa(label: str, text: str) -> tuple[str, str]:
    question = f"What does the '{label}' section describe?"
    answer = _first_sentences(text, 1) or text.strip()
    return question, answer


def _parse_qa_lines(raw: str, limit: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if "Q:" not in line or "A:" not in line:
            continue
        q_part, _, a_part = line.partition("A:")
        question = q_part.replace("Q:", "").strip(" |").strip()
        answer = a_part.strip(" |").strip()
        if question and answer:
            pairs.append((question, answer))
        if len(pairs) >= limit:
            break
    return pairs


def generate_anchors(text: str, k: int = 6) -> list[str]:
    """Deterministic top-k content keywords plus simple variants (no LLM).

    Ranks content tokens by frequency (ties broken by first appearance), then appends
    a handful of bigram variants built from adjacent top keywords. Stable order.
    """
    tokens = _tok(text)
    content = [t for t in tokens if t not in _STOP and len(t) > 2]
    if not content:
        return []

    first_seen: dict[str, int] = {}
    for i, t in enumerate(content):
        if t not in first_seen:
            first_seen[t] = i
    counts = Counter(content)
    ranked = sorted(content_unique(content), key=lambda t: (-counts[t], first_seen[t]))
    top = ranked[:k]

    anchors: list[str] = list(top)
    seen = set(anchors)
    # Simple variants: adjacent bigrams drawn from the original token order, restricted
    # to top keywords, to capture short phrases deterministically.
    top_set = set(top)
    for a, b in zip(content, content[1:]):
        if a in top_set and b in top_set and a != b:
            variant = f"{a} {b}"
            if variant not in seen:
                anchors.append(variant)
                seen.add(variant)
        if len(anchors) >= k * 2:
            break
    return anchors


def content_unique(tokens: list[str]) -> list[str]:
    """Order-preserving unique helper for token lists."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extract_entities(text: str) -> list[str]:
    """Extract ACRONYMS, CamelCase identifiers, and error-code-like tokens.

    Deterministic, dedup'd, and sorted. No LLM.
    """
    found: set[str] = set()
    found.update(_ACRONYM.findall(text))
    found.update(_CAMEL.findall(text))
    found.update(_ERRCODE.findall(text))
    return sorted(found)
