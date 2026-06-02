"""Multi-representation generation tests (offline, deterministic).

Builds a small CanonicalDoc by hand and exercises the four representation
producers. The offline GENERATOR has no ``.complete``, so summaries and Q/A fall
back to extractive paths and stay deterministic.
"""

from __future__ import annotations

from app.parsing.canonical import CanonicalBlock, CanonicalDoc, CanonicalSection
from app.representations import (
    extract_entities,
    generate_anchors,
    generate_qa_pairs,
    generate_summaries,
)


def _doc() -> CanonicalDoc:
    return CanonicalDoc(
        doc_title="Security Policy",
        source_format="md",
        sections=[
            CanonicalSection(
                path=["Token Management"],
                blocks=[
                    CanonicalBlock(type="heading", text="Token Management", level=1),
                    CanonicalBlock(
                        type="paragraph",
                        text=(
                            "Contractor accounts use JWT tokens with a 4-hour TTL. "
                            "Employee tokens expire after 8 hours."
                        ),
                    ),
                ],
            ),
            CanonicalSection(
                path=["Audit"],
                blocks=[
                    CanonicalBlock(type="heading", text="Audit", level=1),
                    CanonicalBlock(
                        type="paragraph",
                        text="Audit logs are retained for 2 years for compliance.",
                    ),
                ],
            ),
        ],
    )


async def test_generate_summaries_offline() -> None:
    summaries = await generate_summaries(_doc())
    assert summaries  # one per non-empty section
    assert all(s.text.strip() for s in summaries)


async def test_generate_qa_pairs_offline() -> None:
    pairs = await generate_qa_pairs(_doc())
    assert pairs
    assert all(p.question.strip() and p.answer.strip() for p in pairs)


def test_generate_anchors_respects_k() -> None:
    text = "JWT token TTL controls expiry for contractor and employee accounts."
    k = 3
    anchors = generate_anchors(text, k=k)
    assert anchors  # non-empty
    # Top-k keywords plus a bounded number of bigram variants (capped at 2*k).
    assert len(anchors) <= 2 * k
    keywords = [a for a in anchors if " " not in a]
    assert len(keywords) <= k


def test_extract_entities_finds_acronyms() -> None:
    entities = extract_entities("The JWT and TTL use RS256")
    assert "JWT" in entities
    assert "TTL" in entities
