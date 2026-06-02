"""Optional LLM refinement of canonical JSON.

Off by default and offline-safe: with `enabled=False` (or when the active
generator has no `.complete` method, as the offline default does), this is a pure
passthrough. When enabled against a real LLM, it asks the model to clean each
section's text — fixing OCR artifacts, broken hyphenation, merged words, and stray
whitespace — WITHOUT adding, removing, or inventing facts. Any failure on a section
leaves that section untouched, and the input document is never mutated in place.
"""

from __future__ import annotations

from app import runtime
from app.parsing.canonical import CanonicalBlock, CanonicalDoc

_MAX_SECTIONS = 20

_PROMPT = (
    "You are cleaning text extracted from a document. Fix OCR artifacts, broken "
    "hyphenation, merged words, and stray whitespace. Do NOT add, remove, or invent "
    "any facts. Do NOT summarize or rephrase meaning. Return ONLY the cleaned text, "
    "with no preamble, commentary, or formatting.\n\n"
    "TEXT:\n{text}"
)


async def refine_canonical(doc: CanonicalDoc, *, enabled: bool = False) -> CanonicalDoc:
    """Return a refined copy of `doc`, or the doc unchanged when refinement is off."""
    if not enabled:
        return doc

    fn = getattr(runtime.GENERATOR, "complete", None)
    if fn is None:
        return doc

    refined = doc.model_copy(deep=True)
    for section in refined.sections[:_MAX_SECTIONS]:
        original = section.text()
        if not original.strip():
            continue
        try:
            cleaned = await fn(_PROMPT.format(text=original))
        except Exception:
            continue
        if not isinstance(cleaned, str) or not cleaned.strip():
            continue
        section.blocks = [CanonicalBlock(type="paragraph", text=cleaned.strip())]

    return refined


__all__ = ["refine_canonical"]
