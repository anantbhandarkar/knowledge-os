"""The canonical document model — the durable JSON intermediate for ALL formats.

Every parser returns a CanonicalDoc. It is JSON-serializable (Pydantic), preserves
heading hierarchy, tables (as structured rows), and page numbers, and is what the
chunker consumes. Persist it to `data/canonical/<doc>.json` so re-chunking/re-embedding
never needs the raw file again.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

BlockType = Literal["heading", "paragraph", "list", "table", "code", "caption"]


class CanonicalBlock(BaseModel):
    """One atomic content unit within a section."""
    type: BlockType
    text: str = ""                       # plain-text rendering (tables: TSV-ish)
    level: int | None = None             # heading depth (1-6)
    page: int | None = None              # source page, when known
    rows: list[list[str]] | None = None  # table cells, when type == "table"

    def to_markdown(self) -> str:
        if self.type == "heading" and self.level:
            return f"{'#' * min(self.level, 6)} {self.text}"
        if self.type == "table" and self.rows:
            if not self.rows:
                return ""
            head = "| " + " | ".join(self.rows[0]) + " |"
            sep = "| " + " | ".join("---" for _ in self.rows[0]) + " |"
            body = "\n".join("| " + " | ".join(r) + " |" for r in self.rows[1:])
            return "\n".join([head, sep, body]).rstrip()
        if self.type == "code":
            return f"```\n{self.text}\n```"
        return self.text


class CanonicalSection(BaseModel):
    """A heading-delimited section: its heading path + ordered content blocks."""
    path: list[str] = Field(default_factory=list)  # e.g. ["Auth", "Token Management"]
    blocks: list[CanonicalBlock] = Field(default_factory=list)

    def text(self, *, include_headings: bool = True) -> str:
        blocks = self.blocks if include_headings else [
            b for b in self.blocks if b.type != "heading"
        ]
        return "\n\n".join(b.to_markdown() for b in blocks).strip()


class CanonicalDoc(BaseModel):
    """Structure-preserving, JSON-serializable representation of any source doc."""
    doc_title: str
    source_format: str                       # "pdf" | "docx" | "pptx" | "html" | "md" | "txt"
    source_path: str | None = None
    sections: list[CanonicalSection] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def summary_hint(self) -> str:
        """First non-empty paragraph — a cheap 1-line doc summary for context prefixes."""
        for s in self.sections:
            for b in s.blocks:
                if b.type == "paragraph" and b.text.strip():
                    return b.text.strip()[:200]
        return self.doc_title

    def to_markdown(self) -> str:
        out: list[str] = []
        for s in self.sections:
            out.append(s.text())
        return "\n\n".join(p for p in out if p).strip()

    def iter_section_texts(self):
        """Yield (section_path, body_text) for the chunker — headings excluded
        (they're already carried in section_path + the contextual-embedding prefix)."""
        for s in self.sections:
            t = s.text(include_headings=False)
            if t:
                yield s.path, t
