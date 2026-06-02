"""Parser protocol + format detection + dispatch.

Each format parser is registered against the extensions it handles and returns a
CanonicalDoc. `parse_file()` is the single entry point the ingestion layer calls.
Implementations live in app/parsing/parsers.py (native) and pandoc_parser.py (universal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.parsing.canonical import CanonicalDoc

# Extension -> canonical format name.
_EXT_FORMAT = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".html": "html", ".htm": "html",
    ".md": "md", ".markdown": "md",
    ".txt": "txt", ".text": "txt",
}


class Parser(Protocol):
    formats: tuple[str, ...]
    def parse(self, path: str | Path, *, doc_title: str | None = None) -> CanonicalDoc: ...


# Format -> Parser instance. Populated by register() at import of parsers modules.
_REGISTRY: dict[str, Parser] = {}


def register(parser: Parser) -> Parser:
    for fmt in parser.formats:
        _REGISTRY[fmt] = parser
    return parser


def detect_format(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    fmt = _EXT_FORMAT.get(ext)
    if fmt is None:
        raise ValueError(f"Unsupported file extension: {ext!r} ({path})")
    return fmt


def get_parser(fmt: str) -> Parser:
    # Import implementations lazily so optional heavy deps load only when needed.
    if not _REGISTRY:
        from app.parsing import parsers  # noqa: F401  (registers native parsers)
    parser = _REGISTRY.get(fmt)
    if parser is None:
        # pandoc fallback covers many additional formats.
        from app.parsing.pandoc_parser import PandocParser
        return PandocParser()
    return parser


def parse_file(path: str | Path, *, doc_title: str | None = None) -> CanonicalDoc:
    """Single entry point: raw file -> CanonicalDoc JSON."""
    fmt = detect_format(path)
    return get_parser(fmt).parse(path, doc_title=doc_title)
