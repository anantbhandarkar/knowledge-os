"""Parsing layer (Phase 2).

Design principle: the pipeline NEVER consumes raw PDF/DOCX/PPTX. Every format is
first parsed into one **canonical JSON document model** (`CanonicalDoc`) — a
structure-preserving, inspectable, version-controllable intermediate. Chunking,
representations, and retrieval all consume the canonical JSON, not the raw bytes.

Quality ladder for producing canonical JSON (best effort first):
  1. Native structure-aware parser (python-docx / python-pptx / pdfplumber / Docling)
  2. pandoc universal conversion (anything pandoc reads -> markdown -> canonical)
  3. optional LLM refinement (clean noisy extraction into well-formed JSON)
"""

from app.parsing.canonical import CanonicalBlock, CanonicalDoc, CanonicalSection
from app.parsing.base import Parser, detect_format, get_parser, parse_file

__all__ = [
    "CanonicalBlock", "CanonicalDoc", "CanonicalSection",
    "Parser", "detect_format", "get_parser", "parse_file",
]
