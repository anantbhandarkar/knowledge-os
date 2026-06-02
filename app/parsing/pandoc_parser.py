"""Universal parsers built on external converters.

`PandocParser` shells out to pandoc to turn HTML into GitHub-Flavored Markdown,
then reuses the native markdown converter to produce a CanonicalDoc.
`DoclingParser` is an opt-in high-fidelity PDF backend; it is intentionally NOT
auto-registered so the default PDF path stays dependency-light and offline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from app.parsing.base import register
from app.parsing.canonical import CanonicalDoc


class PandocParser:
    """Convert HTML (and other pandoc-supported formats) via the pandoc CLI."""

    formats: tuple[str, ...] = ("html",)

    def parse(self, path: str | Path, *, doc_title: str | None = None) -> CanonicalDoc:
        from app.parsing.parsers import markdown_to_canonical

        src = Path(path)
        try:
            proc = subprocess.run(
                ["pandoc", str(src), "-t", "gfm"],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("pandoc not installed; brew install pandoc") from exc

        title = doc_title or src.stem
        doc = markdown_to_canonical(proc.stdout, doc_title=title)
        doc.source_format = "html"
        doc.source_path = str(src)
        return doc


class DoclingParser:
    """High-fidelity PDF backend using docling. Opt-in; not auto-registered."""

    formats: tuple[str, ...] = ("pdf",)

    def parse(self, path: str | Path, *, doc_title: str | None = None) -> CanonicalDoc:
        from app.parsing.parsers import markdown_to_canonical

        try:
            from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("pip install docling for high-fidelity PDF parsing") from exc

        src = Path(path)
        converter = DocumentConverter()
        result = converter.convert(str(src))
        markdown = result.document.export_to_markdown()

        title = doc_title or src.stem
        doc = markdown_to_canonical(markdown, doc_title=title)
        doc.source_format = "pdf"
        doc.source_path = str(src)
        return doc


register(PandocParser())
