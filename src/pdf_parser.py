from __future__ import annotations

import logging
from pathlib import Path
import re

import fitz

from src.schemas import PaperChunk


logger = logging.getLogger(__name__)
PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")
MAX_CHARS_PER_CHUNK = 1800


def _split_page_text(page_text: str) -> list[str]:
    paragraphs = [
        re.sub(r"[ \t]+", " ", paragraph).strip()
        for paragraph in PARAGRAPH_SEPARATOR.split(page_text)
    ]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]

    if not paragraphs and page_text.strip():
        paragraphs = [page_text.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        paragraph_size = len(paragraph)
        if current and current_size + paragraph_size + 2 > MAX_CHARS_PER_CHUNK:
            chunks.append("\n\n".join(current))
            current = []
            current_size = 0

        if paragraph_size > MAX_CHARS_PER_CHUNK:
            for start in range(0, paragraph_size, MAX_CHARS_PER_CHUNK):
                chunks.append(paragraph[start : start + MAX_CHARS_PER_CHUNK])
            continue

        current.append(paragraph)
        current_size += paragraph_size + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def parse_pdf(pdf_path: Path) -> list[PaperChunk]:
    """Parse a PDF into page-aware paper chunks using PyMuPDF."""
    if not pdf_path.exists():
        raise ValueError(f"PDF file does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Uploaded file must be a PDF.")

    chunks: list[PaperChunk] = []
    try:
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                page_text = page.get_text("text")
                for chunk_index, content in enumerate(_split_page_text(page_text), start=1):
                    chunks.append(
                        PaperChunk(
                            chunk_id=f"{pdf_path.stem}-p{page_index}-c{chunk_index}",
                            content=content,
                            page_num=page_index,
                            section_title=None,
                            source_file=pdf_path.name,
                        )
                    )
    except fitz.FileDataError as exc:
        raise ValueError("Unable to parse PDF. The file may be corrupted or encrypted.") from exc

    logger.info("Parsed PDF %s into %d chunks", pdf_path, len(chunks))
    return chunks
