from __future__ import annotations

import logging
from pathlib import Path
import re
from dataclasses import dataclass

import fitz

from src.schemas import PaperChunk, PdfTextQuality


logger = logging.getLogger(__name__)
PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")
MAX_CHARS_PER_CHUNK = 1800
LOW_TEXT_PAGE_CHARS = 80
MIN_AVG_CHARS_PER_PAGE = 120
MIN_READABLE_CHAR_RATIO = 0.45
COMMON_SECTION_TITLES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "proposed method",
    "experiments",
    "experiment",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgments",
    "acknowledgements",
    "摘要",
    "引言",
    "介绍",
    "相关工作",
    "方法",
    "实验",
    "实验设置",
    "结果",
    "评估",
    "讨论",
    "结论",
    "参考文献",
}


@dataclass(frozen=True)
class _ParagraphUnit:
    content: str
    page_num: int
    section_title: str | None


@dataclass(frozen=True)
class _SectionGroup:
    section_title: str | None
    units: list[_ParagraphUnit]


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


def _normalize_text_block(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _paragraphs_from_page(page_text: str) -> list[str]:
    paragraphs = [_normalize_text_block(paragraph) for paragraph in PARAGRAPH_SEPARATOR.split(page_text)]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if not paragraphs and page_text.strip():
        paragraphs = [_normalize_text_block(page_text)]
    return paragraphs


def _looks_like_section_heading(line: str) -> bool:
    normalized = re.sub(r"[ \t]+", " ", line).strip()
    if not normalized or len(normalized) > 120:
        return False

    lowered = normalized.lower().strip(".:")
    if lowered in COMMON_SECTION_TITLES:
        return True

    if re.match(r"^(?:section\s+)?[1-9]\d?(?:\.\d+)*\.?\s+\S", lowered):
        return len(normalized.split()) <= 14

    if re.match(r"^[IVXLCDM]+\.\s+\S", normalized):
        return len(normalized.split()) <= 14

    if re.match(r"^[A-Z]\.\s+\S", normalized):
        return len(normalized.split()) <= 14

    if normalized.endswith((".", ",", ";")):
        return False

    return False


def _is_noise_line(line: str) -> bool:
    lowered = line.lower()
    if line.isdigit():
        return True
    noise_markers = {
        "ieee transactions",
        "authorized licensed use",
        "downloaded on",
        "restrictions apply",
        "digital object identifier",
        "zhu et al.:",
        "vol.",
        "no.",
    }
    return any(marker in lowered for marker in noise_markers)


def _extract_heading(paragraph: str) -> str | None:
    first_line = paragraph.splitlines()[0].strip()
    abstract_match = re.match(r"^(abstract)\s*[—-]", first_line, flags=re.IGNORECASE)
    if abstract_match:
        return "Abstract"
    if _looks_like_section_heading(first_line):
        return first_line
    return None


def _toc_sections_by_page(document: fitz.Document) -> dict[int, str]:
    sections_by_page: dict[int, str] = {}
    try:
        toc = document.get_toc(simple=True)
    except Exception:
        return sections_by_page

    for _level, title, page_num in toc:
        if page_num >= 1:
            sections_by_page[page_num] = str(title).strip()
    return sections_by_page


def _is_standalone_heading_line(line: str, heading: str) -> bool:
    normalized_line = re.sub(r"[ \t]+", " ", line).strip()
    normalized_heading = re.sub(r"[ \t]+", " ", heading).strip()
    return normalized_line == normalized_heading


def _extract_paragraph_units(document: fitz.Document) -> tuple[list[_ParagraphUnit], bool]:
    units: list[_ParagraphUnit] = []
    toc_sections = _toc_sections_by_page(document)
    active_section: str | None = None
    found_heading = bool(toc_sections)

    def flush_buffer(buffer: list[str], page_num: int) -> None:
        if not buffer:
            return
        units.append(
            _ParagraphUnit(
                content="\n".join(buffer).strip(),
                page_num=page_num,
                section_title=active_section,
            )
        )
        buffer.clear()

    for page_index, page in enumerate(document, start=1):
        if page_index in toc_sections:
            active_section = toc_sections[page_index]

        buffer: list[str] = []
        for raw_line in page.get_text("text").splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                flush_buffer(buffer, page_index)
                continue
            if _is_noise_line(line):
                continue

            in_references = (active_section or "").lower().strip(".:") == "references"
            heading = None if in_references else _extract_heading(line)
            if heading:
                flush_buffer(buffer, page_index)
                active_section = heading
                found_heading = True
                if _is_standalone_heading_line(line, heading):
                    continue

            buffer.append(line)
            buffer_size = sum(len(item) + 1 for item in buffer)
            if buffer_size >= 600 and line.endswith((".", "。", ":", "：")):
                flush_buffer(buffer, page_index)

        flush_buffer(buffer, page_index)

    return units, found_heading


def _group_units_by_section(units: list[_ParagraphUnit]) -> list[_SectionGroup]:
    groups: list[_SectionGroup] = []
    current_section: str | None = None
    current_units: list[_ParagraphUnit] = []

    for unit in units:
        if current_units and unit.section_title != current_section:
            groups.append(_SectionGroup(section_title=current_section, units=current_units))
            current_units = []

        current_section = unit.section_title
        current_units.append(unit)

    if current_units:
        groups.append(_SectionGroup(section_title=current_section, units=current_units))

    return groups


def _slugify(value: str | None) -> str:
    if not value:
        return "unknown"
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "-", value).strip("-").lower()
    return slug[:48] or "section"


def _build_section_chunks(pdf_path: Path, groups: list[_SectionGroup]) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    chunk_index = 1

    for group in groups:
        current: list[_ParagraphUnit] = []
        current_size = 0

        def flush_current() -> None:
            nonlocal chunk_index, current, current_size
            if not current:
                return
            page_start = current[0].page_num
            page_end = current[-1].page_num
            section_slug = _slugify(group.section_title)
            chunks.append(
                PaperChunk(
                    chunk_id=f"{pdf_path.stem}-{section_slug}-c{chunk_index}",
                    content="\n\n".join(unit.content for unit in current),
                    page_num=page_start,
                    page_end=page_end,
                    section_title=group.section_title,
                    source_file=pdf_path.name,
                    chunking_strategy="section",
                )
            )
            chunk_index += 1
            current = []
            current_size = 0

        for unit in group.units:
            paragraph_size = len(unit.content)
            if current and current_size + paragraph_size + 2 > MAX_CHARS_PER_CHUNK:
                flush_current()

            if paragraph_size > MAX_CHARS_PER_CHUNK:
                for start in range(0, paragraph_size, MAX_CHARS_PER_CHUNK):
                    content = unit.content[start : start + MAX_CHARS_PER_CHUNK]
                    chunks.append(
                        PaperChunk(
                            chunk_id=f"{pdf_path.stem}-{_slugify(group.section_title)}-c{chunk_index}",
                            content=content,
                            page_num=unit.page_num,
                            page_end=unit.page_num,
                            section_title=group.section_title,
                            source_file=pdf_path.name,
                            chunking_strategy="section",
                        )
                    )
                    chunk_index += 1
                continue

            current.append(unit)
            current_size += paragraph_size + 2

        flush_current()

    return chunks


def _build_page_chunks(pdf_path: Path, document: fitz.Document) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    for page_index, page in enumerate(document, start=1):
        page_text = page.get_text("text")
        for chunk_index, content in enumerate(_split_page_text(page_text), start=1):
            chunks.append(
                PaperChunk(
                    chunk_id=f"{pdf_path.stem}-p{page_index}-c{chunk_index}",
                    content=content,
                    page_num=page_index,
                    page_end=page_index,
                    section_title=None,
                    source_file=pdf_path.name,
                    chunking_strategy="page",
                )
            )
    return chunks


def _is_readable_char(char: str) -> bool:
    if char.isalnum():
        return True
    return "\u4e00" <= char <= "\u9fff"


def assess_pdf_text_quality(pdf_path: Path) -> PdfTextQuality:
    """Estimate whether PDF text extraction looks good enough before OCR fallback."""
    if not pdf_path.exists():
        raise ValueError(f"PDF file does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Uploaded file must be a PDF.")

    try:
        with fitz.open(pdf_path) as document:
            page_texts = [page.get_text("text") for page in document]
    except fitz.FileDataError as exc:
        raise ValueError("Unable to parse PDF. The file may be corrupted or encrypted.") from exc

    page_count = len(page_texts)
    page_char_counts = [len(text.strip()) for text in page_texts]
    extracted_pages = sum(count > 0 for count in page_char_counts)
    low_text_pages = sum(count < LOW_TEXT_PAGE_CHARS for count in page_char_counts)
    avg_chars_per_page = sum(page_char_counts) / page_count if page_count else 0.0
    compact_text = "".join(text for text in page_texts if not text.isspace())
    readable_chars = sum(1 for char in compact_text if _is_readable_char(char))
    readable_char_ratio = readable_chars / len(compact_text) if compact_text else 0.0
    low_text_ratio = low_text_pages / page_count if page_count else 1.0
    needs_ocr = (
        page_count == 0
        or extracted_pages == 0
        or low_text_ratio >= 0.5
        or avg_chars_per_page < MIN_AVG_CHARS_PER_PAGE
        or readable_char_ratio < MIN_READABLE_CHAR_RATIO
    )
    message = (
        "Text extraction looks sparse or noisy; OCR fallback may be needed."
        if needs_ocr
        else "Text extraction looks usable; OCR is probably not needed."
    )
    return PdfTextQuality(
        page_count=page_count,
        extracted_pages=extracted_pages,
        low_text_pages=low_text_pages,
        avg_chars_per_page=round(avg_chars_per_page, 2),
        readable_char_ratio=round(readable_char_ratio, 3),
        needs_ocr=needs_ocr,
        message=message,
    )


def parse_pdf(pdf_path: Path) -> list[PaperChunk]:
    """Parse a PDF into section-aware paper chunks using PyMuPDF."""
    if not pdf_path.exists():
        raise ValueError(f"PDF file does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Uploaded file must be a PDF.")

    chunks: list[PaperChunk] = []
    try:
        with fitz.open(pdf_path) as document:
            units, found_heading = _extract_paragraph_units(document)
            if found_heading and units:
                chunks = _build_section_chunks(pdf_path, _group_units_by_section(units))
            else:
                chunks = _build_page_chunks(pdf_path, document)
    except fitz.FileDataError as exc:
        raise ValueError("Unable to parse PDF. The file may be corrupted or encrypted.") from exc

    logger.info("Parsed PDF %s into %d chunks", pdf_path, len(chunks))
    return chunks
