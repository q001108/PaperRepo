from pathlib import Path

import fitz

from src.pdf_parser import assess_pdf_text_quality, parse_pdf


def test_parse_pdf_returns_page_aware_chunks(tmp_path: Path):
    pdf_path = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Abstract\n\nThis paper introduces a test method.")
    document.save(pdf_path)
    document.close()

    chunks = parse_pdf(pdf_path)

    assert chunks
    assert chunks[0].source_type == "paper"
    assert chunks[0].page_num == 1
    assert chunks[0].section_title == "Abstract"
    assert chunks[0].chunking_strategy == "section"
    assert "This paper introduces a test method." in chunks[0].content


def test_parse_pdf_assigns_section_titles_and_page_ranges(tmp_path: Path):
    pdf_path = tmp_path / "sectioned.pdf"
    document = fitz.open()
    page1 = document.new_page()
    page1.insert_text((72, 72), "1 Introduction\n\nThis section introduces the problem.")
    page2 = document.new_page()
    page2.insert_text((72, 72), "2 Method\n\nThe method builds a section-aware retrieval pipeline.")
    document.save(pdf_path)
    document.close()

    chunks = parse_pdf(pdf_path)

    section_titles = {chunk.section_title for chunk in chunks}
    assert "1 Introduction" in section_titles
    assert "2 Method" in section_titles
    assert all(chunk.page_end is not None for chunk in chunks)


def test_assess_pdf_text_quality_flags_sparse_pdf(tmp_path: Path):
    pdf_path = tmp_path / "scan_like.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    quality = assess_pdf_text_quality(pdf_path)

    assert quality.page_count == 1
    assert quality.extracted_pages == 0
    assert quality.needs_ocr
