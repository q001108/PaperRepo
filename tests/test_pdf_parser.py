from pathlib import Path

import fitz

from src.pdf_parser import parse_pdf


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
    assert "Abstract" in chunks[0].content
