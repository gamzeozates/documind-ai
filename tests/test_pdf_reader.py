import pymupdf
import pytest

from app.pdf_reader import extract_pdf_pages


def test_extract_pdf_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "DocuMind AI test document",
        )
        document.save(pdf_path)

    result = extract_pdf_pages(str(pdf_path))

    assert len(result) == 1
    assert result[0]["document_name"] == "sample.pdf"
    assert result[0]["page_number"] == 1
    assert "DocuMind AI test document" in result[0]["text"]


def test_extract_pdf_pages_missing_file(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(str(missing_path))
