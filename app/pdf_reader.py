from pathlib import Path

import pymupdf


def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extract text and page numbers from a PDF document."""

    pdf_path = Path(file_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    pages = []

    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text")

            pages.append(
                {
                    "document_name": pdf_path.name,
                    "page_number": page_index + 1,
                    "text": text.strip(),
                }
            )

    return pages
