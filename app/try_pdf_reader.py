from pathlib import Path

from app.pdf_reader import extract_pdf_pages


def main():
    pdf_path = Path("data/sample_document.pdf")

    pages = extract_pdf_pages(str(pdf_path))

    print(f"Total pages: {len(pages)}")

    for page in pages:
        print("-" * 50)
        print(f"Document: {page['document_name']}")
        print(f"Page: {page['page_number']}")
        print(f"Characters: {len(page['text'])}")

        print("Text preview:")
        print(page["text"][:300])


if __name__ == "__main__":
    main()
