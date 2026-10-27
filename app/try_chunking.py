from pathlib import Path

from app.pdf_reader import extract_pdf_pages
from app.text_chunker import chunk_pages


def main():
    pdf_path = Path("data/sample_document.pdf")

    pages = extract_pdf_pages(str(pdf_path))

    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=100,
    )

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks:
        print("-" * 50)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Document: {chunk['document_name']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Characters: {len(chunk['text'])}")

        print("Text preview:")
        print(chunk["text"][:200])


if __name__ == "__main__":
    main()
