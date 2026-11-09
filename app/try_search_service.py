
from pathlib import Path

from app.embedding_service import EmbeddingService
from app.pdf_reader import extract_pdf_pages
from app.semantic_search import SemanticSearch
from app.text_chunker import chunk_pages


def main() -> None:
    pdf_path = Path("data/sample_document.pdf")

    print("Reading PDF...")
    pages = extract_pdf_pages(str(pdf_path))

    print("Creating chunks...")
    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=100,
    )

    print("Loading embedding model...")
    embedding_service = EmbeddingService()

    print("Indexing document chunks...")
    search_engine = SemanticSearch(embedding_service)
    search_engine.index_chunks(chunks)

    question = "Gençlerin ekonomik sorunları nelerdir?"

    print("Searching document...")
    results = search_engine.search(
        query=question,
        top_k=3,
    )

    print(f"\nQuestion: {question}")
    print(f"Total chunks: {len(chunks)}")
    print("-" * 60)

    for result in results:
        print(f"Rank: {result['rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Document: {result['document_name']}")
        print(f"Page: {result['page_number']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['text'][:300]}")
        print("-" * 60)


if __name__ == "__main__":
    main()
