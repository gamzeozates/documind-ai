
from pathlib import Path
from tempfile import TemporaryDirectory

from app.document_utils import calculate_document_id
from app.embedding_service import EmbeddingService
from app.pdf_reader import extract_pdf_pages
from app.qdrant_store import QdrantStore
from app.text_chunker import chunk_pages


def main() -> None:
    pdf_path = Path("data/sample_document.pdf")

    # 1. Generate a stable document identifier.
    document_id = calculate_document_id(pdf_path)

    # 2. Extract and split the PDF text.
    pages = extract_pdf_pages(str(pdf_path))
    chunks = chunk_pages(pages)

    print("Document:", pdf_path.name)
    print("Pages:", len(pages))
    print("Chunks:", len(chunks))

    if not chunks:
        raise ValueError("No text chunks were extracted")

    # 3. Generate vector embeddings.
    embedding_service = EmbeddingService()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.encode_texts(texts)

    print("Embedding shape:", embeddings.shape)

    # 4. Use an isolated, temporary Qdrant database.
    with TemporaryDirectory() as temp_dir:
        store = QdrantStore(
            storage_path=temp_dir,
            collection_name="pipeline_test",
            vector_size=embeddings.shape[1],
        )

        try:
            store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=document_id,
            )

            print("Stored points:", store.count())

            # 5. Search using a natural-language question.
            question = "Gençlerin ekonomik sorunları nelerdir?"
            query_vector = embedding_service.encode_query(
                question
            )

            results = store.search(
                query_vector=query_vector,
                top_k=3,
                document_id=document_id,
            )

            print("\nQuestion:", question)

            for result in results:
                print(
                    f"\nRank: {result['rank']}"
                    f"\nSimilarity: {result['similarity']:.4f}"
                    f"\nDocument: {result['document_name']}"
                    f"\nPage: {result['page_number']}"
                    f"\nText: {result['text'][:250]}..."
                )
        finally:
            store.close()


if __name__ == "__main__":
    main()
