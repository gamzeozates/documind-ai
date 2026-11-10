
from pathlib import Path
from tempfile import TemporaryDirectory

from app.document_utils import calculate_document_id
from app.embedding_service import EmbeddingService
from app.pdf_reader import extract_pdf_pages
from app.qdrant_store import QdrantStore
from app.text_chunker import chunk_pages


def main() -> None:
    pdf_path = Path("data/sample_document.pdf")
    document_id = calculate_document_id(pdf_path)

    pages = extract_pdf_pages(str(pdf_path))
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No text chunks were extracted")

    embedding_service = EmbeddingService()

    embeddings = embedding_service.encode_texts(
        [chunk["text"] for chunk in chunks]
    )

    questions = [
        "Gençlerin ekonomik sorunları nelerdir?",
        "Gençler sosyal faaliyetlerde hangi sorunları yaşıyor?",
        "Mars'ta yaşamak için hangi teknolojiler gereklidir?",
        "Bir otomobilin motor yağı nasıl değiştirilir?",
    ]

    with TemporaryDirectory() as temp_dir:
        store = QdrantStore(
            storage_path=temp_dir,
            collection_name="similarity_test",
            vector_size=embeddings.shape[1],
        )

        try:
            store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=document_id,
            )

            for question in questions:
                query_vector = embedding_service.encode_query(
                    question
                )

                results = store.search(
                    query_vector=query_vector,
                    top_k=3,
                    document_id=document_id,
                )

                print(f"\nQUESTION: {question}")

                for result in results:
                    print(
                        f"Rank {result['rank']}: "
                        f"Similarity {result['similarity']:.4f} | "
                        f"Chunk {result['chunk_id']}"
                    )

        finally:
            store.close()


if __name__ == "__main__":
    main()
