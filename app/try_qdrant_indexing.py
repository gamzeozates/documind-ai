
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.embedding_service import EmbeddingService
from app.pdf_reader import extract_pdf_pages
from app.text_chunker import chunk_pages

COLLECTION_NAME = "documind_documents"
STORAGE_PATH = Path("data/qdrant_storage")


def main() -> None:
    # Step 1: Read PDF and create chunks
    pages = extract_pdf_pages("data/sample_document.pdf")

    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=100,
    )

    print(f"Total chunks: {len(chunks)}")

    # Step 2: Generate embeddings
    embedding_service = EmbeddingService()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.encode_texts(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # Step 3: Connect to Qdrant
    client = QdrantClient(path=str(STORAGE_PATH))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError("Qdrant collection does not exist")

        # Step 4: Prepare Qdrant points
        points = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            point = models.PointStruct(
                id=chunk["chunk_id"],
                vector=embedding.tolist(),
                payload={
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                },
            )

            points.append(point)

        # Step 5: Save points
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        # Step 6: Verify stored point count
        count_result = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        )

        print(f"Stored points: {count_result.count}")
        print("Indexing completed successfully!")

    finally:
        client.close()


if __name__ == "__main__":
    main()
