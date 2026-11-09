
from pathlib import Path

from qdrant_client import QdrantClient

from app.embedding_service import EmbeddingService

COLLECTION_NAME = "documind_documents"
STORAGE_PATH = Path("data/qdrant_storage")


def main() -> None:
    question = "Gençlerin ekonomik sorunları nelerdir?"

    print("Loading embedding model...")
    embedding_service = EmbeddingService()

    print("Encoding question...")
    query_vector = embedding_service.encode_query(question)

    print("Connecting to Qdrant...")
    client = QdrantClient(path=str(STORAGE_PATH))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError("Qdrant collection does not exist")

        print("Searching stored vectors...")

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=3,
            with_payload=True,
        )

        print(f"\nQuestion: {question}")
        print(f"Results found: {len(response.points)}")
        print("-" * 60)

        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}

            print(f"Rank: {rank}")
            print(f"Chunk ID: {payload.get('chunk_id')}")
            print(f"Document: {payload.get('document_name')}")
            print(f"Page: {payload.get('page_number')}")
            print(f"Similarity: {point.score:.4f}")
            print(f"Text: {str(payload.get('text', ''))[:300]}")
            print("-" * 60)

    finally:
        client.close()


if __name__ == "__main__":
    main()
