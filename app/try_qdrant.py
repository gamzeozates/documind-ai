
from pathlib import Path

from qdrant_client import QdrantClient, models

COLLECTION_NAME = "documind_documents"
VECTOR_SIZE = 384


def main() -> None:
    storage_path = Path("data/qdrant_storage")
    storage_path.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(storage_path))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

            print("Collection created successfully!")
        else:
            print("Collection already exists.")

        collection = client.get_collection(COLLECTION_NAME)

        print(f"Collection name: {COLLECTION_NAME}")
        print(f"Vector size: {collection.config.params.vectors.size}")
        print(f"Distance: {collection.config.params.vectors.distance}")
        print(f"Stored points: {collection.points_count}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
