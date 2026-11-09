
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models


class QdrantStore:
    """Manage a local Qdrant vector collection."""

    def __init__(
        self,
        storage_path: str = "data/qdrant_storage",
        collection_name: str = "documind_documents",
        vector_size: int = 384,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size

        Path(storage_path).mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(path=storage_path)

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )


    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        document_id: str,
    ) -> None:
        """Store document chunks using deterministic UUIDs."""

        if not document_id.strip():
            raise ValueError("document_id must not be empty")

        if not chunks:
            raise ValueError("chunks must not be empty")

        if (
            embeddings.ndim != 2
            or embeddings.shape != (len(chunks), self.vector_size)
        ):
            raise ValueError("Invalid embeddings shape")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must contain finite numbers")

        points = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_id = chunk["chunk_id"]

            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"documind:{document_id}:{chunk_id}",
                )
            )

            point = models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "document_id": document_id,
                    "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )



    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        document_id: str | None = None,
    ) -> list[dict]:
        """Search vectors, optionally filtering by document ID."""

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if query_vector.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")

        if query_vector.shape[0] != self.vector_size:
            raise ValueError("Invalid query vector size")

        query_filter = None

        if document_id is not None:
            if not document_id.strip():
                raise ValueError("document_id must not be blank")

            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results = []

        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}

            results.append(
                {
                    **payload,
                    "rank": rank,
                    "similarity": float(point.score),
                }
            )

        return results


    def delete_document(self, document_id: str) -> None:
        """Delete all points belonging to a document."""

        if not document_id.strip():
            raise ValueError("document_id must not be blank")

        document_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=document_filter,
            ),
            wait=True,
        )

    def replace_document(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        document_id: str,
    ) -> None:
        """Replace all stored chunks for an existing document."""

        if not document_id.strip():
            raise ValueError("document_id must not be blank")

        if not chunks:
            raise ValueError("chunks must not be empty")

        if (
            embeddings.ndim != 2
            or embeddings.shape != (len(chunks), self.vector_size)
        ):
            raise ValueError("Invalid embeddings shape")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must contain finite numbers")

        self.delete_document(document_id)

        self.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
        )


    def count(self) -> int:
        """Return the number of stored points."""

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def close(self) -> None:
        """Close the Qdrant client."""

        self.client.close()
