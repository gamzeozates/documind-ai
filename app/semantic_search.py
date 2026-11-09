
import numpy as np

from app.embedding_service import EmbeddingService


class SemanticSearch:
    """Search document chunks using embedding similarity."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def index_chunks(self, chunks: list[dict]) -> None:
        """Create embeddings for document chunks."""

        if not chunks:
            raise ValueError("chunks must not be empty")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embedding_service.encode_texts(texts)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Invalid chunk embedding shape")

        self.chunks = [chunk.copy() for chunk in chunks]
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the most similar indexed chunks."""

        if self.embeddings is None:
            raise RuntimeError("No chunks have been indexed")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_embedding = self.embedding_service.encode_query(query)

        if (
            query_embedding.ndim != 1
            or query_embedding.shape[0] != self.embeddings.shape[1]
        ):
            raise ValueError("Invalid query embedding shape")

        scores = self.embeddings @ query_embedding

        ranked_indices = np.argsort(-scores, kind="stable")
        selected_indices = ranked_indices[:top_k]

        results = []

        for rank, index in enumerate(selected_indices, start=1):
            chunk = self.chunks[int(index)]

            results.append(
                {
                    **chunk,
                    "rank": rank,
                    "similarity": float(scores[index]),
                }
            )

        return results
