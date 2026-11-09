
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


class EmbeddingService:
    """Generate normalized embeddings for text."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Convert a list of texts into embedding vectors."""

        if not texts:
            raise ValueError("texts must not be empty")

        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain empty strings")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Convert a user question into one embedding vector."""

        if not query.strip():
            raise ValueError("query must not be empty")

        embeddings = self.encode_texts([query])

        return embeddings[0]
