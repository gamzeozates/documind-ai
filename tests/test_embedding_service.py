
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.embedding_service import EmbeddingService


@pytest.fixture
def embedding_service():
    with patch(
        "app.embedding_service.SentenceTransformer"
    ) as mock_model_class:
        mock_model = MagicMock()

        mock_model.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        mock_model_class.return_value = mock_model

        service = EmbeddingService()

        yield service


def test_encode_texts(embedding_service):
    result = embedding_service.encode_texts(
        ["First text", "Second text"]
    )

    assert result.shape == (2, 3)
    assert result.dtype == np.float32


def test_encode_query(embedding_service):
    result = embedding_service.encode_query(
        "What is this document about?"
    )

    assert result.shape == (3,)
    assert result.dtype == np.float32


def test_encode_texts_rejects_empty_list(embedding_service):
    with pytest.raises(ValueError):
        embedding_service.encode_texts([])


def test_encode_texts_rejects_blank_text(embedding_service):
    with pytest.raises(ValueError):
        embedding_service.encode_texts(["Valid text", "   "])


def test_encode_query_rejects_blank_query(embedding_service):
    with pytest.raises(ValueError):
        embedding_service.encode_query("   ")
