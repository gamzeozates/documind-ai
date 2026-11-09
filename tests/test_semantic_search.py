
from unittest.mock import Mock

import numpy as np
import pytest

from app.semantic_search import SemanticSearch


@pytest.fixture
def search_engine():
    embedding_service = Mock()

    embedding_service.encode_texts.return_value = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    embedding_service.encode_query.return_value = np.array(
        [0.0, 1.0],
        dtype=np.float32,
    )

    chunks = [
        {
            "chunk_id": 1,
            "document_name": "guide.pdf",
            "page_number": 1,
            "text": "Spor faaliyetleri",
        },
        {
            "chunk_id": 2,
            "document_name": "guide.pdf",
            "page_number": 2,
            "text": "Ekonomik sorunlar",
        },
        {
            "chunk_id": 3,
            "document_name": "guide.pdf",
            "page_number": 3,
            "text": "Gençlerin istihdam sorunları",
        },
    ]

    engine = SemanticSearch(embedding_service)
    engine.index_chunks(chunks)

    return engine


def test_search_returns_most_similar_first(search_engine):
    results = search_engine.search("Ekonomik sorunlar", top_k=3)

    assert [result["chunk_id"] for result in results] == [2, 3, 1]
    assert [result["rank"] for result in results] == [1, 2, 3]


def test_search_respects_top_k(search_engine):
    results = search_engine.search("Ekonomi", top_k=2)

    assert len(results) == 2


def test_search_preserves_source_metadata(search_engine):
    results = search_engine.search("Ekonomi", top_k=1)

    assert results[0]["document_name"] == "guide.pdf"
    assert results[0]["page_number"] == 2
    assert results[0]["chunk_id"] == 2


def test_search_without_index_raises_error():
    engine = SemanticSearch(Mock())

    with pytest.raises(RuntimeError):
        engine.search("Ekonomik sorunlar")


def test_search_rejects_invalid_top_k(search_engine):
    with pytest.raises(ValueError):
        search_engine.search("Ekonomi", top_k=0)
