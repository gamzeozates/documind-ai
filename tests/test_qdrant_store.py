
import numpy as np
import pytest

from app.qdrant_store import QdrantStore


@pytest.fixture
def store(tmp_path):
    database_path = tmp_path / "qdrant_test"

    qdrant_store = QdrantStore(
        storage_path=str(database_path),
        collection_name="test_documents",
        vector_size=3,
    )

    try:
        yield qdrant_store
    finally:
        qdrant_store.close()


def test_collection_created(store):
    assert store.client.collection_exists("test_documents")
    assert store.count() == 0


def test_upsert_chunks(store):
    chunks = [
        {
            "document_name": "example.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Yapay zeka sistemleri",
        },
        {
            "document_name": "example.pdf",
            "page_number": 2,
            "chunk_id": 2,
            "text": "Makine ogrenmesi",
        },
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="document-001",
    )

    assert store.count() == 2

    results = store.search(
        query_vector=np.array([1.0, 0.0, 0.0]),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["document_name"] == "example.pdf"
    assert results[0]["page_number"] == 1
    assert results[0]["document_id"] == "document-001"
    assert results[0]["chunk_id"] == 1


def test_upsert_rejects_empty_document_id(store):
    chunks = [
        {
            "document_name": "test.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Test",
        }
    ]

    embeddings = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id="",
        )


def test_upsert_rejects_invalid_shape(store):
    chunks = [
        {
            "document_name": "test.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Test",
        }
    ]

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id="document-001",
        )


def test_reindex_same_document_does_not_duplicate(store):
    chunks = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "First document chunk",
        },
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": 2,
            "text": "Second document chunk",
        },
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="guide-001",
    )

    assert store.count() == 2

    store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="guide-001",
    )

    assert store.count() == 2


def test_different_documents_do_not_overwrite(store):
    first_chunks = [
        {
            "document_name": "document_a.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "First document content",
        }
    ]

    second_chunks = [
        {
            "document_name": "document_b.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Second document content",
        }
    ]

    first_embedding = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    second_embedding = np.array(
        [[0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    store.upsert_chunks(
        chunks=first_chunks,
        embeddings=first_embedding,
        document_id="document-a",
    )

    store.upsert_chunks(
        chunks=second_chunks,
        embeddings=second_embedding,
        document_id="document-b",
    )

    assert store.count() == 2

    first_results = store.search(
        query_vector=np.array([1.0, 0.0, 0.0]),
        top_k=1,
    )

    second_results = store.search(
        query_vector=np.array([0.0, 1.0, 0.0]),
        top_k=1,
    )

    assert first_results[0]["document_id"] == "document-a"
    assert second_results[0]["document_id"] == "document-b"

    assert first_results[0]["document_name"] == "document_a.pdf"
    assert second_results[0]["document_name"] == "document_b.pdf"


def test_search_filters_by_document_id(store):
    first_chunks = [
        {
            "document_name": "first.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "First document content",
        }
    ]

    second_chunks = [
        {
            "document_name": "second.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Second document content",
        }
    ]

    embeddings = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    store.upsert_chunks(
        chunks=first_chunks,
        embeddings=embeddings,
        document_id="document-a",
    )

    store.upsert_chunks(
        chunks=second_chunks,
        embeddings=embeddings,
        document_id="document-b",
    )

    results = store.search(
        query_vector=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        top_k=5,
        document_id="document-a",
    )

    assert len(results) == 1
    assert results[0]["document_id"] == "document-a"
    assert results[0]["document_name"] == "first.pdf"


def test_delete_document_removes_only_matching_points(store):
    first_chunks = [
        {
            "document_name": "first.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "First document, chunk one",
        },
        {
            "document_name": "first.pdf",
            "page_number": 2,
            "chunk_id": 2,
            "text": "First document, chunk two",
        },
    ]

    second_chunks = [
        {
            "document_name": "second.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Second document content",
        }
    ]

    first_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float32,
    )

    second_embeddings = np.array(
        [[0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    store.upsert_chunks(
        chunks=first_chunks,
        embeddings=first_embeddings,
        document_id="document-a",
    )

    store.upsert_chunks(
        chunks=second_chunks,
        embeddings=second_embeddings,
        document_id="document-b",
    )

    assert store.count() == 3

    store.delete_document("document-a")

    assert store.count() == 1

    deleted_results = store.search(
        query_vector=np.array([1.0, 0.0, 0.0]),
        top_k=5,
        document_id="document-a",
    )

    remaining_results = store.search(
        query_vector=np.array([0.0, 1.0, 0.0]),
        top_k=5,
        document_id="document-b",
    )

    assert deleted_results == []
    assert len(remaining_results) == 1
    assert remaining_results[0]["document_id"] == "document-b"
    assert remaining_results[0]["document_name"] == "second.pdf"


def test_replace_document_removes_obsolete_chunks(store):
    old_chunks = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": i,
            "text": f"Old content {i}",
        }
        for i in range(1, 6)
    ]

    old_embeddings = np.tile(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        (5, 1),
    )

    store.upsert_chunks(
        chunks=old_chunks,
        embeddings=old_embeddings,
        document_id="guide-001",
    )

    assert store.count() == 5

    new_chunks = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": i,
            "text": f"Updated content {i}",
        }
        for i in range(1, 4)
    ]

    new_embeddings = np.tile(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        (3, 1),
    )

    store.replace_document(
        chunks=new_chunks,
        embeddings=new_embeddings,
        document_id="guide-001",
    )

    assert store.count() == 3

    results = store.search(
        query_vector=np.array([0.0, 1.0, 0.0]),
        top_k=10,
        document_id="guide-001",
    )

    assert len(results) == 3
    assert {result["chunk_id"] for result in results} == {1, 2, 3}
    assert all(
        result["text"].startswith("Updated content")
        for result in results
    )


def test_upsert_rejects_nan_embeddings(store):
    chunks = [
        {
            "document_name": "invalid.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Invalid embedding example",
        }
    ]

    embeddings = np.array(
        [[1.0, np.nan, 0.0]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="finite"):
        store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id="invalid-document",
        )

    assert store.count() == 0
