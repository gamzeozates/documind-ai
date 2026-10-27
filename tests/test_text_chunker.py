from itertools import pairwise

import pytest

from app.text_chunker import chunk_pages


def test_chunk_pages_preserves_metadata():
    pages = [
        {
            "document_name": "sample.pdf",
            "page_number": 3,
            "text": "A" * 1500,
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=100,
    )

    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 3
    assert chunks[0]["document_name"] == "sample.pdf"
    assert chunks[0]["chunk_id"] == 1
    assert chunks[1]["chunk_id"] == 2


def test_chunk_pages_overlap():
    text = "".join(str(i % 10) for i in range(20))

    pages = [
        {
            "document_name": "test.pdf",
            "page_number": 1,
            "text": text,
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=12,
        chunk_overlap=4,
    )

    assert chunks[0]["text"][-4:] == chunks[1]["text"][:4]


def test_empty_page_is_skipped():
    pages = [
        {
            "document_name": "empty.pdf",
            "page_number": 1,
            "text": "",
        }
    ]

    assert chunk_pages(pages) == []


def test_invalid_chunk_settings():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=100, chunk_overlap=100)


def test_chunk_pages_preserves_words():
    pages = [
        {
            "document_name": "example.pdf",
            "page_number": 2,
            "text": (
                "Yapay zeka sistemleri belgeleri analiz eder "
                "ve kullanicilarin sorularina yanit verir."
            ),
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=35,
        chunk_overlap=10,
    )

    original_words = pages[0]["text"].split()

    for word in original_words:
        assert any(word in chunk["text"].split() for chunk in chunks)

    for chunk in chunks:
        assert len(chunk["text"]) <= 35


def test_multiple_pages_preserve_page_numbers():
    pages = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "text": "Birinci sayfanin icerigi burada bulunuyor.",
        },
        {
            "document_name": "guide.pdf",
            "page_number": 2,
            "text": "Ikinci sayfanin icerigi burada bulunuyor.",
        },
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2
    assert all(chunk["document_name"] == "guide.pdf" for chunk in chunks)


def test_chunk_pages_preserves_all_words_in_order():
    original_text = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda mu nu xi omicron pi rho sigma"
    )

    pages = [
        {
            "document_name": "test.pdf",
            "page_number": 1,
            "text": original_text,
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=30,
        chunk_overlap=0,
    )

    reconstructed_text = " ".join(chunk["text"] for chunk in chunks)

    assert reconstructed_text == original_text


def test_long_word_respects_chunk_size():
    pages = [
        {
            "document_name": "technical.pdf",
            "page_number": 1,
            "text": "mikrodenetleyiciler",
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=10,
        chunk_overlap=3,
    )

    assert len(chunks) > 0

    assert all(len(chunk["text"]) <= 10 for chunk in chunks)

    assert all(chunk["page_number"] == 1 for chunk in chunks)


def test_long_word_preserves_all_characters():
    original_text = "mikrodenetleyiciler"

    pages = [
        {
            "document_name": "technical.pdf",
            "page_number": 1,
            "text": original_text,
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=10,
        chunk_overlap=0,
    )

    reconstructed_text = "".join(chunk["text"] for chunk in chunks)

    assert reconstructed_text == original_text


def test_chunk_overlap_preserves_shared_words():
    pages = [
        {
            "document_name": "policy.pdf",
            "page_number": 1,
            "text": ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"),
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=30,
        chunk_overlap=10,
    )

    assert len(chunks) >= 2

    for current, following in pairwise(chunks):
        current_words = current["text"].split()
        following_words = following["text"].split()

        assert any(word in following_words for word in current_words)

        assert current["page_number"] == following["page_number"]


def test_overlap_is_at_chunk_boundaries():
    pages = [
        {
            "document_name": "overlap.pdf",
            "page_number": 1,
            "text": ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"),
        }
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=30,
        chunk_overlap=10,
    )

    assert len(chunks) >= 2

    for current, following in pairwise(chunks):
        current_words = current["text"].split()
        following_words = following["text"].split()

        has_boundary_overlap = any(
            current_words[-count:] == following_words[:count]
            for count in range(
                1,
                min(len(current_words), len(following_words)) + 1,
            )
        )

        assert has_boundary_overlap
