
from app.citation_utils import (
    extract_citations,
    validate_citations,
)


def test_extract_citations():
    answer = "Bilgi [1] ve [2] kaynaklarında bulunuyor."

    result = extract_citations(answer)

    assert result == [1, 2]


def test_valid_citations():
    answer = "Ekonomik sorunlar bulunmaktadır [1]."

    sources = [
        {"id": 1},
        {"id": 2},
    ]

    result = validate_citations(answer, sources)

    assert result["is_valid"] is True
    assert result["invalid_ids"] == []
    assert result["cited_ids"] == [1]


def test_invalid_citation():
    answer = "Bu bilgi [1] ve [7] kaynaklarında var."

    sources = [
        {"id": 1},
        {"id": 2},
    ]

    result = validate_citations(answer, sources)

    assert result["is_valid"] is False
    assert result["invalid_ids"] == [7]


def test_duplicate_citations():
    answer = "İlk bilgi [1]. İkinci bilgi [1]."

    sources = [
        {"id": 1},
    ]

    result = validate_citations(answer, sources)

    assert result["is_valid"] is True
    assert result["cited_ids"] == [1]


def test_multiple_invalid_citations():
    answer = "Bilgiler [1], [5] ve [9] kaynaklarında."

    sources = [
        {"id": 1},
        {"id": 2},
    ]

    result = validate_citations(answer, sources)

    assert result["is_valid"] is False
    assert result["invalid_ids"] == [5, 9]


def test_answer_without_citations():
    answer = "Bu cevapta kaynak numarası bulunmuyor."

    sources = [
        {"id": 1},
    ]

    result = validate_citations(answer, sources)

    assert result["cited_ids"] == []
    assert result["invalid_ids"] == []
    assert result["is_valid"] is True


def test_required_citation_is_missing():
    answer = "Gençlerin ekonomik sorunları bulunmaktadır."

    sources = [
        {"id": 1},
        {"id": 2},
    ]

    result = validate_citations(
        answer,
        sources,
        require_citation=True,
    )

    assert result["cited_ids"] == []
    assert result["invalid_ids"] == []
    assert result["missing_citations"] is True
    assert result["is_valid"] is False


def test_required_citation_is_present():
    answer = "Gençlerin ekonomik sorunları bulunmaktadır [1]."

    sources = [
        {"id": 1},
        {"id": 2},
    ]

    result = validate_citations(
        answer,
        sources,
        require_citation=True,
    )

    assert result["cited_ids"] == [1]
    assert result["invalid_ids"] == []
    assert result["missing_citations"] is False
    assert result["is_valid"] is True
