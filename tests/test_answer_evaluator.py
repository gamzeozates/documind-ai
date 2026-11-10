
from unittest.mock import MagicMock

import pytest

from app.answer_evaluator import (
    AnswerEvaluator,
    EvaluationResult,
    EvaluationStatus,
)


def test_evaluator_returns_not_evaluated():
    evaluator = AnswerEvaluator()

    result = evaluator.evaluate(
        answer="Gençler ekonomik sorunlar yaşamaktadır [1].",
        sources=[
            {
                "id": 1,
                "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            }
        ],
    )

    assert result.status == EvaluationStatus.NOT_EVALUATED
    assert result.is_supported is False
    assert result.confidence is None


def test_supported_status():
    result = EvaluationResult(
        status=EvaluationStatus.SUPPORTED,
        explanation="The source supports the answer.",
        confidence=0.95,
    )

    assert result.is_supported is True
    assert result.status.value == "supported"


def test_unsupported_status():
    result = EvaluationResult(
        status=EvaluationStatus.UNSUPPORTED,
        explanation="The source contradicts the answer.",
        confidence=0.90,
    )

    assert result.is_supported is False
    assert result.status.value == "unsupported"


def test_evaluator_rejects_empty_answer():
    evaluator = AnswerEvaluator()

    with pytest.raises(ValueError, match="Answer"):
        evaluator.evaluate(
            answer="   ",
            sources=[{"text": "Valid source text"}],
        )


def test_evaluator_rejects_empty_sources():
    evaluator = AnswerEvaluator()

    with pytest.raises(ValueError, match="source"):
        evaluator.evaluate(
            answer="Valid answer",
            sources=[],
        )


def test_evaluator_rejects_source_without_text():
    evaluator = AnswerEvaluator()

    with pytest.raises(ValueError, match="source"):
        evaluator.evaluate(
            answer="Valid answer",
            sources=[{"id": 1, "text": ""}],
        )


def test_llm_evaluator_supported_answer():
    llm_service = MagicMock()

    llm_service.generate.return_value = (
        '{"status": "supported", '
        '"explanation": "Cevap kaynakla uyumludur."}'
    )

    evaluator = AnswerEvaluator(llm_service=llm_service)

    result = evaluator.evaluate(
        answer="Gençler ekonomik sorunlar yaşıyor [1].",
        sources=[
            {
                "id": 1,
                "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            }
        ],
    )

    assert result.status == EvaluationStatus.SUPPORTED
    assert result.is_supported is True
    assert result.explanation == "Cevap kaynakla uyumludur."

    llm_service.generate.assert_called_once()


def test_llm_evaluator_unsupported_answer():
    llm_service = MagicMock()

    llm_service.generate.return_value = (
        '{"status": "unsupported", '
        '"explanation": "Kaynakta işsizlik oranı bulunmuyor."}'
    )

    evaluator = AnswerEvaluator(llm_service=llm_service)

    result = evaluator.evaluate(
        answer="Gençlerin %80'i işsizdir [1].",
        sources=[
            {
                "id": 1,
                "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            }
        ],
    )

    assert result.status == EvaluationStatus.UNSUPPORTED
    assert result.is_supported is False
    assert "işsizlik oranı" in result.explanation

    llm_service.generate.assert_called_once()


def test_llm_evaluator_invalid_json():
    llm_service = MagicMock()
    llm_service.generate.return_value = "Invalid JSON response"

    evaluator = AnswerEvaluator(llm_service=llm_service)

    result = evaluator.evaluate(
        answer="Gençler ekonomik sorunlar yaşıyor [1].",
        sources=[
            {
                "id": 1,
                "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            }
        ],
    )

    assert result.status == EvaluationStatus.NOT_EVALUATED
    assert result.is_supported is False
    assert result.confidence is None
    assert "could not be validated" in result.explanation
