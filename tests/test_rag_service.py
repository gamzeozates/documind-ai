
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.answer_evaluator import EvaluationResult, EvaluationStatus
from app.rag_service import RAGService


@pytest.fixture
def rag_service():
    embedding_service = MagicMock()
    embedding_service.encode_query.return_value = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    qdrant_store = MagicMock()
    llm_service = MagicMock()

    return RAGService(
        embedding_service=embedding_service,
        qdrant_store=qdrant_store,
        llm_service=llm_service,
    )


def test_ask_returns_answer_with_sources(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk_id": 1,
            "text": "Gençlerin ekonomik sorunları vardır.",
            "similarity": 0.85,
        }
    ]

    rag_service.llm_service.generate.return_value = (
        "Ekonomik sorunlar yaşanmaktadır. [1]"
    )

    result = rag_service.ask(
        "Gençlerin sorunları nelerdir?",
        document_id="document-001",
    )

    assert result["answer"] == "Ekonomik sorunlar yaşanmaktadır. [1]"
    assert result["question"] == "Gençlerin sorunları nelerdir?"

    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_name"] == "guide.pdf"
    assert result["sources"][0]["page_number"] == 2

    rag_service.qdrant_store.search.assert_called_once_with(
        query_vector=rag_service.embedding_service.encode_query.return_value,
        top_k=3,
        document_id="document-001",
    )

    prompt = rag_service.llm_service.generate.call_args.args[0]

    assert "Gençlerin ekonomik sorunları vardır." in prompt
    assert "Page: 2" in prompt
    assert "Cite supporting passages" in prompt


def test_ask_returns_message_when_no_results(rag_service):
    rag_service.qdrant_store.search.return_value = []

    result = rag_service.ask("Belgelerde ne yazıyor?")

    assert result["sources"] == []
    assert "ilgili bilgi bulunamadı" in result["answer"]

    rag_service.llm_service.generate.assert_not_called()


def test_ask_rejects_empty_question(rag_service):
    with pytest.raises(ValueError, match="Question"):
        rag_service.ask("   ")

    rag_service.embedding_service.encode_query.assert_not_called()


def test_ask_rejects_invalid_top_k(rag_service):
    with pytest.raises(ValueError, match="top_k"):
        rag_service.ask("RAG nedir?", top_k=0)

    rag_service.qdrant_store.search.assert_not_called()


def test_ask_skips_llm_for_low_similarity(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Unrelated content",
            "similarity": 0.22,
        }
    ]

    result = rag_service.ask("Mars'ta yaşam mümkün mü?")

    assert result["sources"] == []
    assert "ilgili bilgi bulunamadı" in result["answer"]

    rag_service.llm_service.generate.assert_not_called()


def test_ask_uses_high_similarity_sources(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk_id": 1,
            "text": "Gençlerin ekonomik sorunları vardır.",
            "similarity": 0.72,
        },
        {
            "document_name": "guide.pdf",
            "page_number": 3,
            "chunk_id": 2,
            "text": "Unrelated content",
            "similarity": 0.15,
        },
    ]

    rag_service.llm_service.generate.return_value = (
        "Gençler ekonomik sorunlar yaşamaktadır. [1]"
    )

    result = rag_service.ask("Gençlerin sorunları nelerdir?")

    assert result["answer"] == (
        "Gençler ekonomik sorunlar yaşamaktadır. [1]"
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["page_number"] == 2

    prompt = rag_service.llm_service.generate.call_args.args[0]

    assert "Gençlerin ekonomik sorunları vardır." in prompt
    assert "Unrelated content" not in prompt

    rag_service.llm_service.generate.assert_called_once()


def test_ask_rejects_invalid_llm_citation(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk_id": 1,
            "text": "Gençlerin ekonomik sorunları vardır.",
            "similarity": 0.85,
        }
    ]

    rag_service.llm_service.generate.return_value = (
        "Gençlerin ekonomik sorunları bulunmaktadır [99]."
    )

    result = rag_service.ask(
        question="Gençlerin sorunları nelerdir?",
    )

    assert result["question"] == "Gençlerin sorunları nelerdir?"

    assert result["answer"] == (
        "Oluşturulan cevabın kaynak numaraları "
        "doğrulanamadı. Lütfen soruyu yeniden deneyin."
    )

    assert result["sources"] == []

    assert "[99]" not in result["answer"]

    rag_service.llm_service.generate.assert_called_once()


def test_ask_rejects_answer_without_citations(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk_id": 1,
            "text": "Gençlerin ekonomik sorunları vardır.",
            "similarity": 0.85,
        }
    ]

    rag_service.llm_service.generate.return_value = (
        "Gençlerin ekonomik sorunları bulunmaktadır."
    )

    result = rag_service.ask(
        question="Gençlerin sorunları nelerdir?",
    )

    assert result["question"] == "Gençlerin sorunları nelerdir?"

    assert result["answer"] == (
        "Oluşturulan cevabın kaynak numaraları "
        "doğrulanamadı. Lütfen soruyu yeniden deneyin."
    )

    assert result["sources"] == []

    rag_service.llm_service.generate.assert_called_once()


def test_ask_includes_source_text(rag_service):
    source_text = "Gençler ekonomik sorunlar yaşamaktadır."

    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 3,
            "chunk_id": 2,
            "text": source_text,
            "similarity": 0.85,
        }
    ]

    rag_service.llm_service.generate.return_value = (
        "Gençler ekonomik sorunlar yaşamaktadır [1]."
    )

    result = rag_service.ask(
        "Gençlerin sorunları nelerdir?"
    )

    assert len(result["sources"]) == 1

    source = result["sources"][0]

    assert source["id"] == 1
    assert source["document_name"] == "guide.pdf"
    assert source["page_number"] == 3
    assert source["chunk_id"] == 2
    assert source["text"] == source_text



def test_ask_includes_supported_evaluation(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk_id": 1,
            "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            "similarity": 0.85,
        }
    ]

    rag_service.llm_service.generate.return_value = (
        "Gençler ekonomik güçlüklerle karşılaşmaktadır [1]."
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        status=EvaluationStatus.SUPPORTED,
        explanation="Cevap kaynakla uyumludur.",
        confidence=None,
    )

    rag_service.answer_evaluator = evaluator

    result = rag_service.ask("Gençlerin sorunları nelerdir?")

    assert result["evaluation"]["status"] == "supported"
    assert result["evaluation"]["is_supported"] is True
    assert result["evaluation"]["confidence"] is None
    assert result["answer"].endswith("[1].")

    evaluator.evaluate.assert_called_once_with(
        answer=result["answer"],
        sources=result["sources"],
    )


def test_ask_includes_unsupported_evaluation(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            "similarity": 0.85,
        }
    ]

    rag_service.llm_service.generate.return_value = (
        "Gençlerin %80'i işsizdir [1]."
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        status=EvaluationStatus.UNSUPPORTED,
        explanation="Kaynakta %80 işsizlik bilgisi yok.",
        confidence=None,
    )

    rag_service.answer_evaluator = evaluator

    result = rag_service.ask("İşsizlik oranı nedir?")

    assert result["evaluation"]["status"] == "unsupported"
    assert result["evaluation"]["is_supported"] is False
    assert "işsizlik bilgisi yok" in (
        result["evaluation"]["explanation"]
    )


def test_ask_without_evaluator_preserves_response(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            "similarity": 0.85,
        }
    ]

    rag_service.llm_service.generate.return_value = (
        "Gençler ekonomik sorunlar yaşamaktadır [1]."
    )

    result = rag_service.ask("Gençlerin sorunları nelerdir?")

    assert "evaluation" not in result
    assert len(result["sources"]) == 1
    assert result["answer"].endswith("[1].")


def test_ask_blocks_unsupported_answer(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            "similarity": 0.85,
        }
    ]

    generated_answer = "Gençlerin %80'i işsizdir [1]."

    rag_service.llm_service.generate.return_value = (
        generated_answer
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        status=EvaluationStatus.UNSUPPORTED,
        explanation="Kaynakta %80 işsizlik oranı bulunmuyor.",
        confidence=None,
    )

    rag_service.answer_evaluator = evaluator

    result = rag_service.ask("Gençlerin işsizlik oranı nedir?")

    assert result["evaluation"]["status"] == "unsupported"
    assert result["evaluation"]["is_supported"] is False

    assert result["answer"] == (
        "Oluşturulan cevap kaynaklar tarafından "
        "yeterince doğrulanamadığı için "
        "gösterilemiyor. Lütfen sorunuzu "
        "farklı şekilde ifade edin."
    )

    assert generated_answer not in result["answer"]
    assert len(result["sources"]) == 1

    evaluator.evaluate.assert_called_once_with(
        answer=generated_answer,
        sources=result["sources"],
    )


def test_ask_blocks_not_evaluated_answer(rag_service):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 2,
            "chunk_id": 3,
            "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            "similarity": 0.82,
        }
    ]

    generated_answer = (
        "Gençler ekonomik güçlüklerle karşılaşmaktadır [1]."
    )

    rag_service.llm_service.generate.return_value = (
        generated_answer
    )

    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(
        status=EvaluationStatus.NOT_EVALUATED,
        explanation="Evaluation response could not be validated.",
        confidence=None,
    )

    rag_service.answer_evaluator = evaluator

    result = rag_service.ask("Gençlerin sorunları nelerdir?")

    assert result["evaluation"]["status"] == "not_evaluated"
    assert result["evaluation"]["is_supported"] is False

    assert result["answer"] == (
        "Oluşturulan cevap kaynaklar tarafından "
        "yeterince doğrulanamadığı için "
        "gösterilemiyor. Lütfen sorunuzu "
        "farklı şekilde ifade edin."
    )

    assert generated_answer not in result["answer"]
    assert len(result["sources"]) == 1

    evaluator.evaluate.assert_called_once()


def test_ask_handles_evaluator_api_error(
    rag_service,
    caplog,
):
    rag_service.qdrant_store.search.return_value = [
        {
            "document_name": "guide.pdf",
            "page_number": 1,
            "chunk_id": 1,
            "text": "Gençler ekonomik sorunlar yaşamaktadır.",
            "similarity": 0.85,
        }
    ]

    generated_answer = (
        "Gençler ekonomik sorunlar yaşamaktadır [1]."
    )

    rag_service.llm_service.generate.return_value = (
        generated_answer
    )

    evaluator = MagicMock()
    evaluator.evaluate.side_effect = RuntimeError(
        "Simulated Gemini API connection failure"
    )

    rag_service.answer_evaluator = evaluator

    result = rag_service.ask(
        "Gençlerin ekonomik sorunları nelerdir?"
    )

    assert result["evaluation"]["status"] == "not_evaluated"
    assert result["evaluation"]["is_supported"] is False
    assert result["evaluation"]["confidence"] is None

    assert result["answer"] == (
        "Cevap değerlendirme işlemi "
        "tamamlanamadığı için oluşturulan "
        "cevap gösterilemiyor. "
        "Lütfen daha sonra tekrar deneyin."
    )

    assert generated_answer not in result["answer"]
    assert len(result["sources"]) == 1

    evaluator.evaluate.assert_called_once_with(
        answer=generated_answer,
        sources=result["sources"],
    )

    assert "Answer evaluation failed." in caplog.text
    assert (
        "Simulated Gemini API connection failure"
        in caplog.text
    )
