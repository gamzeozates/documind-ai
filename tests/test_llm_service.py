
from unittest.mock import MagicMock, patch

import pytest

from app.llm_service import LLMService


def test_generate_returns_response():
    with patch("app.llm_service.genai.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.text = "RAG, kaynaklı cevap üretir."

        mock_client.return_value.models.generate_content.return_value = (
            mock_response
        )

        service = LLMService(api_key="test-api-key")

        try:
            result = service.generate("RAG nedir?")

            assert result == "RAG, kaynaklı cevap üretir."
            mock_client.return_value.models.generate_content.assert_called_once()
        finally:
            service.close()


def test_generate_rejects_empty_prompt():
    with patch("app.llm_service.genai.Client"):
        service = LLMService(api_key="test-api-key")

        try:
            with pytest.raises(ValueError, match="Prompt"):
                service.generate("   ")
        finally:
            service.close()


def test_generate_rejects_empty_response():
    with patch("app.llm_service.genai.Client") as mock_client:
        mock_response = MagicMock()
        mock_response.text = ""

        mock_client.return_value.models.generate_content.return_value = (
            mock_response
        )

        service = LLMService(api_key="test-api-key")

        try:
            with pytest.raises(RuntimeError, match="empty response"):
                service.generate("RAG nedir?")
        finally:
            service.close()


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        LLMService()
