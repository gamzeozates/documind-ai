
import os

from dotenv import load_dotenv
from google import genai

DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"

load_dotenv()


class LLMService:
    """Generate text using the Gemini API."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")

        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not configured"
            )

        self.model_name = model_name
        self.client = genai.Client(api_key=resolved_key)

    def generate(self, prompt: str) -> str:
        """Generate a text response from a prompt."""

        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        answer = response.text

        if not answer or not answer.strip():
            raise RuntimeError(
                "LLM returned an empty response"
            )

        return answer.strip()

    def close(self) -> None:
        """Release API client resources."""
        self.client.close()
