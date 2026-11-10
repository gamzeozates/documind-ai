
import os

from google import genai


def main() -> None:
    """Test the connection to the Gemini API."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is missing."
        )

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=(
                "Türkçe olarak tek cümleyle cevap ver: "
                "RAG (Retrieval-Augmented Generation) nedir?"
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        print("Model: gemini-3.5-flash-lite")
        print("Response:", response.text)

    finally:
        client.close()


if __name__ == "__main__":
    main()
