
from app.answer_evaluator import AnswerEvaluator
from app.llm_service import LLMService


def main() -> None:
    sources = [
        {
            "id": 1,
            "text": "Gençler ekonomik sorunlar yaşamaktadır.",
        }
    ]

    examples = [
        (
            "SUPPORTED EXAMPLE",
            "Gençler ekonomik güçlüklerle karşılaşmaktadır [1].",
        ),
        (
            "UNSUPPORTED EXAMPLE",
            "Gençlerin %80'i işsizdir [1].",
        ),
    ]

    llm_service = LLMService()

    try:
        evaluator = AnswerEvaluator(
            llm_service=llm_service,
        )

        for label, answer in examples:
            result = evaluator.evaluate(
                answer=answer,
                sources=sources,
            )

            print(f"\n{label}")
            print("Answer:", answer)
            print("Status:", result.status.value)
            print("Explanation:", result.explanation)

    finally:
        llm_service.close()


if __name__ == "__main__":
    main()
