
from pathlib import Path
from tempfile import TemporaryDirectory

from app.answer_evaluator import AnswerEvaluator
from app.document_utils import calculate_document_id
from app.embedding_service import EmbeddingService
from app.llm_service import LLMService
from app.pdf_reader import extract_pdf_pages
from app.qdrant_store import QdrantStore
from app.rag_service import RAGService
from app.text_chunker import chunk_pages


def main() -> None:
    pdf_path = Path("data/sample_document.pdf")

    pages = extract_pdf_pages(pdf_path)
    chunks = chunk_pages(pages)

    document_id = calculate_document_id(pdf_path)

    embedding_service = EmbeddingService()
    embeddings = embedding_service.encode_texts(
        [chunk["text"] for chunk in chunks]
    )

    llm_service = LLMService()

    try:
        with TemporaryDirectory() as temp_dir:
            qdrant_store = QdrantStore(
                storage_path=temp_dir,
            )

            try:
                qdrant_store.upsert_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    document_id=document_id,
                )

                evaluator = AnswerEvaluator(
                    llm_service=llm_service,
                )

                rag_service = RAGService(
                    embedding_service=embedding_service,
                    qdrant_store=qdrant_store,
                    llm_service=llm_service,
                    answer_evaluator=evaluator,
                )

                result = rag_service.ask(
                    question=(
                        "Gençlerin ekonomik sorunları nelerdir?"
                    ),
                    top_k=3,
                )

                print("\nQUESTION")
                print(result["question"])

                print("\nANSWER")
                print(result["answer"])

                print("\nSOURCES")
                for source in result["sources"]:
                    print(
                        f"[{source['id']}] "
                        f"{source['document_name']} "
                        f"- Page {source['page_number']} "
                        f"- Similarity: {source['similarity']:.4f}"
                    )

                print("\nEVALUATION")
                evaluation = result.get("evaluation")

                if evaluation is None:
                    print("Evaluation was not performed.")
                else:
                    print("Status:", evaluation["status"])
                    print(
                        "Is supported:",
                        evaluation["is_supported"],
                    )
                    print(
                        "Explanation:",
                        evaluation["explanation"],
                    )

            finally:
                qdrant_store.close()

    finally:
        llm_service.close()


if __name__ == "__main__":
    main()
