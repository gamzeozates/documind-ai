
from pathlib import Path
from tempfile import TemporaryDirectory

from app.document_utils import calculate_document_id
from app.embedding_service import EmbeddingService
from app.llm_service import LLMService
from app.pdf_reader import extract_pdf_pages
from app.qdrant_store import QdrantStore
from app.rag_service import RAGService
from app.text_chunker import chunk_pages


def main() -> None:
    pdf_path = Path("data/sample_document.pdf")
    document_id = calculate_document_id(pdf_path)

    pages = extract_pdf_pages(str(pdf_path))
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No text found in the PDF")

    print("Document:", pdf_path.name)
    print("Pages:", len(pages))
    print("Chunks:", len(chunks))

    embedding_service = EmbeddingService()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_service.encode_texts(texts)

    print("Embedding shape:", embeddings.shape)

    # Use a temporary database to protect existing data.
    with TemporaryDirectory() as temp_dir:
        store = QdrantStore(
            storage_path=temp_dir,
            collection_name="rag_demo",
            vector_size=embeddings.shape[1],
        )

        llm_service = None

        try:
            store.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                document_id=document_id,
            )

            print("Stored points:", store.count())

            llm_service = LLMService()

            rag_service = RAGService(
                embedding_service=embedding_service,
                qdrant_store=store,
                llm_service=llm_service,
            )


            question = (
    "Mars gezegeninde insanların yaşayabilmesi için "
    "hangi teknolojiler gereklidir?"
    )


            print("Similarity threshold:", 0.40)


            result = rag_service.ask(
                question=question,
                top_k=3,
                document_id=document_id,
                min_similarity=0.40,
            )

            print("\nQUESTION")
            print(result["question"])

            print("\nANSWER")
            print(result["answer"])

            print("\nSOURCES")

            for source in result["sources"]:
                print(
                    f"[{source['id']}] "
                    f"{source['document_name']} | "
                    f"Page {source['page_number']} | "
                    f"Chunk {source['chunk_id']} | "
                    f"Similarity {source['similarity']:.4f}"
                )
        finally:
            if llm_service is not None:
                llm_service.close()

            store.close()


if __name__ == "__main__":
    main()
