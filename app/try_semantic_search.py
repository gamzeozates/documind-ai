
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.pdf_reader import extract_pdf_pages
from app.text_chunker import chunk_pages


def main() -> None:
    pdf_path = Path("data/sample_document.pdf")

    pages = extract_pdf_pages(str(pdf_path))

    chunks = chunk_pages(
        pages,
        chunk_size=800,
        chunk_overlap=100,
    )

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
    )

    chunk_texts = [chunk["text"] for chunk in chunks]

    chunk_embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True,
    )

    question = "Gençlerin ekonomik sorunları nelerdir?"

    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    scores = chunk_embeddings @ question_embedding

    ranked_indices = scores.argsort()[::-1]

    top_k = min(3, len(chunks))

    print(f"Question: {question}")
    print(f"Total chunks: {len(chunks)}")
    print("-" * 60)

    for rank, index in enumerate(ranked_indices[:top_k], start=1):
        chunk = chunks[int(index)]
        score = float(scores[index])

        print(f"Rank: {rank}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Document: {chunk['document_name']}")
        print(f"Page: {chunk['page_number']}")
        print(f"Similarity: {score:.4f}")
        print(f"Text: {chunk['text'][:300]}")
        print("-" * 60)


if __name__ == "__main__":
    main()
