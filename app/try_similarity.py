
from sentence_transformers import SentenceTransformer


def main() -> None:
    model_name = (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    model = SentenceTransformer(model_name, device="cpu")

    sentences = [
        "Üniversite mezunları iş bulmakta zorlanıyor.",
        "Gençlerin istihdam sorunları giderek artıyor.",
        "Bugün hava oldukça güneşli.",
    ]

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
    )

    similarity_matrix = embeddings @ embeddings.T

    print("Cosine Similarity Matrix:")
    print(similarity_matrix.round(3))

    print("\nPairwise similarities:")

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = float(similarity_matrix[i, j])

            print("-" * 50)
            print(f"Sentence {i + 1} vs Sentence {j + 1}")
            print(f"Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
