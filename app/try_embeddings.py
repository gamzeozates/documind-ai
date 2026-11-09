
from sentence_transformers import SentenceTransformer


def main() -> None:
    model_name = (
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("Loading embedding model...")

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

    print(f"Number of sentences: {len(sentences)}")
    print(f"Embedding shape: {embeddings.shape}")

    for index, sentence in enumerate(sentences):
        print("-" * 50)
        print(f"Sentence {index + 1}: {sentence}")
        print(f"First 5 values: {embeddings[index][:5]}")


if __name__ == "__main__":
    main()
