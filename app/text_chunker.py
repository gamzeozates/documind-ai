def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    """Split PDF pages into chunks while respecting word boundaries."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size")

    chunks = []

    for page in pages:
        text = " ".join(page["text"].split())

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            if end < len(text):
                last_space = text.rfind(" ", start, end + 1)

                if last_space > start:
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "document_name": page["document_name"],
                        "page_number": page["page_number"],
                        "chunk_id": len(chunks) + 1,
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            next_start = max(0, end - chunk_overlap)

            if next_start > 0:
                previous_space = text.rfind(" ", 0, next_start + 1)

                if previous_space != -1:
                    next_start = previous_space + 1

            if next_start <= start:
                next_start = end

            start = next_start

    return chunks
