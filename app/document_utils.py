
import hashlib
from pathlib import Path


def calculate_document_id(file_path: str | Path) -> str:
    """Calculate a SHA-256 identifier from file contents."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()
