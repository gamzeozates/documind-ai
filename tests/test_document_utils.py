
import hashlib

import pytest

from app.document_utils import calculate_document_id


def test_same_content_produces_same_id(tmp_path):
    first_file = tmp_path / "first.pdf"
    second_file = tmp_path / "second.pdf"

    first_file.write_bytes(b"Same PDF content")
    second_file.write_bytes(b"Same PDF content")

    first_id = calculate_document_id(first_file)
    second_id = calculate_document_id(second_file)

    assert first_id == second_id
    assert len(first_id) == 64


def test_different_content_produces_different_ids(tmp_path):
    first_file = tmp_path / "first.pdf"
    second_file = tmp_path / "second.pdf"

    first_file.write_bytes(b"First document")
    second_file.write_bytes(b"Second document")

    assert calculate_document_id(first_file) != calculate_document_id(
        second_file
    )


def test_document_id_matches_sha256(tmp_path):
    file_path = tmp_path / "example.pdf"
    content = b"DocuMind AI document"

    file_path.write_bytes(content)

    expected_id = hashlib.sha256(content).hexdigest()

    assert calculate_document_id(file_path) == expected_id


def test_missing_document_raises_error(tmp_path):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        calculate_document_id(missing_file)
