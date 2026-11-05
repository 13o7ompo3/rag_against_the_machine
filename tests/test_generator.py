import os
import tempfile
import pytest
from src.models import MinimalSource
from src.generator import read_chunk, extract_chunk_text


def test_read_chunk():
    sample_text = "0123456789ABCDEF"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(sample_text)
        file_path = f.name

    try:
        content = read_chunk(file_path, 2, 7)
        assert content == "23456"
    finally:
        os.remove(file_path)


def test_extract_chunk_text():
    sample_text = "0123456789ABCDEF"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        f.write(sample_text)
        file_path = f.name

    try:
        source = MinimalSource(
            file_path=file_path,
            first_character_index=5,
            last_character_index=10,
        )
        chunk = extract_chunk_text(source)
        assert chunk == "56789"
    finally:
        os.remove(file_path)


def test_extract_chunk_text_missing_file():
    source = MinimalSource(
        file_path="non_existent_file.txt",
        first_character_index=0,
        last_character_index=10,
    )
    # Should catch exception and return empty string without raising
    chunk = extract_chunk_text(source)
    assert chunk == ""
