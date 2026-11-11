import os
import pytest
from src.chunker import (
    _is_indexable_file,
    _hash_text,
    chunk_file,
    collect_chunk_delta,
)


def test_is_indexable_file():
    assert _is_indexable_file("script.py") is True
    assert _is_indexable_file("README.md") is True
    assert _is_indexable_file("notes.txt") is True
    assert _is_indexable_file("data.json") is False
    assert _is_indexable_file("table.csv") is False
    assert _is_indexable_file("binary.bin") is False


def test_hash_text():
    text1 = "Hello world"
    text2 = "Hello world"
    text3 = "Different text"

    assert _hash_text(text1) == _hash_text(text2)
    assert _hash_text(text1) != _hash_text(text3)


def test_chunk_file_character_indices():
    code = (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )
    chunks = chunk_file("calc.py", code, max_chunk_size=50)

    assert len(chunks) > 0
    for chunked_doc in chunks:
        source = chunked_doc.source
        # Verify indices are valid and non-negative
        assert source.first_character_index >= 0
        assert source.last_character_index > source.first_character_index
        # Verify the slice matches the chunk text
        extracted_text = code[source.first_character_index:source.last_character_index]
        assert chunked_doc.text == f"{extracted_text} {source.file_path}"


def test_collect_chunk_delta_missing_dir_raises():
    """Verify the bug fix: missing directory raises FileNotFoundError instead of silent wipe."""
    with pytest.raises(FileNotFoundError):
        collect_chunk_delta("non_existent_directory_12345")


def test_collect_chunk_delta_workflow(temp_raw_dir):
    # Initial indexing run
    deleted, added, manifest = collect_chunk_delta(temp_raw_dir, old_manifest={})

    assert len(deleted) == 0
    assert len(added) > 0
    # .py and .md should be in manifest; .csv should not be
    manifest_paths = [os.path.basename(p) for p in manifest.keys()]
    assert "test_code.py" in manifest_paths
    assert "readme.md" in manifest_paths
    assert "data.csv" not in manifest_paths

    # Second run without changes should yield zero additions and zero deletions
    deleted_2, added_2, manifest_2 = collect_chunk_delta(temp_raw_dir, old_manifest=manifest)
    assert len(deleted_2) == 0
    assert len(added_2) == 0
    assert manifest_2 == manifest

    # Delete a file and verify it appears in deleted_file_paths
    py_path = os.path.join(temp_raw_dir, "test_code.py")
    os.remove(py_path)

    deleted_3, added_3, manifest_3 = collect_chunk_delta(temp_raw_dir, old_manifest=manifest)
    assert py_path in deleted_3
    assert py_path not in manifest_3


def test_collect_chunk_delta_unreadable_file_does_not_crash(tmp_path):
    """Verify RAT-001: unreadable file is skipped without crashing."""
    unreadable = tmp_path / "locked.py"
    unreadable.write_text("x = 1")
    os.chmod(unreadable, 0o000)

    try:
        deleted, added, manifest = collect_chunk_delta(str(tmp_path))
        assert str(unreadable) not in manifest
    finally:
        os.chmod(unreadable, 0o644)


def test_chunk_markdown_repeated_text_offsets():
    """Verify RAT-002: sliding overlap does not snap back to offset 0."""
    from src.chunker import chunk_markdown
    text = "Heading A\n" + "line 1\n" * 200 + "Heading B\n" + "line 2\n" * 200
    chunks = chunk_markdown("test.md", text, max_chunk_size=1000)

    for chunk in chunks:
        src = chunk.source
        extracted = text[src.first_character_index:src.last_character_index]
        assert chunk.text == f"{extracted} {src.file_path}"


def test_split_by_length_returns_chunked_documents():
    """Verify split_by_length directly returns a list of ChunkedDocument."""
    from src.chunker import split_by_length, ChunkedDocument
    text = "0123456789"
    chunks = split_by_length("test.txt", text, chunk_size=5, overlap=2)

    assert len(chunks) == 3
    assert all(isinstance(c, ChunkedDocument) for c in chunks)
    assert chunks[0].source.first_character_index == 0
    assert chunks[0].source.last_character_index == 5
    assert chunks[0].text == "01234 test.txt"

    assert chunks[1].source.first_character_index == 3
    assert chunks[1].source.last_character_index == 8
    assert chunks[1].text == "34567 test.txt"

    assert chunks[2].source.first_character_index == 6
    assert chunks[2].source.last_character_index == 10
    assert chunks[2].text == "6789 test.txt"


