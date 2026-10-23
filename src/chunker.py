from typing import Any
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import MinimalSource
import logging
import re


logger = logging.getLogger(__name__)

def split_by_length(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks with a sliding overlap.

    The chunk boundaries are defined by character offsets rather than semantic
    separators, which keeps the implementation simple and deterministic.

    Args:
        text: The source text to split.
        chunk_size: Maximum number of characters in each chunk.
        overlap: Number of characters to repeat between adjacent chunks.

    Returns:
        A list of string chunks covering the full text.

    Raises:
        ValueError: If the chunk size is invalid or if overlap is not smaller
            than the chunk size.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size to avoid infinite loops.")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

        start += step

    return chunks


def merge_splits(splits: list[str], chunk_size: int) -> list[str]:
    """Merge adjacent string fragments until they exceed the configured size.

    This helper preserves the order of the input fragments while concatenating
    them into larger chunks that stay within the provided maximum length.

    Args:
        splits: Source fragments to merge.
        chunk_size: Maximum allowed size for each merged chunk.

    Returns:
        A list of merged string chunks.
    """
    if not splits:
        return []

    merged_chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for split in splits:
        added_length = len(split)

        if current_length + added_length <= chunk_size:
            current_chunk.append(split)
            current_length += added_length
        else:
            if current_chunk:
                merged_chunks.append("".join(current_chunk))

            current_chunk = [split]
            current_length = len(split)

    if current_chunk:
        merged_chunks.append("".join(current_chunk))

    return merged_chunks



def split_by_separators(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Split text recursively on a prioritized list of separator patterns.

    The function tries the first separator, then recursively splits any resulting
    segment that is still too large using the remaining separators. The final
    result is merged back to respect the maximum chunk size.

    Args:
        text: The text to split.
        separators: Ordered regex-style separators to try from highest to lowest
            priority.
        chunk_size: Maximum size of a returned chunk.

    Returns:
        A list of text chunks created by recursive segmentation.
    """
    if not separators:
        return split_by_length(text, chunk_size, 0)
    splits: list[str] = re.split(f"({separators[0]})", text)
    ret: list[str] = []
    for split in splits:
        if len(split) < chunk_size:
            ret.append(split)
            continue
        ret.extend(split_by_separators(split, separators[1:], chunk_size))
    return merge_splits(ret, chunk_size)


def _is_indexable_file(file_path: str) -> bool:
    """Determine if a file is indexable based on its extension.

    Args:
        file_path: The path of the file to check.

    Returns:
        True if the file is indexable, False otherwise.
    """
    return (
        file_path.endswith('.py')
        or file_path.endswith('.md')
        or file_path.endswith('.txt')
    )


@dataclass(frozen=True)
class ChunkedDocument:
    """A tuple containing a MinimalSource
    and its corresponding chunk of text."""
    source: MinimalSource
    text: str


def _process_splits(
    splits: list[str],
    file_path: str,
    text: str,
) -> List[ChunkedDocument]:
    """Map text splits back to their absolute character offsets in the source.

    Each returned chunk is paired with a MinimalSource record that captures the
    original file path and the first/last character indexes in the source text.

    Args:
        splits: The already-generated chunks for the file.
        file_path: The path of the file being processed.
        text: The original full text from which the chunks were derived.

    Returns:
        A list of ChunkedDocument entries containing the source metadata and the
        corresponding text chunk.
    """
    chunks: List[ChunkedDocument] = []
    current_offset = 0

    for chunk in splits:
        start_idx = text.find(chunk, current_offset)

        if start_idx == -1:
            start_idx = text.find(chunk)

        end_idx = start_idx + len(chunk)

        chunks.append(
            ChunkedDocument(
                source=MinimalSource(
                    file_path=file_path,
                    first_character_index=start_idx,
                    last_character_index=end_idx,
                ),
                text=chunk,
            )
        )

        current_offset = end_idx

    return chunks


def _hash_text(text: str) -> str:
    """Compute a SHA-256 hash of the given text.

    Args:
        text: The text to be hashed.

    Returns:
        A hexadecimal string representing the SHA-256 hash of the text.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def chunk_markdown(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> List[ChunkedDocument]:
    """Chunk Markdown text using a simple fixed-size sliding window.

    This implementation does not perform semantic markdown-aware splitting; it
    instead creates fixed-size chunks with a small overlap to retain context.

    Args:
        file_path: The path of the Markdown file being processed.
        text: The original Markdown text to be chunked.
        max_chunk_size: The maximum size of each chunk.

    Returns:
        A list of ChunkedDocument records for the generated chunks.
    """

    splits = split_by_length(text, max_chunk_size, (max_chunk_size * 15) // 100)
    return _process_splits(splits, file_path, text)


def chunk_python(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> List[ChunkedDocument]:
    """Chunk Python source code by structural boundaries when possible.

    The splitter prefers class and function definitions, then falls back to blank
    lines, newline boundaries, and word boundaries to keep chunks readable while
    respecting the configured maximum size.

    Args:
        file_path: The path of the Python file being processed.
        text: The original Python source code to be chunked.
        max_chunk_size: The maximum size of each chunk.

    Returns:
        A list of ChunkedDocument records for the generated chunks.
    """

    separators = ["\n[ \t]*class ", "\n[ \t]*def ", "\n\n", "\n", " "]
    splits = split_by_separators(text, separators, max_chunk_size)
    return _process_splits(splits, file_path, text)


def chunk_file(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> List[ChunkedDocument]:
    """Chunk a file based on its extension.

    Supported file types are Python, Markdown, and plain text. Unsupported file
    types return an empty list.

    Args:
        file_path: The file path to inspect.
        text: The full file content to be chunked.
        max_chunk_size: The maximum size of each generated chunk.

    Returns:
        A list of ChunkedDocument entries for the file, or an empty list for
        unsupported file types.
    """
    if file_path.endswith('.py'):
        return chunk_python(file_path, text, max_chunk_size)
    if file_path.endswith('.md') or file_path.endswith('.txt'):
        return chunk_markdown(file_path, text, max_chunk_size)
    return []


def collect_chunk_delta(
    raw_dir: str,
    old_manifest: Dict[str, str] | None = None,
    max_chunk_size: int = 2000,
) -> tuple[list[str], list[ChunkedDocument], Dict[str, str]]:
    """Collect chunk deltas for files under ``raw_dir``.

    Args:
        raw_dir: The directory containing the raw files to be indexed.
        old_manifest: The previous manifest mapping file paths to their hashes.
        max_chunk_size: The maximum size of each chunk.

    Returns:
        A tuple containing the list of deleted file paths,
        the list of added chunks, and the new manifest.
    """

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    previous_manifest = old_manifest or {}
    new_manifest: Dict[str, str] = {}
    added_chunks: List[ChunkedDocument] = []

    for root, _, files in os.walk(raw_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            if not _is_indexable_file(file_path):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as file_handle:
                    text = file_handle.read()
            except Exception as exc:
                logger.debug(f"Error reading file {file_path}: {exc}")

            file_hash = _hash_text(text)
            new_manifest[file_path] = file_hash

            if previous_manifest.get(file_path) != file_hash:
                added_chunks.extend(
                    chunk_file(file_path, text, max_chunk_size)
                )

    deleted_file_paths = [
        file_path
        for file_path, file_hash in previous_manifest.items()
        if new_manifest.get(file_path) != file_hash
    ]

    return deleted_file_paths, added_chunks, new_manifest
