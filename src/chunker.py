import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from .models import MinimalSource
import logging


logger = logging.getLogger(__name__)


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
    splitter: RecursiveCharacterTextSplitter,
    file_path: str,
    text: str,
) -> List[ChunkedDocument]:
    """
    Internal helper to map LangChain's string chunks back to their
    absolute character indices in the original text.

    Args:
        splitter: An instance of RecursiveCharacterTextSplitter.
        file_path: The path of the file being processed.
        text: The original text to be chunked.

    Returns:
        A list of tuples, each containing a MinimalSource object
        and the corresponding chunk.
    """
    splits = splitter.split_text(text)
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
    """
    Wrapper for Markdown chunking using LangChain.
    Prioritizes splitting at headers, paragraphs, and lists.

    Args:
        file_path: The path of the Markdown file being processed.
        text: The original Markdown text to be chunked.
        max_chunk_size: The maximum size of each chunk.

    Returns:
        A list of tuples, each containing a MinimalSource object
        and the corresponding chunk.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=max_chunk_size,
        chunk_overlap=(max_chunk_size // 10) * 3,
        strip_whitespace=False,
    )
    return _process_splits(splitter, file_path, text)


def chunk_python(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> List[ChunkedDocument]:
    """
    Wrapper for Python chunking using LangChain.
    Prioritizes splitting at classes and functions.

    Args:
        file_path: The path of the Python file being processed.
        text: The original Python code to be chunked.
        max_chunk_size: The maximum size of each chunk.

    Returns:
        A list of tuples, each containing a MinimalSource object
        and the corresponding chunk.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=(max_chunk_size // 10) * 3,
        strip_whitespace=False,
    )
    return _process_splits(splitter, file_path, text)


def chunk_file(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> List[ChunkedDocument]:
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
                raise RuntimeError(
                    f"Error reading file {file_path}: {exc}"
                ) from exc

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
