import os
from dataclasses import dataclass
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.models import MinimalSource
import logging


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkedDocument:
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
        chunk_overlap=max_chunk_size // 10
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
        chunk_overlap=max_chunk_size // 10
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


def collect_chunked_documents(
    raw_dir: str,
    max_chunk_size: int = 2000,
) -> List[ChunkedDocument]:
    documents: List[ChunkedDocument] = []

    for root, _, files in os.walk(raw_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8') as file_handle:
                    text = file_handle.read()
            except Exception as exc:
                logger.info(f"Error reading file {file_path}: {exc}")
                continue

            documents.extend(chunk_file(file_path, text, max_chunk_size))

    return documents
