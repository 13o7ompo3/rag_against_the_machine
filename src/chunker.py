from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.models import MinimalSource
import logging


logger = logging.getLogger(__name__)


def _process_splits(splitter: RecursiveCharacterTextSplitter, file_path: str,
                    text: str) -> List[Tuple[MinimalSource, str]]:
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
    chunks = []
    current_offset = 0

    for chunk in splits:
        start_idx = text.find(chunk, current_offset)

        if start_idx == -1:
            start_idx = text.find(chunk)

        end_idx = start_idx + len(chunk)

        chunks.append((MinimalSource(
            file_path=file_path,
            first_character_index=start_idx,
            last_character_index=end_idx
        ), chunk))

        current_offset = end_idx

    return chunks


def chunk_markdown(file_path: str, text: str, max_chunk_size: int = 2000
                   ) -> List[Tuple[MinimalSource, str]]:
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


def chunk_python(file_path: str, text: str, max_chunk_size: int = 2000
                 ) -> List[Tuple[MinimalSource, str]]:
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
