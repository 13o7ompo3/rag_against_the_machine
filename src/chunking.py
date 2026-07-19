from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from src.models import MinimalSource
import logging
import os


logger = logging.getLogger(__name__)


def _process_splits(splitter: RecursiveCharacterTextSplitter, file_path: str, text: str) -> List[MinimalSource]:
    """
    Internal helper to map LangChain's string chunks back to their 
    absolute character indices in the original text.
    """
    splits = splitter.split_text(text)
    chunks = []
    current_offset = 0

    for chunk in splits:
        # Find the exact start index of the chunk in the original text
        start_idx = text.find(chunk, current_offset)

        if start_idx == -1:
            start_idx = text.find(chunk)

        end_idx = start_idx + len(chunk)

        chunks.append(MinimalSource(
            file_path=file_path,
            first_character_index=start_idx,
            last_character_index=end_idx
        ))

        # Advance the pointer to the end of the current chunk
        current_offset = end_idx

    return chunks


def chunk_markdown(file_path: str, text: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    """
    Wrapper for Markdown chunking using LangChain.
    Prioritizes splitting at headers, paragraphs, and lists.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=max_chunk_size,
        chunk_overlap=0
    )
    return _process_splits(splitter, file_path, text)


def chunk_python(file_path: str, text: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    """
    Wrapper for Python chunking using LangChain.
    Prioritizes splitting at classes and functions.
    """
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=max_chunk_size,
        chunk_overlap=0
    )
    return _process_splits(splitter, file_path, text)


def index_file(file_path: str, max_chunk_size: int = 2000) -> List[MinimalSource]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        os._exit(1)

    if file_path.endswith('.py'):
        return chunk_python(file_path, text, max_chunk_size)
    elif file_path.endswith('.md'):
        return chunk_markdown(file_path, text, max_chunk_size)
    else:
        # Fallback for unrecognized file types
        return chunk_markdown(file_path, text, max_chunk_size)
