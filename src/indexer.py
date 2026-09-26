import os
import sys
import pickle
import logging
from typing import List, Tuple
from src.bm25 import BM25
from tqdm import tqdm
import secrets
from dotenv import set_key
from hashlib import sha512
from src.models import MinimalSource
from src.chunker import chunk_markdown, chunk_python
from src.tokenizer import tokenize


logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(
        self,
        raw_dir: str,
        processed_dir: str,
        max_chunk_size: int = 2000,
    ) -> None:
        """Initializes the IndexBuilder.

        Args:
            raw_dir: The directory containing the raw files to be indexed.
            processed_dir: The directory where the indexing will be stored.
            max_chunk_size: The maximum size of each chunk.
        """
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.max_chunk_size = max_chunk_size

    def _collect_files(self) -> List[str]:
        """Collects all files from the raw directory.

        Returns:
            A list of file paths.
        """
        all_files: List[str] = []
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files

    def _read_text_file(self, file_path: str) -> str | None:
        """Reads the content of a text file.

        Args:
            file_path: The path of the file to be read.

        Returns:
            The content of the file as a string, or None if an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.info(f"Error reading file {file_path}: {e}")
            return None

    def _chunk_file(
        self,
        file_path: str,
        text: str,
    ) -> List[Tuple[MinimalSource, str]]:
        """Chunks a file based on its type (Markdown or Python).

        Args:
            file_path: The path of the file to be chunked.
            text: The content of the file.

        Returns:
            A list of tuples, each containing a MinimalSource object
            and the corresponding chunk.
        """
        if file_path.endswith('.py'):
            return chunk_python(file_path, text, self.max_chunk_size)
        return chunk_markdown(file_path, text, self.max_chunk_size)

    def _build_corpus(self) -> Tuple[List[List[str]], List[MinimalSource]]:
        """Builds the corpus for BM25 indexing.

        Returns:
            A tuple containing:
            - A list of tokenized documents (corpus).
            - A list of MinimalSource objects corresponding to valid chunks.
        """
        corpus_tokens: List[List[str]] = []
        valid_chunks: List[MinimalSource] = []

        for file_path in tqdm(
            self._collect_files(),
            desc="Chunking and Tokenizing",
        ):
            text = self._read_text_file(file_path)
            if text is None:
                continue

            for chunk, chunk_text in self._chunk_file(file_path, text):
                tokens = tokenize(chunk_text)
                if tokens:
                    corpus_tokens.append(tokens)
                    valid_chunks.append(chunk)

        return corpus_tokens, valid_chunks

    def _persist(self, bm25: BM25, valid_chunks: List[MinimalSource]) -> None:
        """Persists the BM25 index and chunk metadata to disk.

        Args:
            bm25: The BM25 index to be persisted.
            valid_chunks: The list of MinimalSource objects.
        """
        bm25_bytes = pickle.dumps(bm25)
        chunks_bytes = pickle.dumps(valid_chunks)
        key = secrets.token_bytes(32)
        hash = sha512(key + bm25_bytes + chunks_bytes).hexdigest()
        set_key('.env', 'KEY', key.hex())
        set_key('.env', 'HASH', hash)

        os.makedirs(self.processed_dir, exist_ok=True)
        with (open(os.path.join(self.processed_dir,
                                'bm25_index.pkl'), 'wb') as f,
              open(os.path.join(self.processed_dir,
                                'chunks_metadata.pkl'), 'wb') as g):
            f.write(bm25_bytes)
            g.write(chunks_bytes)

    def build(self) -> None:
        """Builds the BM25 index and persists it to disk."""

        corpus_tokens, valid_chunks = self._build_corpus()

        print("Building BM25 index...")
        bm25 = BM25(corpus_tokens)

        try:
            self._persist(bm25, valid_chunks)
        except Exception as e:
            logger.error(f"Error persisting index: {e}")
            sys.exit(1)

        print(
            "Ingestion complete! Indexed "
            f"{len(valid_chunks)} chunks under {self.processed_dir}"
        )


def build_index(
    raw_dir: str,
    processed_dir: str,
    max_chunk_size: int = 2000,
) -> None:
    """Builds the BM25 index from raw files.

    Args:
        raw_dir: The directory containing the raw files to be indexed.
        processed_dir: The directory where the indexing will be stored.
        max_chunk_size: The maximum size of each chunk.
    """
    IndexBuilder(raw_dir, processed_dir, max_chunk_size).build()
