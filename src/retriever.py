import os
import hmac
import pickle
import secrets
from hashlib import sha512
from typing import List

import numpy as np
from dotenv import load_dotenv, set_key

from src.bm25 import BM25
from src.chunker import collect_chunked_documents
from src.models import MinimalSearchResults, MinimalSource
from src.tokenizer import tokenize
from src.search_index import SearchIndex


class BM25Index(SearchIndex):
    """BM25-backed search index with the same interface as semantic search."""

    def __init__(self, processed_dir: str | None = None):
        self.bm25: BM25 | None = None
        self.chunks: List[MinimalSource] = []
        self.processed_dir = processed_dir

        if processed_dir is not None:
            self.load(processed_dir)

    def build_from_raw(
        self,
        raw_dir: str,
        processed_dir: str,
        max_chunk_size: int = 2000,
    ) -> None:
        chunked_documents = collect_chunked_documents(raw_dir, max_chunk_size)

        corpus = []
        chunks: List[MinimalSource] = []
        for chunk in chunked_documents:
            tokens = tokenize(chunk.text)
            if tokens:
                corpus.append(tokens)
                chunks.append(chunk.source)

        self.bm25 = BM25(corpus)
        self.chunks = chunks
        self.processed_dir = processed_dir
        self.save(processed_dir)

    def save(self, processed_dir: str) -> None:
        if self.bm25 is None:
            raise RuntimeError("Cannot persist an empty index.")

        try:
            bm25_bytes = pickle.dumps(self.bm25)
            chunks_bytes = pickle.dumps(self.chunks)
        except Exception:
            raise RuntimeError("Failed to serialize BM25 index data.")

        try:
            key = secrets.token_bytes(32)
            integrity_hash = hmac.new(
                key,
                bm25_bytes + chunks_bytes,
                sha512,
            ).hexdigest()
            set_key('.env', 'KEY1', key.hex())
            set_key('.env', 'HASH1', integrity_hash)
        except Exception:
            raise RuntimeError("Failed to write BM25 integrity metadata.")

        try:
            os.makedirs(processed_dir, exist_ok=True)
            bm25_path = os.path.join(processed_dir, 'bm25_index.pkl')
            chunks_path = os.path.join(processed_dir, 'chunks_metadata.pkl')
            with (
                open(bm25_path, 'wb') as f,
                open(chunks_path, 'wb') as g,
            ):
                f.write(bm25_bytes)
                g.write(chunks_bytes)
        except OSError:
            raise RuntimeError(
                f"Failed to save BM25 index to '{processed_dir}'."
            )

    def load(self, processed_dir: str) -> None:
        if not load_dotenv():
            raise EnvironmentError(
                "Failed to load .env file. Please ensure it exists and is "
                "readable."
            )

        key = os.getenv('KEY1')
        if key is None:
            raise ValueError("KEY1 environment variable is not set.")

        try:
            key_bytes = bytes.fromhex(key)
        except ValueError:
            raise ValueError(
                "KEY1 must be a valid 32-byte hex string."
            )

        if len(key_bytes) != 32:
            raise ValueError("KEY1 must decode to exactly 32 bytes.")

        stored_hash = os.getenv('HASH1', '')
        bm25_path = os.path.join(processed_dir, 'bm25_index.pkl')
        chunks_path = os.path.join(processed_dir, 'chunks_metadata.pkl')

        try:
            with (
                open(bm25_path, 'rb') as f,
                open(chunks_path, 'rb') as g,
            ):
                bm25_bytes = f.read()
                chunks_bytes = g.read()
        except OSError:
            raise RuntimeError(
                f"Failed to read BM25 files from '{processed_dir}'."
            )

        try:
            calculated_hash = hmac.new(
                key_bytes,
                bm25_bytes + chunks_bytes,
                sha512,
            ).hexdigest()
            if not hmac.compare_digest(calculated_hash, stored_hash):
                raise ValueError(
                    "BM25 data integrity check failed. The index may have "
                    "been tampered with."
                )
        except ValueError:
            raise
        except Exception:
            raise RuntimeError("Failed to validate BM25 index integrity.")

        try:
            self.bm25 = pickle.loads(bm25_bytes)
            self.chunks = pickle.loads(chunks_bytes)
        except Exception:
            raise RuntimeError("Failed to deserialize BM25 index data.")

    def search(
        self,
        query: str,
        k: int,
        question_id: str,
    ) -> MinimalSearchResults:
        if self.bm25 is None:
            raise RuntimeError("Index not loaded.")

        try:
            tokenized_query = tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
        except Exception:
            raise RuntimeError(
                f"Failed to compute BM25 scores for query: {query!r}."
            )

        top_k_indices = np.argsort(scores)[-k:][::-1]
        retrieved_sources: List[MinimalSource] = [
            self.chunks[i] for i in top_k_indices
        ]

        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources,
        )
