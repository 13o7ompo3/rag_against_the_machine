import hmac
import os
import pickle
import secrets
from hashlib import sha512
from typing import List

import numpy as np
from dotenv import load_dotenv, set_key

from .bm25 import BM25
from .chunker import collect_chunk_delta
from .models import MinimalSearchResults, MinimalSource
from .search_index import SearchIndex
from .tokenizer import tokenize
import logging

logger = logging.getLogger(__name__)


class BM25Index(SearchIndex):
    """BM25-backed search index with the same interface as semantic search."""

    def __init__(self, processed_dir: str | None = None):
        super().__init__()
        self.bm25: BM25 | None = None
        self.processed_dir = processed_dir

    def build_from_raw(
        self,
        raw_dir: str,
        processed_dir: str,
        max_chunk_size: int = 2000,
    ) -> None:
        try:
            self.load(processed_dir)
            if self.max_chunk_size != max_chunk_size:
                raise ValueError("max_chunk_size changed")
        except Exception as e:
            logger.info(f"Failed to load existing index: {e}."
                        " Rebuilding the index.")
            self.bm25 = None
            self.chunks_metadata: list[MinimalSource] = []
            self.manifest: dict[str, str] = {}
            self.max_chunk_size = max_chunk_size

        deleted_file_paths, added_chunks, new_manifest = collect_chunk_delta(
            raw_dir,
            self.manifest,
            max_chunk_size,
        )

        if deleted_file_paths and self.chunks_metadata:
            deleted_set = set(deleted_file_paths)
            deleted_indices = [
                idx
                for idx, chunk in enumerate(self.chunks_metadata)
                if chunk.file_path in deleted_set
            ]

            if deleted_indices:
                if self.bm25 is None:
                    raise RuntimeError("Index not loaded.")

                self.bm25.remove_documents(deleted_indices)

                for idx in sorted(deleted_indices, reverse=True):
                    self.chunks_metadata.pop(idx)

        added_corpus: List[List[str]] = []
        added_sources: List[MinimalSource] = []
        for chunk in added_chunks:
            tokens = tokenize(chunk.text)
            if tokens:
                added_corpus.append(tokens)
                added_sources.append(chunk.source)

        if added_corpus:
            if self.bm25 is None:
                self.bm25 = BM25(added_corpus)
            else:
                self.bm25.add_documents(added_corpus)

            self.chunks_metadata.extend(added_sources)

        self.manifest = new_manifest
        self.save(processed_dir)

    def save(self, processed_dir: str) -> None:
        if self.bm25 is None:
            raise RuntimeError("Cannot persist an empty index.")

        try:
            bm25_bytes = pickle.dumps(self.bm25)
            chunks_bytes = pickle.dumps(self.chunks_metadata)
            manifest_bytes = pickle.dumps(self.manifest)
            max_chunk_size_bytes = pickle.dumps(self.max_chunk_size)
        except Exception:
            raise RuntimeError("Failed to serialize BM25 index data.")

        try:
            key = secrets.token_bytes(32)
            integrity_hash = hmac.new(
                key,
                bm25_bytes + chunks_bytes + manifest_bytes + max_chunk_size_bytes,
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
            manifest_path = os.path.join(processed_dir, 'manifest.pkl')
            max_chunk_size_path = os.path.join(processed_dir, 'max_chunk_size.pkl')
            with (open(bm25_path, 'wb') as bm25_handle,
                 open(chunks_path, 'wb') as chunks_handle,
                 open(manifest_path, 'wb') as manifest_handle,
                 open(max_chunk_size_path, 'wb') as max_chunk_size_handle):
                bm25_handle.write(bm25_bytes)
                chunks_handle.write(chunks_bytes)
                manifest_handle.write(manifest_bytes)
                max_chunk_size_handle.write(max_chunk_size_bytes)
        except OSError:
            raise RuntimeError(
                f"Failed to save BM25 index to '{processed_dir}'."
            )

    def load(self, processed_dir: str) -> None:
        bm25_path = os.path.join(processed_dir, 'bm25_index.pkl')
        chunks_path = os.path.join(processed_dir, 'chunks_metadata.pkl')
        manifest_path = os.path.join(processed_dir, 'manifest.pkl')
        max_chunk_size_path = os.path.join(processed_dir, 'max_chunk_size.pkl')

        if not all(
            os.path.exists(path)
            for path in (bm25_path, chunks_path, manifest_path, max_chunk_size_path)
        ):
            raise RuntimeError(
                f"BM25 index files not found in '{processed_dir}'."
            )

        if not load_dotenv(override=True):
            raise EnvironmentError("Failed to load .env file.")

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

        try:
            with (
                open(bm25_path, 'rb') as bm25_handle,
                open(chunks_path, 'rb') as chunks_handle,
                open(manifest_path, 'rb') as manifest_handle,
                open(max_chunk_size_path, 'rb') as max_chunk_size_handle,
            ):
                bm25_bytes = bm25_handle.read()
                chunks_bytes = chunks_handle.read()
                manifest_bytes = manifest_handle.read()
                max_chunk_size_bytes = max_chunk_size_handle.read()
        except OSError:
            raise RuntimeError(
                f"Failed to read BM25 files from '{processed_dir}'."
            )

        try:
            calculated_hash = hmac.new(
                key_bytes,
                bm25_bytes + chunks_bytes + manifest_bytes + max_chunk_size_bytes,
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
            self.chunks_metadata = pickle.loads(chunks_bytes)
            self.manifest = pickle.loads(manifest_bytes)
            self.max_chunk_size = pickle.loads(max_chunk_size_bytes)
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
            self.chunks_metadata[i] for i in top_k_indices
        ]

        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources,
        )
