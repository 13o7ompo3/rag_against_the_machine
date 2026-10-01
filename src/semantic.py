import os
import hmac
import pickle
import secrets
from hashlib import sha512
from io import BytesIO
from typing import List

import torch
from dotenv import load_dotenv, set_key
from sentence_transformers import SentenceTransformer, util

from src.chunker import collect_chunked_documents
from src.models import MinimalSearchResults, MinimalSource
from src.search_index import SearchIndex


class SemanticIndex(SearchIndex):
    """A vector index built with a lightweight CPU model for embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings: torch.Tensor | None = None
        self.chunks_metadata: List[MinimalSource] = []

    def build_from_raw(
        self,
        raw_dir: str,
        processed_dir: str,
        max_chunk_size: int = 2000,
    ) -> None:
        """Chunk the corpus, build embeddings, and save the index.

        Args:
            raw_dir: The directory containing the raw files to be indexed.
            processed_dir: The directory where the indexing will be stored.
            max_chunk_size: The maximum size of each chunk.
        """
        chunked_documents = collect_chunked_documents(raw_dir, max_chunk_size)

        texts = [chunk.text for chunk in chunked_documents]
        chunks = [chunk.source for chunk in chunked_documents]

        self.chunks_metadata = chunks

        self.embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        self.save(processed_dir)

    def save(self, processed_dir: str) -> None:
        """Serialize embeddings and metadata to disk."""
        if self.embeddings is None:
            raise RuntimeError("Cannot save an empty index.")

        try:
            embeddings_buffer = BytesIO()
            torch.save(self.embeddings, embeddings_buffer)
            embeddings_bytes = embeddings_buffer.getvalue()
            metadata_bytes = pickle.dumps(self.chunks_metadata)
        except Exception:
            raise RuntimeError("Failed to serialize semantic index data.")

        try:
            key = secrets.token_bytes(32)
            integrity_hash = hmac.new(
                key,
                embeddings_bytes + metadata_bytes,
                sha512,
            ).hexdigest()
            set_key('.env', 'KEY2', key.hex())
            set_key('.env', 'HASH2', integrity_hash)
        except Exception:
            raise RuntimeError("Failed to write semantic integrity metadata.")

        try:
            os.makedirs(processed_dir, exist_ok=True)
            embeddings_path = os.path.join(
                processed_dir,
                'semantic_embeddings.pt',
            )
            with open(embeddings_path, 'wb') as file_handle:
                file_handle.write(embeddings_bytes)

            metadata_path = os.path.join(
                processed_dir,
                'semantic_metadata.pkl',
            )
            with open(metadata_path, 'wb') as file_handle:
                file_handle.write(metadata_bytes)
        except OSError:
            raise RuntimeError(
                f"Failed to save semantic index to '{processed_dir}'."
            )

    def load(self, processed_dir: str) -> None:
        """Load the serialized embeddings and metadata from disk."""
        tensor_path = os.path.join(processed_dir, 'semantic_embeddings.pt')
        metadata_path = os.path.join(processed_dir, 'semantic_metadata.pkl')

        if not load_dotenv():
            raise EnvironmentError("Failed to load .env file.")

        key = os.getenv('KEY2')
        if key is None:
            raise ValueError("KEY2 environment variable is not set.")

        stored_hash = os.getenv('HASH2', '')

        try:
            key_bytes = bytes.fromhex(key)
        except ValueError:
            raise ValueError("KEY2 must be a valid 32-byte hex string.")

        if len(key_bytes) != 32:
            raise ValueError("KEY2 must decode to exactly 32 bytes.")

        try:
            with open(tensor_path, 'rb') as file_handle:
                embeddings_bytes = file_handle.read()
            with open(metadata_path, 'rb') as file_handle:
                metadata_bytes = file_handle.read()
        except OSError:
            raise RuntimeError(
                f"Failed to read semantic files from '{processed_dir}'."
            )

        try:
            calculated_hash = hmac.new(
                key_bytes,
                embeddings_bytes + metadata_bytes,
                sha512,
            ).hexdigest()
            if not hmac.compare_digest(calculated_hash, stored_hash):
                raise ValueError(
                    "Semantic data integrity check failed. The index may "
                    "have been tampered with."
                )
        except ValueError:
            raise
        except Exception:
            raise RuntimeError("Failed to validate semantic index integrity.")

        try:
            self.embeddings = torch.load(
                BytesIO(embeddings_bytes),
                map_location='cpu',
            )
            self.chunks_metadata = pickle.loads(metadata_bytes)
        except Exception:
            raise RuntimeError("Failed to deserialize semantic index data.")

    def search(
        self,
        query: str,
        k: int,
        question_id: str,
    ) -> MinimalSearchResults:
        if self.embeddings is None:
            raise RuntimeError("Index not loaded.")

        if len(self.chunks_metadata) == 0:
            return MinimalSearchResults(
                question_id=question_id,
                question=query,
                retrieved_sources=[],
            )

        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        except Exception:
            raise RuntimeError(
                f"Failed to encode or score semantic query: {query!r}."
            )

        top_k = min(k, len(self.chunks_metadata))
        try:
            top_results = torch.topk(cos_scores, k=top_k)
        except Exception:
            raise RuntimeError("Failed to rank semantic search results.")

        retrieved_sources: List[MinimalSource] = []
        for _, idx in zip(top_results.values, top_results.indices):
            retrieved_sources.append(self.chunks_metadata[int(idx)])

        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources,
        )
