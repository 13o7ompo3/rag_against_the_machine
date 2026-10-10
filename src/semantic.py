import hmac
import logging
import os
import pickle
import secrets
from hashlib import sha512
from io import BytesIO
from typing import List

import torch
from dotenv import load_dotenv, set_key
from sentence_transformers import SentenceTransformer, util

from .chunker import collect_chunk_delta
from .models import MinimalSearchResults, MinimalSource
from .search_index import SearchIndex

logger = logging.getLogger(__name__)


class SemanticIndex(SearchIndex):
    """A vector index built with a lightweight CPU model for embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.embeddings: torch.Tensor | None = None
        self.env_key_name = 'KEY2'

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
        try:
            self.load(processed_dir)
            if self.max_chunk_size != max_chunk_size:
                raise ValueError("max_chunk_size changed")
        except Exception as e:
            logger.info(f"Failed to load existing index: {e}."
                        " Rebuilding the index.")
            self.embeddings = None
            self.chunks_metadata: list[MinimalSource] = []
            self.manifest: dict[str, str] = {}
            self.max_chunk_size = max_chunk_size
            self.query_cache.clear()

        deleted_file_paths, added_chunks, new_manifest = collect_chunk_delta(
            raw_dir,
            self.manifest,
            max_chunk_size,
        )

        if deleted_file_paths and self.chunks_metadata:
            deleted_set = set(deleted_file_paths)
            mask = [
                chunk.file_path not in deleted_set
                for chunk in self.chunks_metadata
            ]

            self.chunks_metadata = [
                chunk
                for chunk, keep in zip(self.chunks_metadata, mask)
                if keep
            ]

            if self.embeddings is not None and self.embeddings.numel() > 0:
                self.embeddings = self.embeddings[mask]

        added_texts = [chunk.text for chunk in added_chunks if chunk.text]
        added_sources = [chunk.source for chunk in added_chunks if chunk.text]

        if added_texts:
            added_embeddings = self.model.encode(
                added_texts,
                convert_to_tensor=True,
                show_progress_bar=True,
            )

            if self.embeddings is None or self.embeddings.numel() == 0:
                self.embeddings = added_embeddings
            else:
                self.embeddings = torch.cat(
                    [self.embeddings, added_embeddings],
                    dim=0,
                )

            self.chunks_metadata.extend(added_sources)

        if deleted_file_paths or added_chunks:
            self.manifest = new_manifest
            self.query_cache.clear()
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
            manifest_bytes = pickle.dumps(self.manifest)
            max_chunk_size_bytes = pickle.dumps(self.max_chunk_size)
        except Exception:
            raise RuntimeError("Failed to serialize semantic index data.")

        try:
            os.makedirs(processed_dir, exist_ok=True)
            tensor_path = os.path.join(
                processed_dir, 'semantic_embeddings.pt')
            metadata_path = os.path.join(
                processed_dir, 'semantic_metadata.pkl')
            manifest_path = os.path.join(
                processed_dir, 'semantic_manifest.pkl')
            max_chunk_size_path = os.path.join(
                processed_dir, 'semantic_max_chunk_size.pkl')

            with (open(tensor_path + '.tmp', 'wb') as tensor_handle,
                 open(metadata_path + '.tmp', 'wb') as metadata_handle,
                 open(manifest_path + '.tmp', 'wb') as manifest_handle,
                 open(max_chunk_size_path + '.tmp', 'wb') as max_chunk_size_handle):
                tensor_handle.write(embeddings_bytes)
                metadata_handle.write(metadata_bytes)
                manifest_handle.write(manifest_bytes)
                max_chunk_size_handle.write(max_chunk_size_bytes)

            os.replace(tensor_path + '.tmp', tensor_path)
            os.replace(metadata_path + '.tmp', metadata_path)
            os.replace(manifest_path + '.tmp', manifest_path)
            os.replace(max_chunk_size_path + '.tmp', max_chunk_size_path)
        except OSError:
            raise RuntimeError(
                f"Failed to save semantic index to '{processed_dir}'."
            )

        try:
            key = secrets.token_bytes(32)
            integrity_hash = hmac.new(
                key,
                embeddings_bytes + metadata_bytes + manifest_bytes + max_chunk_size_bytes,
                sha512,
            ).hexdigest()
            set_key('.env', 'KEY2', key.hex())
            set_key('.env', 'HASH2', integrity_hash)
        except Exception:
            raise RuntimeError("Failed to write semantic integrity metadata.")
        self._save_cache()

    def load(self, processed_dir: str) -> None:
        """Load the serialized embeddings and metadata from disk."""
        tensor_path = os.path.join(processed_dir, 'semantic_embeddings.pt')
        metadata_path = os.path.join(processed_dir, 'semantic_metadata.pkl')
        manifest_path = os.path.join(processed_dir, 'semantic_manifest.pkl')
        max_chunk_size_path = os.path.join(processed_dir, 'semantic_max_chunk_size.pkl')

        if not all(os.path.exists(path)
                   for path in (tensor_path, metadata_path, manifest_path, max_chunk_size_path)):
            raise RuntimeError(
                f"Semantic index files not found in '{processed_dir}'."
            )

        if not load_dotenv(override=True):
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
            with open(manifest_path, 'rb') as file_handle:
                manifest_bytes = file_handle.read()
            with open(max_chunk_size_path, 'rb') as file_handle:
                max_chunk_size_bytes = file_handle.read()
        except OSError:
            raise RuntimeError(
                f"Failed to read semantic files from '{processed_dir}'."
            )

        try:
            calculated_hash = hmac.new(
                key_bytes,
                embeddings_bytes + metadata_bytes + manifest_bytes + max_chunk_size_bytes,
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
            self.manifest = pickle.loads(manifest_bytes)
            self.max_chunk_size = pickle.loads(max_chunk_size_bytes)
        except Exception:
            raise RuntimeError("Failed to deserialize semantic index data.")
        
        self.processed_dir = processed_dir
        self._load_cache()

    def search(
        self,
        query: str,
        k: int,
        question_id: str,
    ) -> MinimalSearchResults:
        if self.embeddings is None:
            raise RuntimeError("Index not loaded.")

        cache_key = (query, k)
        if cache_key in self.query_cache:
            retrieved_sources = self.query_cache[cache_key]
            return MinimalSearchResults(
                question_id=question_id,
                question=query,
                retrieved_sources=retrieved_sources,
            )

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

        self.query_cache[(query, k)] = retrieved_sources
        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources,
        )
