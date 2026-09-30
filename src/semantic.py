import os
import pickle
from typing import List

import torch
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

        os.makedirs(processed_dir, exist_ok=True)

        torch.save(
            self.embeddings,
            os.path.join(processed_dir, 'semantic_embeddings.pt'),
        )

        with open(
            os.path.join(processed_dir, 'semantic_metadata.pkl'),
            'wb',
        ) as file_handle:
            pickle.dump(self.chunks_metadata, file_handle)

    def load(self, processed_dir: str) -> None:
        """Load the serialized embeddings and metadata from disk."""
        tensor_path = os.path.join(processed_dir, 'semantic_embeddings.pt')
        metadata_path = os.path.join(processed_dir, 'semantic_metadata.pkl')

        self.embeddings = torch.load(tensor_path, map_location='cpu')

        with open(metadata_path, 'rb') as file_handle:
            self.chunks_metadata = pickle.load(file_handle)

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

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        top_k = min(k, len(self.chunks_metadata))
        top_results = torch.topk(cos_scores, k=top_k)

        retrieved_sources: List[MinimalSource] = []
        for _, idx in zip(top_results.values, top_results.indices):
            retrieved_sources.append(self.chunks_metadata[int(idx)])

        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources,
        )
