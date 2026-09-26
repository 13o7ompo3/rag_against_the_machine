import os
import pickle
from typing import List
import numpy as np

from src.bm25 import BM25
from src.models import MinimalSource, MinimalSearchResults
from src.tokenizer import tokenize
from dotenv import load_dotenv
from hashlib import sha512


class BM25Retriever:
    def __init__(self, processed_dir: str):
        """
        Initializes the BM25Retriever.

        Args:
            processed_dir: The directory containing the pre-processed data.
        """
        if not load_dotenv():
            raise EnvironmentError("Failed to load .env file. "
                                   "Please ensure it exists and is readable.")
        key = os.getenv("KEY")
        if key is None:
            raise ValueError("KEY environment variable is not set.")

        try:
            key_bytes = bytes.fromhex(key)
        except ValueError:
            raise ValueError(
                "KEY environment variable is not a valid hex string.")

        hash = os.getenv("HASH", "")

        with (open(os.path.join(processed_dir, 'bm25_index.pkl'), 'rb') as f,
              open(os.path.join(processed_dir,
                                'chunks_metadata.pkl'), 'rb') as g):
            bm25 = f.read()
            chunks = g.read()
            if sha512(key_bytes + bm25 + chunks).hexdigest() != hash:
                raise ValueError("Data integrity check failed. "
                                 "The data may have been tampered with.")
            self.bm25: BM25 = pickle.loads(bm25)
            self.chunks: List[MinimalSource] = pickle.loads(chunks)

    def search_single(self,
                      query: str,
                      question_id: str,
                      k: int) -> MinimalSearchResults:
        """Search for the top-k most relevant chunks for a given query.

        Args:
            query: The search query.
            question_id: The ID of the question.
            k: The number of top results to return.

        Returns:
            A MinimalSearchResults object containing the search results.
        """
        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_k_indices = np.argsort(scores)[-k:][::-1]

        retrieved_sources: List[MinimalSource] = [self.chunks[i]
                                                  for i in top_k_indices]

        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=retrieved_sources
        )
