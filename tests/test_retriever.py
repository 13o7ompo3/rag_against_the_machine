import os
import tempfile
import pytest
from src.retriever import BM25Index


def test_bm25_index_build_and_search(temp_raw_dir, monkeypatch):
    with tempfile.TemporaryDirectory() as processed_dir:
        index = BM25Index()
        # Build index from dummy raw directory
        index.build_from_raw(temp_raw_dir, processed_dir, max_chunk_size=500)

        assert index.bm25 is not None
        assert len(index.chunks_metadata) > 0

        # Run a search
        results = index.search(query="hello", k=2, question_id="test_q1")
        assert results.question_id == "test_q1"
        assert len(results.retrieved_sources) > 0


def test_bm25_index_save_and_load(temp_raw_dir):
    with tempfile.TemporaryDirectory() as processed_dir:
        index = BM25Index()
        index.build_from_raw(temp_raw_dir, processed_dir, max_chunk_size=500)

        # Load into a fresh instance
        loaded_index = BM25Index()
        loaded_index.load(processed_dir)

        assert loaded_index.bm25 is not None
        assert len(loaded_index.chunks_metadata) == len(index.chunks_metadata)

        # Search should yield same results
        res1 = index.search(query="markdown", k=1, question_id="q1")
        res2 = loaded_index.search(query="markdown", k=1, question_id="q1")
        assert len(res1.retrieved_sources) == len(res2.retrieved_sources)
        assert res1.retrieved_sources[0].file_path == res2.retrieved_sources[0].file_path

