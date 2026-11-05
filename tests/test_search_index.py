import os
import secrets
import tempfile
import pytest
from dotenv import set_key
from src.search_index import SearchIndex
from src.models import MinimalSearchResults, MinimalSource


class DummyIndex(SearchIndex):
    def __init__(self, processed_dir: str, env_key_name: str = "DUMMY_KEY"):
        super().__init__()
        self.processed_dir = processed_dir
        self.env_key_name = env_key_name

    def build_from_raw(self, raw_dir: str, processed_dir: str, max_chunk_size: int = 2000) -> None:
        pass

    def load(self, processed_dir: str) -> None:
        pass

    def save(self, processed_dir: str) -> None:
        pass

    def search(self, query: str, k: int, question_id: str) -> MinimalSearchResults:
        return MinimalSearchResults(question_id=question_id, question=query, retrieved_sources=[])


def test_cache_save_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_key = secrets.token_bytes(32).hex()
        set_key(".env", "DUMMY_KEY", dummy_key)

        index = DummyIndex(processed_dir=tmp_dir)
        sample_source = MinimalSource(file_path="test.py", first_character_index=0, last_character_index=10)
        index.query_cache[("sample query", 1)] = [sample_source]

        index._save_cache()

        # Create a new index instance and load the cache
        loaded_index = DummyIndex(processed_dir=tmp_dir)
        loaded_index._load_cache()

        assert ("sample query", 1) in loaded_index.query_cache
        cached_source = loaded_index.query_cache[("sample query", 1)][0]
        assert cached_source.file_path == "test.py"


def test_cache_tamper_detection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_key = secrets.token_bytes(32).hex()
        set_key(".env", "DUMMY_KEY", dummy_key)

        index = DummyIndex(processed_dir=tmp_dir)
        sample_source = MinimalSource(file_path="test.py", first_character_index=0, last_character_index=10)
        index.query_cache[("query", 2)] = [sample_source]
        index._save_cache()

        cache_path = os.path.join(tmp_dir, "dummyindex_cache.pkl")
        assert os.path.exists(cache_path)

        # Tamper with the cache content by altering the payload
        with open(cache_path, "r+b") as f:
            f.seek(-5, os.SEEK_END)
            f.write(b"CORRUPT")

        # Load cache: HMAC check should fail and safely fallback to empty cache
        loaded_index = DummyIndex(processed_dir=tmp_dir)
        loaded_index._load_cache()
        assert loaded_index.query_cache == {}


def test_cache_key_mismatch_rejection():
    with tempfile.TemporaryDirectory() as tmp_dir:
        key1 = secrets.token_bytes(32).hex()
        set_key(".env", "DUMMY_KEY", key1)

        index = DummyIndex(processed_dir=tmp_dir)
        index.query_cache[("query", 1)] = []
        index._save_cache()

        # Simulate rotating the key in the environment
        key2 = secrets.token_bytes(32).hex()
        set_key(".env", "DUMMY_KEY", key2)

        loaded_index = DummyIndex(processed_dir=tmp_dir)
        loaded_index._load_cache()
        # Different key means HMAC does not match -> safely empty
        assert loaded_index.query_cache == {}

