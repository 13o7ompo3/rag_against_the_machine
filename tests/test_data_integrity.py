import os
import secrets
import shutil
import tempfile
import pytest
from dotenv import dotenv_values, set_key

from src.retriever import BM25Index
from src.semantic import SemanticIndex
from src.search_index import SearchIndex
from src.models import MinimalSearchResults, MinimalSource


@pytest.fixture(autouse=True)
def restore_env_per_test():
    """Ensure every test function leaves .env exactly as it started."""
    env_path = ".env"
    original_env_content = None
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            original_env_content = f.read()

    yield

    if original_env_content is not None:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(original_env_content)
    elif os.path.exists(env_path):
        os.remove(env_path)


@pytest.fixture
def bm25_indexed_dir(temp_raw_dir):
    """Builds a valid BM25 index in a temp directory and records keys in .env."""
    with tempfile.TemporaryDirectory() as proc_dir:
        index = BM25Index()
        index.build_from_raw(temp_raw_dir, proc_dir, max_chunk_size=500)
        yield proc_dir


@pytest.fixture
def semantic_indexed_dir(temp_raw_dir):
    """Builds a valid Semantic index in a temp directory and records keys in .env."""
    with tempfile.TemporaryDirectory() as proc_dir:
        index = SemanticIndex()
        index.build_from_raw(temp_raw_dir, proc_dir, max_chunk_size=500)
        yield proc_dir


# =========================================================================
# 1. BM25Index Integrity Tests
# =========================================================================

@pytest.mark.parametrize("missing_file", [
    "bm25_index.pkl",
    "chunks_metadata.pkl",
    "manifest.pkl",
    "max_chunk_size.pkl",
])
def test_bm25_missing_index_files_raise_runtime_error(bm25_indexed_dir, missing_file):
    """Removing any required index file must trigger a RuntimeError."""
    target_path = os.path.join(bm25_indexed_dir, missing_file)
    os.remove(target_path)

    index = BM25Index()
    with pytest.raises(RuntimeError, match="BM25 index files not found"):
        index.load(bm25_indexed_dir)


def test_bm25_missing_env_file_raises_environment_error(bm25_indexed_dir):
    """If .env file is deleted, load() must fail with EnvironmentError."""
    if os.path.exists(".env"):
        os.remove(".env")

    index = BM25Index()
    with pytest.raises(EnvironmentError, match="Failed to load .env file"):
        index.load(bm25_indexed_dir)


def test_bm25_empty_env_file_raises_environment_error(bm25_indexed_dir):
    """If .env is completely empty, load() must fail with EnvironmentError."""
    with open(".env", "w", encoding="utf-8") as f:
        f.write("")

    index = BM25Index()
    with pytest.raises(EnvironmentError, match="Failed to load .env file"):
        index.load(bm25_indexed_dir)


def test_bm25_missing_key1_in_env(bm25_indexed_dir):
    """If KEY1 is absent from .env, load() must raise ValueError."""
    env = dotenv_values(".env")
    with open(".env", "w", encoding="utf-8") as f:
        for k, v in env.items():
            if k != "KEY1":
                f.write(f"{k}='{v}'\n")

    index = BM25Index()
    with pytest.raises(ValueError, match="KEY1 environment variable is not set"):
        index.load(bm25_indexed_dir)


def test_bm25_invalid_hex_key_format(bm25_indexed_dir):
    """If KEY1 is not valid hexadecimal, load() must reject it."""
    set_key(".env", "KEY1", "this_is_not_a_valid_hex_string_zzzz")

    index = BM25Index()
    with pytest.raises(ValueError, match="KEY1 must be a valid 32-byte hex string"):
        index.load(bm25_indexed_dir)


@pytest.mark.parametrize("invalid_byte_length", [16, 20, 48, 64])
def test_bm25_invalid_key_byte_length(bm25_indexed_dir, invalid_byte_length):
    """If KEY1 decodes to anything other than exactly 32 bytes, load() must reject it."""
    invalid_key = secrets.token_bytes(invalid_byte_length).hex()
    set_key(".env", "KEY1", invalid_key)

    index = BM25Index()
    with pytest.raises(ValueError, match="KEY1 must decode to exactly 32 bytes"):
        index.load(bm25_indexed_dir)


@pytest.mark.parametrize("file_to_tamper", [
    "bm25_index.pkl",
    "chunks_metadata.pkl",
    "manifest.pkl",
    "max_chunk_size.pkl",
])
def test_bm25_payload_tampering_triggers_hmac_failure(bm25_indexed_dir, file_to_tamper):
    """A single bit flip in any binary payload must fail HMAC verification."""
    target_path = os.path.join(bm25_indexed_dir, file_to_tamper)
    with open(target_path, "r+b") as f:
        content = bytearray(f.read())
        # Flip the first byte
        content[0] ^= 0xFF
        f.seek(0)
        f.write(content)

    index = BM25Index()
    with pytest.raises(ValueError, match="BM25 data integrity check failed"):
        index.load(bm25_indexed_dir)


def test_bm25_env_hash_tampering_triggers_failure(bm25_indexed_dir):
    """Tampering with HASH1 in .env must fail HMAC comparison."""
    original_hash = dotenv_values(".env").get("HASH1", "")
    # Change last character of the hash
    altered_char = "0" if original_hash[-1] != "0" else "1"
    altered_hash = original_hash[:-1] + altered_char
    set_key(".env", "HASH1", altered_hash)

    index = BM25Index()
    with pytest.raises(ValueError, match="BM25 data integrity check failed"):
        index.load(bm25_indexed_dir)


def test_bm25_rotated_key_mismatch_triggers_failure(bm25_indexed_dir):
    """Replacing KEY1 with a valid but different 32-byte key must fail verification."""
    new_random_key = secrets.token_bytes(32).hex()
    set_key(".env", "KEY1", new_random_key)

    index = BM25Index()
    with pytest.raises(ValueError, match="BM25 data integrity check failed"):
        index.load(bm25_indexed_dir)


def test_bm25_atomic_save_leaves_no_tmp_files(temp_raw_dir):
    """Saving must write to .tmp and atomically swap; no .tmp files should linger."""
    with tempfile.TemporaryDirectory() as proc_dir:
        index = BM25Index()
        index.build_from_raw(temp_raw_dir, proc_dir, max_chunk_size=500)

        all_files = os.listdir(proc_dir)
        tmp_files = [f for f in all_files if f.endswith(".tmp")]
        assert len(tmp_files) == 0
        assert "bm25_index.pkl" in all_files
        assert "chunks_metadata.pkl" in all_files
        assert "manifest.pkl" in all_files
        assert "max_chunk_size.pkl" in all_files


# =========================================================================
# 2. SemanticIndex Integrity Tests
# =========================================================================

@pytest.mark.parametrize("missing_file", [
    "semantic_embeddings.pt",
    "semantic_metadata.pkl",
    "semantic_manifest.pkl",
    "semantic_max_chunk_size.pkl",
])
def test_semantic_missing_index_files_raise_runtime_error(semantic_indexed_dir, missing_file):
    """Removing any required semantic index file must trigger a RuntimeError."""
    target_path = os.path.join(semantic_indexed_dir, missing_file)
    os.remove(target_path)

    index = SemanticIndex()
    with pytest.raises(RuntimeError, match="Semantic index files not found"):
        index.load(semantic_indexed_dir)


def test_semantic_missing_key2_in_env(semantic_indexed_dir):
    """If KEY2 is absent from .env, load() must raise ValueError."""
    env = dotenv_values(".env")
    with open(".env", "w", encoding="utf-8") as f:
        for k, v in env.items():
            if k != "KEY2":
                f.write(f"{k}='{v}'\n")

    index = SemanticIndex()
    with pytest.raises(ValueError, match="KEY2 environment variable is not set"):
        index.load(semantic_indexed_dir)


def test_semantic_invalid_hex_key_format(semantic_indexed_dir):
    """If KEY2 is not valid hexadecimal, load() must reject it."""
    set_key(".env", "KEY2", "invalid_hex_characters_here!!")

    index = SemanticIndex()
    with pytest.raises(ValueError, match="KEY2 must be a valid 32-byte hex string"):
        index.load(semantic_indexed_dir)


def test_semantic_invalid_key_byte_length(semantic_indexed_dir):
    """If KEY2 is not 32 bytes, load() must reject it."""
    set_key(".env", "KEY2", secrets.token_bytes(16).hex())

    index = SemanticIndex()
    with pytest.raises(ValueError, match="KEY2 must decode to exactly 32 bytes"):
        index.load(semantic_indexed_dir)


@pytest.mark.parametrize("file_to_tamper", [
    "semantic_embeddings.pt",
    "semantic_metadata.pkl",
    "semantic_manifest.pkl",
    "semantic_max_chunk_size.pkl",
])
def test_semantic_payload_tampering_triggers_hmac_failure(semantic_indexed_dir, file_to_tamper):
    """Modifying even 1 byte in semantic binary files must fail HMAC verification."""
    target_path = os.path.join(semantic_indexed_dir, file_to_tamper)
    with open(target_path, "r+b") as f:
        content = bytearray(f.read())
        content[0] ^= 0xFF
        f.seek(0)
        f.write(content)

    index = SemanticIndex()
    with pytest.raises(ValueError, match="Semantic data integrity check failed"):
        index.load(semantic_indexed_dir)


def test_semantic_env_hash_tampering_triggers_failure(semantic_indexed_dir):
    """Tampering with HASH2 in .env must fail HMAC comparison."""
    original_hash = dotenv_values(".env").get("HASH2", "")
    altered_char = "a" if original_hash[-1] != "a" else "b"
    set_key(".env", "HASH2", original_hash[:-1] + altered_char)

    index = SemanticIndex()
    with pytest.raises(ValueError, match="Semantic data integrity check failed"):
        index.load(semantic_indexed_dir)


def test_semantic_atomic_save_leaves_no_tmp_files(temp_raw_dir):
    """Saving semantic index must leave no orphaned .tmp files."""
    with tempfile.TemporaryDirectory() as proc_dir:
        index = SemanticIndex()
        index.build_from_raw(temp_raw_dir, proc_dir, max_chunk_size=500)

        all_files = os.listdir(proc_dir)
        tmp_files = [f for f in all_files if f.endswith(".tmp")]
        assert len(tmp_files) == 0
        assert "semantic_embeddings.pt" in all_files
        assert "semantic_metadata.pkl" in all_files
        assert "semantic_manifest.pkl" in all_files
        assert "semantic_max_chunk_size.pkl" in all_files


# =========================================================================
# 3. Query Cache Data Integrity Tests
# =========================================================================

class MockCacheIndex(SearchIndex):
    def __init__(self, processed_dir: str, env_key_name: str = "CACHE_TEST_KEY"):
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


def test_cache_truncated_file_safely_resets_to_empty():
    """Cache file with fewer than 128 bytes (missing valid SHA-512 header) resets to empty."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_key = secrets.token_bytes(32).hex()
        set_key(".env", "CACHE_TEST_KEY", dummy_key)

        cache_path = os.path.join(tmp_dir, "mockcacheindex_cache.pkl")
        with open(cache_path, "wb") as f:
            f.write(b"short_truncated_header_data")

        index = MockCacheIndex(processed_dir=tmp_dir)
        index._load_cache()
        assert index.query_cache == {}


def test_cache_tampered_header_safely_resets_to_empty():
    """Tampering with the stored SHA-512 hex header must safely reset query_cache to empty."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_key = secrets.token_bytes(32).hex()
        set_key(".env", "CACHE_TEST_KEY", dummy_key)

        index = MockCacheIndex(processed_dir=tmp_dir)
        index.query_cache[("sample query", 1)] = [
            MinimalSource(file_path="f.py", first_character_index=0, last_character_index=5)
        ]
        index._save_cache()

        cache_path = os.path.join(tmp_dir, "mockcacheindex_cache.pkl")
        # Flip bytes in the first 128 bytes (HMAC header)
        with open(cache_path, "r+b") as f:
            f.seek(10)
            f.write(b"CORRUPTED_HASH")

        loaded = MockCacheIndex(processed_dir=tmp_dir)
        loaded._load_cache()
        assert loaded.query_cache == {}


def test_cache_tampered_payload_safely_resets_to_empty():
    """Tampering with payload after header must fail HMAC and safely reset to empty."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_key = secrets.token_bytes(32).hex()
        set_key(".env", "CACHE_TEST_KEY", dummy_key)

        index = MockCacheIndex(processed_dir=tmp_dir)
        index.query_cache[("sample query", 1)] = [
            MinimalSource(file_path="f.py", first_character_index=0, last_character_index=5)
        ]
        index._save_cache()

        cache_path = os.path.join(tmp_dir, "mockcacheindex_cache.pkl")
        # Flip bytes after offset 128
        with open(cache_path, "r+b") as f:
            f.seek(140)
            f.write(b"ALTERED_PAYLOAD")

        loaded = MockCacheIndex(processed_dir=tmp_dir)
        loaded._load_cache()
        assert loaded.query_cache == {}


def test_cache_missing_key_in_env_safely_resets_to_empty():
    """If cache key is missing from .env during _load_cache(), it safely defaults to empty."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_key = secrets.token_bytes(32).hex()
        set_key(".env", "CACHE_TEST_KEY", dummy_key)

        index = MockCacheIndex(processed_dir=tmp_dir)
        index.query_cache[("query", 1)] = []
        index._save_cache()

        # Delete the key from .env
        env = dotenv_values(".env")
        with open(".env", "w", encoding="utf-8") as f:
            for k, v in env.items():
                if k != "CACHE_TEST_KEY":
                    f.write(f"{k}='{v}'\n")

        loaded = MockCacheIndex(processed_dir=tmp_dir)
        loaded._load_cache()
        assert loaded.query_cache == {}

