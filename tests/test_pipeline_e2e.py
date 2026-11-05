import os
import tempfile
import pytest
from src.retriever import BM25Index
from src.semantic import SemanticIndex
from src.hybrid import HybridIndex
from src.models import MinimalSource


REAL_PYTHON_CODE = '''"""Database connection and user repository module."""
import os
from typing import Optional, Dict

class DatabaseConnection:
    """Manages pool connections to PostgreSQL database."""
    def __init__(self, host: str = "localhost", port: int = 5432):
        self.host = host
        self.port = port
        self.is_connected = False

    def connect(self) -> bool:
        """Establish connection to PostgreSQL cluster."""
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        """Close connection pool."""
        self.is_connected = False

def query_user_by_email(email: str) -> Optional[Dict[str, str]]:
    """Retrieve user credentials and profile information using user email address."""
    if "@" not in email:
        raise ValueError("Invalid email format.")
    return {"email": email, "role": "admin"}
'''

REAL_MARKDOWN_DOC = '''# Authentication & Security Protocol

## Overview
This document outlines authentication using HMAC-SHA512 and JWT tokens for API requests.

## Session Management
Sessions are stored in Redis with a TTL of 3600 seconds.
When a user logs out, their active token is placed on the revocation blocklist.

## Rate Limiting
API endpoints are throttled to a maximum of 100 requests per minute per IP address.
'''


@pytest.fixture
def realistic_corpus_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "database.py")
        with open(db_path, "w", encoding="utf-8") as f:
            f.write(REAL_PYTHON_CODE)

        doc_path = os.path.join(tmp_dir, "security.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(REAL_MARKDOWN_DOC)

        yield tmp_dir


def test_bm25_real_keyword_search(realistic_corpus_dir):
    with tempfile.TemporaryDirectory() as processed_dir:
        bm25_idx = BM25Index()
        bm25_idx.build_from_raw(realistic_corpus_dir, processed_dir, max_chunk_size=500)

        # Exact keyword search
        results = bm25_idx.search(query="query_user_by_email", k=1, question_id="q_exact")
        assert len(results.retrieved_sources) == 1
        assert "database.py" in results.retrieved_sources[0].file_path

        # Search for markdown security terms
        res_sec = bm25_idx.search(query="rate limiting requests per minute", k=1, question_id="q_sec")
        assert len(res_sec.retrieved_sources) == 1
        assert "security.md" in res_sec.retrieved_sources[0].file_path


def test_semantic_real_concept_search(realistic_corpus_dir):
    with tempfile.TemporaryDirectory() as processed_dir:
        semantic_idx = SemanticIndex()
        semantic_idx.build_from_raw(realistic_corpus_dir, processed_dir, max_chunk_size=500)

        # Semantic concept query (words "lookup customer credentials" do not literally appear in the code)
        results = semantic_idx.search(
            query="lookup customer login and profile by email",
            k=1,
            question_id="q_concept"
        )
        assert len(results.retrieved_sources) == 1
        assert "database.py" in results.retrieved_sources[0].file_path

        # Semantic concept for session expiry
        res_exp = semantic_idx.search(
            query="how long do user sessions remain active before expiring",
            k=1,
            question_id="q_exp"
        )
        assert len(res_exp.retrieved_sources) == 1
        assert "security.md" in res_exp.retrieved_sources[0].file_path


def test_hybrid_real_end_to_end_search(realistic_corpus_dir):
    with tempfile.TemporaryDirectory() as processed_dir:
        bm25 = BM25Index()
        semantic = SemanticIndex()
        hybrid = HybridIndex(bm25, semantic)

        hybrid.build_from_raw(realistic_corpus_dir, processed_dir, max_chunk_size=500)

        # Search hybrid
        results = hybrid.search(query="PostgreSQL connection pool", k=2, question_id="q_hybrid")
        assert len(results.retrieved_sources) > 0
        top_source = results.retrieved_sources[0]
        assert "database.py" in top_source.file_path


def test_real_save_load_hmac_integrity(realistic_corpus_dir):
    with tempfile.TemporaryDirectory() as processed_dir:
        bm25 = BM25Index()
        bm25.build_from_raw(realistic_corpus_dir, processed_dir, max_chunk_size=500)

        # Check atomic save created no leftover .tmp files
        leftover_tmp = [f for f in os.listdir(processed_dir) if f.endswith(".tmp")]
        assert len(leftover_tmp) == 0

        # Reload in a clean instance and verify identical top result
        loaded = BM25Index()
        loaded.load(processed_dir)

        r1 = bm25.search("PostgreSQL", k=1, question_id="q1")
        r2 = loaded.search("PostgreSQL", k=1, question_id="q1")

        assert r1.retrieved_sources[0].file_path == r2.retrieved_sources[0].file_path
        assert r1.retrieved_sources[0].first_character_index == r2.retrieved_sources[0].first_character_index

