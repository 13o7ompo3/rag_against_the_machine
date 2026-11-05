import os
import tempfile
import pytest
from src.retriever import BM25Index
from src.semantic import SemanticIndex


def test_bm25_incremental_add_edit_delete():
    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as processed_dir:
        f1_path = os.path.join(raw_dir, "service.py")
        f2_path = os.path.join(raw_dir, "legacy.py")

        with open(f1_path, "w", encoding="utf-8") as f:
            f.write("def payment_gateway():\n    return 'Stripe integration active'\n")

        with open(f2_path, "w", encoding="utf-8") as f:
            f.write("def deprecated_feature():\n    return 'Obsolete code'\n")

        # 1. Initial build
        index = BM25Index()
        index.build_from_raw(raw_dir, processed_dir, max_chunk_size=500)
        assert len(index.manifest) == 2

        res_legacy = index.search(query="deprecated_feature", k=1, question_id="q1")
        assert len(res_legacy.retrieved_sources) == 1
        assert "legacy.py" in res_legacy.retrieved_sources[0].file_path

        # Populate cache
        assert len(index.query_cache) > 0

        # 2. Modify f1, delete f2, add f3
        with open(f1_path, "w", encoding="utf-8") as f:
            f.write("def payment_gateway():\n    return 'PayPal and Stripe integration v2'\n")

        os.remove(f2_path)

        f3_path = os.path.join(raw_dir, "notification.py")
        with open(f3_path, "w", encoding="utf-8") as f:
            f.write("def send_email():\n    return 'Email sent via SendGrid'\n")

        # 3. Incremental update
        index2 = BM25Index()
        index2.build_from_raw(raw_dir, processed_dir, max_chunk_size=500)

        # Legacy file should no longer be in manifest
        assert f2_path not in index2.manifest
        assert f3_path in index2.manifest

        # Search for legacy should not return legacy.py
        res_deleted = index2.search(query="deprecated_feature", k=1, question_id="q2")
        for s in res_deleted.retrieved_sources:
            assert "legacy.py" not in s.file_path

        # Search for newly added file
        res_new = index2.search(query="SendGrid", k=1, question_id="q3")
        assert len(res_new.retrieved_sources) > 0
        assert "notification.py" in res_new.retrieved_sources[0].file_path


def test_semantic_incremental_update():
    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as processed_dir:
        doc_a = os.path.join(raw_dir, "doc_a.txt")
        with open(doc_a, "w", encoding="utf-8") as f:
            f.write("Machine learning and natural language processing techniques.")

        idx1 = SemanticIndex()
        idx1.build_from_raw(raw_dir, processed_dir, max_chunk_size=500)
        assert len(idx1.chunks_metadata) == 1

        # Add a new document
        doc_b = os.path.join(raw_dir, "doc_b.txt")
        with open(doc_b, "w", encoding="utf-8") as f:
            f.write("Astrophysics and planetary astronomy discoveries.")

        idx2 = SemanticIndex()
        idx2.build_from_raw(raw_dir, processed_dir, max_chunk_size=500)
        assert len(idx2.chunks_metadata) == 2

        # Verify semantic retrieval of new content
        res = idx2.search(query="galaxies and solar system", k=1, question_id="q_astro")
        assert len(res.retrieved_sources) == 1
        assert "doc_b.txt" in res.retrieved_sources[0].file_path

