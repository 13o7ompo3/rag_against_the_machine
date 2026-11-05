import pytest
from src.bm25 import BM25


def test_bm25_scoring():
    corpus = [
        ["deep", "learning", "model", "neural", "network"],
        ["database", "sql", "query", "index", "table"],
        ["climate", "weather", "temperature", "forecast"],
    ]
    bm25 = BM25(corpus)

    scores = bm25.get_scores(["sql", "query"])
    # Document 1 (database) should score highest for ["sql", "query"]
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    assert scores[0] == 0.0
    assert scores[2] == 0.0


def test_bm25_empty_query():
    corpus = [["hello", "world"]]
    bm25 = BM25(corpus)
    scores = bm25.get_scores([])
    assert scores == [0.0]


def test_bm25_add_documents():
    initial_corpus = [["python", "programming"]]
    bm25 = BM25(initial_corpus)
    assert len(bm25.docs_lens) == 1

    new_corpus = [
        ["rust", "systems", "programming"],
        ["golang", "concurrency"],
    ]
    bm25.add_documents(new_corpus)
    assert len(bm25.docs_lens) == 3
    assert len(bm25.docs) == 3

    # Need to rebuild IDF or score
    bm25._rebuild_idf()
    scores = bm25.get_scores(["rust"])
    assert scores[1] > 0.0
    assert scores[0] == 0.0


def test_bm25_remove_documents():
    corpus = [
        ["doc0", "apple"],
        ["doc1", "banana"],
        ["doc2", "cherry"],
    ]
    bm25 = BM25(corpus)

    # Remove doc1 (index 1)
    bm25.remove_documents([1])
    assert len(bm25.docs_lens) == 2
    assert len(bm25.docs) == 2

    # doc2 is now at index 1
    assert "cherry" in bm25.docs[1]
    assert "banana" not in bm25.docs[0]
    assert "banana" not in bm25.docs[1]

