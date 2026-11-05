import pytest
from src.tokenizer import tokenize


def test_tokenize_simple():
    tokens = tokenize("machine learning model")
    assert "machine" in tokens
    assert "learning" in tokens
    assert "model" in tokens


def test_tokenize_snake_case_expansion():
    tokens = tokenize("retrieve_documents_from_corpus")
    assert "retrieve" in tokens
    assert "documents" in tokens
    assert "corpus" in tokens
    assert "retrieve_documents_from_corpus" in tokens


def test_tokenize_punctuation_and_numbers():
    tokens = tokenize("Python 3.12 is fast & scalable!")
    assert "python" in tokens
    assert "3" in tokens
    assert "12" in tokens
    assert "fast" in tokens
    assert "scalable" in tokens


def test_tokenize_stop_words_removal():
    tokens = tokenize("the quick brown fox jumps over a lazy dog")
    # 'the', 'over', 'a' are stop words
    assert "the" not in tokens
    assert "over" not in tokens
    assert "a" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens
    assert "lazy" in tokens
    assert "dog" in tokens


def test_tokenize_empty_and_whitespace():
    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []
    assert tokenize("---...???") == []

