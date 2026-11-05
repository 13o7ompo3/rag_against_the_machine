import pytest
from src.models import (
    MinimalSource,
    UnansweredQuestion,
    AnsweredQuestion,
    RagDataset,
    MinimalSearchResults,
    StudentSearchResults,
)


def test_minimal_source_serialization():
    src = MinimalSource(file_path="main.py", first_character_index=0, last_character_index=42)
    data = src.model_dump()
    assert data == {
        "file_path": "main.py",
        "first_character_index": 0,
        "last_character_index": 42,
    }

    reconstructed = MinimalSource(**data)
    assert reconstructed == src


def test_rag_dataset_with_mixed_questions():
    q1 = UnansweredQuestion(question="What is BM25?")
    q2 = AnsweredQuestion(
        question="Where is main?",
        sources=[MinimalSource(file_path="main.py", first_character_index=0, last_character_index=10)],
        answer="It is in main.py.",
    )

    dataset = RagDataset(rag_questions=[q1, q2])
    assert len(dataset.rag_questions) == 2
    assert isinstance(dataset.rag_questions[0], UnansweredQuestion)
    assert isinstance(dataset.rag_questions[1], AnsweredQuestion)


def test_student_search_results_json_roundtrip():
    res = MinimalSearchResults(
        question_id="q1",
        question="Query",
        retrieved_sources=[
            MinimalSource(file_path="src.py", first_character_index=5, last_character_index=15)
        ],
    )
    student_res = StudentSearchResults(search_results=[res], k=1)
    json_str = student_res.model_dump_json()

    loaded = StudentSearchResults.model_validate_json(json_str)
    assert loaded.k == 1
    assert len(loaded.search_results) == 1
    assert loaded.search_results[0].question_id == "q1"

