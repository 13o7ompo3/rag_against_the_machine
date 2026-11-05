import json
import os
import tempfile
import pytest
from src.__main__ import RagCLI
from src.evaluator import evaluate_results
from src.models import (
    RagDataset,
    AnsweredQuestion,
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
)


SAMPLE_CODE = """def authenticate_user(username, password):
    if username == "admin" and password == "secret":
        return True
    return False

def get_database_version():
    return "PostgreSQL 16.2"
"""


@pytest.fixture
def cli_environment(monkeypatch):
    """Sets up a complete isolated environment with raw data, processed dir, and ground truth."""
    with tempfile.TemporaryDirectory() as base_dir:
        raw_dir = os.path.join(base_dir, "raw")
        processed_dir = os.path.join(base_dir, "processed")
        save_dir = os.path.join(base_dir, "results")
        os.makedirs(raw_dir)
        os.makedirs(processed_dir)
        os.makedirs(save_dir)

        code_file = os.path.join(raw_dir, "auth.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CODE)

        # Build ground-truth dataset matching auth.py
        # "authenticate_user" is from char 0 to char 115 approx
        gt_source = MinimalSource(
            file_path=code_file,
            first_character_index=0,
            last_character_index=115,
        )
        gt_question = AnsweredQuestion(
            question_id="q_auth_1",
            question="How does the system authenticate users?",
            sources=[gt_source],
            answer="By comparing username with admin and password with secret.",
        )
        dataset = RagDataset(rag_questions=[gt_question])

        dataset_path = os.path.join(base_dir, "dataset.json")
        with open(dataset_path, "w", encoding="utf-8") as f:
            f.write(dataset.model_dump_json(indent=2))

        # Change directory to base_dir so relative paths data/raw/ match or mock them
        yield {
            "raw_dir": raw_dir,
            "processed_dir": processed_dir,
            "save_dir": save_dir,
            "code_file": code_file,
            "dataset_path": dataset_path,
        }


def test_cli_search_dataset_and_evaluate_end_to_end(cli_environment, monkeypatch):
    from src.retriever import BM25Index

    raw_dir = cli_environment["raw_dir"]
    processed_dir = cli_environment["processed_dir"]
    save_dir = cli_environment["save_dir"]
    dataset_path = cli_environment["dataset_path"]

    # 1. Build Index
    index = BM25Index()
    index.build_from_raw(raw_dir, processed_dir, max_chunk_size=500)

    # 2. Mock _make_index in RagCLI and redirect load to use our test processed_dir
    cli = RagCLI()
    monkeypatch.setattr(
        cli,
        "_make_index",
        lambda backend: index,
    )
    monkeypatch.setattr(
        index,
        "load",
        lambda path: BM25Index.load(index, processed_dir),
    )

    # 3. Run search_dataset
    cli.search_dataset(
        dataset_path=dataset_path,
        k=3,
        save_directory=save_dir,
        backend="bm25",
    )

    result_file = os.path.join(save_dir, "dataset.json")
    assert os.path.exists(result_file)

    with open(result_file, "r", encoding="utf-8") as f:
        student_results_dict = json.load(f)

    student_results = StudentSearchResults(**student_results_dict)
    assert len(student_results.search_results) == 1
    assert student_results.search_results[0].question_id == "q_auth_1"
    assert len(student_results.search_results[0].retrieved_sources) > 0

    # 4. Evaluate results against ground truth
    # evaluate_results should run without crashing
    evaluate_results(result_file, dataset_path)


def test_cli_search_dataset_batch_resilience(cli_environment, monkeypatch):
    """Verify bug fix: bad query exception does not crash batch run."""
    cli = RagCLI()

    class FaultyIndex:
        def load(self, path):
            pass
        def search(self, query, k, question_id):
            if "fail" in query:
                raise RuntimeError("Simulated search error!")
            return MinimalSearchResults(
                question_id=question_id,
                question=query,
                retrieved_sources=[],
            )
        def _save_cache(self):
            pass

    monkeypatch.setattr(cli, "_make_index", lambda backend: FaultyIndex())

    # Create dataset with 2 questions: one valid, one that triggers failure
    dataset = RagDataset(
        rag_questions=[
            AnsweredQuestion(question_id="q1", question="Normal query", sources=[], answer=""),
            AnsweredQuestion(question_id="q2", question="Please fail this", sources=[], answer=""),
        ]
    )
    dataset_path = os.path.join(cli_environment["save_dir"], "batch_test.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(dataset.model_dump_json())

    # search_dataset should complete gracefully
    cli.search_dataset(
        dataset_path=dataset_path,
        k=1,
        save_directory=cli_environment["save_dir"],
        backend="bm25",
    )

    out_file = os.path.join(cli_environment["save_dir"], "batch_test.json")
    with open(out_file, "r", encoding="utf-8") as f:
        saved = json.load(f)

    # Both questions must be present in output, second one with empty sources fallback
    assert len(saved["search_results"]) == 2
    assert saved["search_results"][0]["question_id"] == "q1"
    assert saved["search_results"][1]["question_id"] == "q2"
    assert saved["search_results"][1]["retrieved_sources"] == []

