import fire
import logging
import json
import uuid
import os
from tqdm import tqdm
from src.indexer import build_index
from src.retriever import BM25Retriever
from src.evaluator import evaluate_results
from src.models import StudentSearchResults, RagDataset

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')


class RagCLI:
    """Command-Line Interface for the RAG pipeline."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""

        logging.info(f"Running indexer with max_chunk_size={max_chunk_size}.")
        build_index("data/raw/", "data/processed/", max_chunk_size)

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query."""
        logging.info(f"Searching for '{query}' returning top {k} results...")
        try:
            retriever = BM25Retriever(processed_dir="data/processed")
        except Exception as e:
            logging.error(f"Failed to initialize BM25Retriever: {e}")
            return
        result = retriever.search_single(query,
                                         question_id=str(uuid.uuid4()),
                                         k=k)

        print(result.model_dump_json(indent=2))

    def search_dataset(self, dataset_path: str,
                       k: int, save_directory: str) -> None:
        """Run search over a whole dataset
        and write a StudentSearchResults JSON file.

        Args:
            dataset_path: Path to the RagDataset JSON file.
            k: Number of top results to return for each query.
            save_directory: Directory to save the StudentSearchResults.
        """
        logging.info(f"Batch searching dataset {dataset_path} (k={k}). "
                     f"Saving to {save_directory}...")

        try:
            retriever = BM25Retriever(processed_dir="data/processed")
        except Exception as e:
            logging.error(f"Failed to initialize BM25Retriever: {e}")
            return

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                if not isinstance(raw_data, dict):
                    raise ValueError("Dataset JSON must be a dictionary.")
                dataset = RagDataset(**raw_data)
        except Exception as e:
            logging.error(f"Failed to load dataset from {dataset_path}: {e}")
            return

        all_results = []

        for item in tqdm(dataset.rag_questions, desc=f"Searching top-{k}"):
            res = retriever.search_single(item.question, item.question_id, k)
            all_results.append(res)

        final_output = StudentSearchResults(
            search_results=all_results,
            k=k
        )
        try:
            os.makedirs(save_directory, exist_ok=True)

            filename = os.path.basename(dataset_path)
            save_path = os.path.join(save_directory, filename)

            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(final_output.model_dump_json(indent=2))
        except Exception as e:
            logging.error(f"Failed to save results to {save_path}: {e}")
            return

        print(f"Saved student_search_results to {save_path}")

    def answer(self, query: str, k: int) -> None:
        """Answer a single query using the retrieved context."""
        logging.info(f"Answering query '{query}' using top {k} retrieved sources...")

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file."""
        logging.info(f"Generating answers for {student_search_results_path}. Saving to {save_directory}...")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        """Report your own recall@k against a ground-truth dataset.

        Args:
            student_search_results_path: Path to the StudentSearchResults file.
            dataset_path: Path to the RagDataset JSON file.
        """
        logging.info(f"Evaluating {student_search_results_path}"
                     f" against ground truth {dataset_path}...")
        evaluate_results(student_search_results_path, dataset_path)


if __name__ == '__main__':
    fire.Fire(RagCLI)
