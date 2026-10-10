import fire
import logging
import json
import uuid
import os
from tqdm import tqdm

from .evaluator import evaluate_results
from .search_index import SearchIndex
from .models import StudentSearchResults, RagDataset, MinimalSearchResults

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
)


class RagCLI:
    """Command-Line Interface for the RAG pipeline."""

    def _make_index(self, backend: str) -> SearchIndex:
        if backend == 'bm25':
            from .retriever import BM25Index
            return BM25Index()
        if backend == 'semantic':
            from .semantic import SemanticIndex
            return SemanticIndex()
        if backend == 'hybrid':
            from .hybrid import HybridIndex
            from .retriever import BM25Index
            from .semantic import SemanticIndex
            return HybridIndex(BM25Index(), SemanticIndex())
        raise ValueError(f"Unsupported backend: {backend}")

    def index(self, backend: str = 'bm25', max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build an index under data/processed/."""

        logging.info(
            f"Running {backend} indexer with max_chunk_size={max_chunk_size}."
        )
        try:
            index = self._make_index(backend)
            index.build_from_raw("data/raw/", "data/processed/",
                                 max_chunk_size)
        except Exception as e:
            logging.error(f"Failed to build {backend} index: {e}")

    def search(self, query: str, k: int, backend: str = 'bm25') -> None:
        """Return the top-k sources for a single query."""
        logging.info(
            f"Searching with {backend} for '{query}' "
            f"returning top {k} results..."
        )
        try:
            index = self._make_index(backend)
            index.load("data/processed/")
        except Exception as e:
            logging.error(f"Failed to initialize {backend} index: {e}")
            return
        result = index.search(
            query,
            k,
            question_id=str(uuid.uuid4()),
        )
        index._save_cache()
        print(result.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        backend: str = 'bm25',
    ) -> None:
        """Run search over a whole dataset
        and write a StudentSearchResults JSON file.

        Args:
            dataset_path: Path to the RagDataset JSON file.
            k: Number of top results to return for each query.
            save_directory: Directory to save the StudentSearchResults.
        """
        logging.info(
            f"Batch searching dataset {dataset_path} (k={k}, "
            f"backend={backend}). Saving to {save_directory}..."
        )

        try:
            index = self._make_index(backend)
            index.load("data/processed/")
        except Exception as e:
            logging.error(f"Failed to initialize {backend} index: {e}")
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
            try:
                res = index.search(
                    item.question,
                    k,
                    question_id=item.question_id,
                )
                all_results.append(res)
            except Exception as e:
                logging.warning(
                    f"Search failed for question_id {item.question_id}: {e}. "
                    "Appending empty results."
                )
                all_results.append(MinimalSearchResults(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=[]
                ))
        index._save_cache()
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
        logging.info(
            f"Answering query '{query}' using top {k} retrieved sources..."
        )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
    ) -> None:
        """Generate answers for a dataset and save the answer file."""
        logging.info(
            f"Generating answers for {student_search_results_path}. "
            f"Saving to {save_directory}..."
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Report your own recall@k against a ground-truth dataset.

        Args:
            student_search_results_path: Path to the StudentSearchResults file.
            dataset_path: Path to the RagDataset JSON file.
        """
        logging.info(
            f"Evaluating {student_search_results_path}"
            f" against ground truth {dataset_path}..."
        )
        evaluate_results(student_search_results_path, dataset_path)


if __name__ == '__main__':
    fire.Fire(RagCLI)
