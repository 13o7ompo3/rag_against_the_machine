import fire
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class RagCLI:
    """Command-Line Interface for the RAG pipeline."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        logging.info(f"Running indexer with max_chunk_size={max_chunk_size}...")

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query."""
        logging.info(f"Searching for '{query}' returning top {k} results...")

    def search_dataset(self, dataset_path: str, k: int, save_directory: str) -> None:
        """Run search over a whole dataset and write a StudentSearchResults JSON file."""
        logging.info(f"Batch searching dataset {dataset_path} (k={k}). Saving to {save_directory}...")

    def answer(self, query: str, k: int) -> None:
        """Answer a single query using the retrieved context."""
        logging.info(f"Answering query '{query}' using top {k} retrieved sources...")

    def answer_dataset(self, student_search_results_path: str, save_directory: str) -> None:
        """Generate answers for a dataset, producing a StudentSearchResultsAndAnswer JSON file."""
        logging.info(f"Generating answers for {student_search_results_path}. Saving to {save_directory}...")

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        """Report your own recall@k against a ground-truth dataset."""
        logging.info(f"Evaluating {student_search_results_path} against ground truth {dataset_path}...")

if __name__ == '__main__':
    fire.Fire(RagCLI)
