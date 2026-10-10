from .search_index import SearchIndex
from .models import MinimalSearchResults, MinimalSource


class HybridIndex(SearchIndex):
    """A hybrid index that combines BM25 and semantic search."""

    def __init__(self, bm25_index: SearchIndex, semantic_index: SearchIndex):
        """Initialize the hybrid index with BM25 and semantic indices.

        Args:
            bm25_index (SearchIndex): An instance of a BM25 index.
            semantic_index (SearchIndex): An instance of a semantic index.
        """
        self.bm25_index = bm25_index
        self.semantic_index = semantic_index

    def build_from_raw(
        self,
        raw_dir: str,
        processed_dir: str,
        max_chunk_size: int = 2000,
    ) -> None:
        """Build both BM25 and semantic indices from raw data.

        Args:
            raw_dir (str): directory containing the raw files to be indexed.
            processed_dir (str): directory where the indexing will be stored.
            max_chunk_size (int): maximum size of each chunk.
        """
        self.bm25_index.build_from_raw(raw_dir, processed_dir, max_chunk_size)
        self.semantic_index.build_from_raw(
            raw_dir, processed_dir, max_chunk_size)

    def load(self, processed_dir: str) -> None:
        """Load both BM25 and semantic indices from disk.

        Args:
            processed_dir (str): The directory where the indices are stored.
        """
        self.bm25_index.load(processed_dir)
        self.semantic_index.load(processed_dir)

    def save(self, processed_dir: str) -> None:
        """Save both BM25 and semantic indices to disk.

        Args:
            processed_dir (str): The directory where the indices will be saved.
        """
        self.bm25_index.save(processed_dir)
        self.semantic_index.save(processed_dir)

    def search(
        self,
        query: str,
        k: int,
        question_id: str,
    ) -> MinimalSearchResults:
        """Perform a hybrid search using both BM25 and semantic search.

        Args:
            query (str): The search query.
            k (int): The number of top results to return.
            question_id (str): The ID of the question being searched.

        Returns:
            MinimalSearchResults: The search results.
        """
        bm25_results = self.bm25_index.search(query, k, question_id)
        semantic_results = self.semantic_index.search(query, k, question_id)

        bm25_sources = [Source(**x.model_dump())
                        for x in bm25_results.retrieved_sources]
        semantic_sources = [Source(**x.model_dump())
                            for x in semantic_results.retrieved_sources]

        merged = merge_by_weighted_rank(bm25_sources, semantic_sources,
                                        1.0, 1.0)

        return MinimalSearchResults(
            question_id=question_id,
            question=query,
            retrieved_sources=merged[:k],  # type: ignore[arg-type]
        )


class Source(MinimalSource):
    """A hashable version of MinimalSource for use in sets and dictionaries."""

    def __eq__(self, other: object) -> bool:
        """Check equality based on file_path and character indices.

        Args:
            other (object): The other object to compare.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, MinimalSource):
            return False
        return (self.file_path == other.file_path and
                self.first_character_index == other.first_character_index and
                self.last_character_index == other.last_character_index)

    def __hash__(self) -> int:
        """Compute a hash based on file_path and character indices.

        Returns:
            int: The computed hash value.
        """
        return hash((self.file_path, self.first_character_index,
                     self.last_character_index))


def merge_by_weighted_rank(list1: list[Source], list2: list[Source],
                           weight1: float, weight2: float) -> list[Source]:
    """Merge two lists of sources based on weighted rank.

    Args:
        list1 (list[Source]): The first list of sources.
        list2 (list[Source]): The second list of sources.
        weight1 (float): The weight for the first list.
        weight2 (float): The weight for the second list.

    Returns:
        list[Source]: The merged list of sources sorted by weighted rank.
    """
    unique_items = list(set(list1) | set(list2))

    def calculate_score(item: Source) -> float:
        score = 0.0
        if item in list1:
            score += list1.index(item) * weight1
        else:
            score += len(list1) * weight1

        if item in list2:
            score += list2.index(item) * weight2
        else:
            score += len(list2) * weight2

        return score

    return sorted(unique_items, key=calculate_score)
