from abc import ABC, abstractmethod

from src.models import MinimalSearchResults


class SearchIndex(ABC):
    """Common interface for searchable index backends."""

    @abstractmethod
    def build_from_raw(
        self,
        raw_dir: str,
        processed_dir: str,
        max_chunk_size: int = 2000,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, processed_dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self, processed_dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        k: int,
        question_id: str,
    ) -> MinimalSearchResults:
        raise NotImplementedError
