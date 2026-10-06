from abc import ABC, abstractmethod

from .models import MinimalSearchResults, MinimalSource


class SearchIndex(ABC):
    """Common interface for searchable index backends."""

    def __init__(self) -> None:
        self.chunks_metadata: list[MinimalSource] = []
        self.manifest: dict[str, str] = {}
        self.max_chunk_size: int = 2000

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
