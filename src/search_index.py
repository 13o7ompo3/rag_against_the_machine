from abc import ABC, abstractmethod
import logging
from .models import MinimalSearchResults, MinimalSource

logger = logging.getLogger(__name__)


class SearchIndex(ABC):
    """Common interface for searchable index backends."""

    def __init__(self) -> None:
        """Initialize empty index state and query cache."""
        self.chunks_metadata: list[MinimalSource] = []
        self.manifest: dict[str, str] = {}
        self.max_chunk_size: int = 2000
        self.query_cache: dict[tuple[str, int], list[MinimalSource]] = {}
        self.processed_dir: str | None = None

    def _save_cache(self) -> None:
        """Serialize and persist HMAC-signed query cache to disk."""
        if not (
            self.processed_dir
            and hasattr(self, 'env_key_name')
            and isinstance(self.env_key_name, str)
        ):
            return

        from dotenv import dotenv_values
        import os
        import hmac
        import pickle
        from hashlib import sha512

        env = dotenv_values('.env')
        key_hex = env.get(self.env_key_name)
        if not key_hex:
            return

        try:
            key_bytes = bytes.fromhex(key_hex)
            cache_bytes = pickle.dumps(self.query_cache)
            cache_hash = hmac.new(key_bytes, cache_bytes, sha512).hexdigest()

            cache_path = os.path.join(
                self.processed_dir,
                f"{self.__class__.__name__.lower()}_cache.pkl",
            )
            tmp_path = cache_path + '.tmp'
            with open(tmp_path, 'wb') as f:
                f.write(cache_hash.encode('utf-8'))
                f.write(cache_bytes)
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            logger.warning(
                f"Failed to save query cache: {exc}"
            )

    def _load_cache(self) -> None:
        """Load and verify HMAC-signed query cache from disk."""
        if not (
            self.processed_dir
            and hasattr(self, 'env_key_name')
            and isinstance(self.env_key_name, str)
        ):
            return

        from dotenv import dotenv_values
        import os
        import hmac
        import pickle
        from hashlib import sha512

        cache_path = os.path.join(
            self.processed_dir,
            f"{self.__class__.__name__.lower()}_cache.pkl",
        )
        if not os.path.exists(cache_path):
            self.query_cache = {}
            return

        env = dotenv_values('.env')
        key_hex = env.get(self.env_key_name)
        if not key_hex:
            self.query_cache = {}
            return

        try:
            key_bytes = bytes.fromhex(key_hex)
            with open(cache_path, 'rb') as f:
                content = f.read()

            if len(content) < 128:
                logger.warning(
                    "Query cache header too short; resetting cache."
                )
                self.query_cache = {}
                return

            stored_hash = content[:128].decode('utf-8')
            cache_bytes = content[128:]

            calculated_hash = hmac.new(
                key_bytes,
                cache_bytes,
                sha512,
            ).hexdigest()
            if not hmac.compare_digest(calculated_hash, stored_hash):
                logger.warning(
                    "Query cache HMAC verification failed; resetting cache."
                )
                self.query_cache = {}
                return

            self.query_cache = pickle.loads(cache_bytes)
        except Exception as exc:
            logger.warning(
                f"Failed to load query cache: {exc}; resetting cache."
            )
            self.query_cache = {}

    @abstractmethod
    def build_from_raw(
        self,
        raw_dir: str,
        processed_dir: str,
        max_chunk_size: int = 2000,
    ) -> None:
        """Build the index from raw documents.

        Args:
            raw_dir: Path to raw documents directory.
            processed_dir: Directory where index files will be stored.
            max_chunk_size: Maximum character length per chunk.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, processed_dir: str) -> None:
        """Load the index and integrity metadata from disk.

        Args:
            processed_dir: Directory containing stored index files.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, processed_dir: str) -> None:
        """Serialize and persist the index and integrity metadata to disk.

        Args:
            processed_dir: Directory to save index files into.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        k: int,
        question_id: str,
    ) -> MinimalSearchResults:
        """Perform search to retrieve top-k matching sources for a query.

        Args:
            query: The search query string.
            k: Number of top results to return.
            question_id: Unique identifier for the question.

        Returns:
            MinimalSearchResults containing retrieved sources.
        """
        raise NotImplementedError
