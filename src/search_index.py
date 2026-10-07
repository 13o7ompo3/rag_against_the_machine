from abc import ABC, abstractmethod

from .models import MinimalSearchResults, MinimalSource


class SearchIndex(ABC):
    """Common interface for searchable index backends."""

    def __init__(self) -> None:
        self.chunks_metadata: list[MinimalSource] = []
        self.manifest: dict[str, str] = {}
        self.max_chunk_size: int = 2000
        self.query_cache: dict[tuple[str, int], list[MinimalSource]] = {}
        self.processed_dir: str | None = None

    def _save_cache(self) -> None:
        if not self.processed_dir or not getattr(self, 'env_key_name', None):
            return
        
        from dotenv import load_dotenv
        import os
        import hmac
        import pickle
        from hashlib import sha512
        
        load_dotenv(override=True)
        key_hex = os.getenv(self.env_key_name) # type: ignore
        if not key_hex:
            return
            
        try:
            key_bytes = bytes.fromhex(key_hex)
            cache_bytes = pickle.dumps(self.query_cache)
            cache_hash = hmac.new(key_bytes, cache_bytes, sha512).hexdigest()
            
            cache_path = os.path.join(self.processed_dir, f"{self.__class__.__name__.lower()}_cache.pkl")
            with open(cache_path, 'wb') as f:
                f.write(cache_hash.encode('utf-8'))
                f.write(cache_bytes)
        except Exception:
            pass

    def _load_cache(self) -> None:
        if not self.processed_dir or not getattr(self, 'env_key_name', None) or not isinstance(self.env_key_name, str):
            return
            
        from dotenv import load_dotenv
        import os
        import hmac
        import pickle
        from hashlib import sha512
        
        cache_path = os.path.join(self.processed_dir, f"{self.__class__.__name__.lower()}_cache.pkl")
        if not os.path.exists(cache_path):
            self.query_cache = {}
            return
            
        load_dotenv(override=True)
        key_hex = os.getenv(self.env_key_name)
        if not key_hex:
            self.query_cache = {}
            return

        try:
            key_bytes = bytes.fromhex(key_hex)
            with open(cache_path, 'rb') as f:
                content = f.read()

            if len(content) < 128:
                self.query_cache = {}
                return

            stored_hash = content[:128].decode('utf-8')
            cache_bytes = content[128:]

            calculated_hash = hmac.new(key_bytes, cache_bytes, sha512).hexdigest()
            if not hmac.compare_digest(calculated_hash, stored_hash):
                self.query_cache = {}
                return

            self.query_cache = pickle.loads(cache_bytes)
        except Exception:
            self.query_cache = {}

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
