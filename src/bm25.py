import math
from collections import Counter
from typing import Dict, List

from tqdm import tqdm


class BM25:
    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initializes the BM25 model.

        Args:
            corpus: A list of documents,
                    where each document is a list of tokens.
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b

        self.corpus = [list(doc) for doc in corpus]
        self.doc_lens = [len(doc) for doc in self.corpus]
        self.doc_freqs = [Counter(doc) for doc in self.corpus]
        self.total_doc_len = sum(self.doc_lens)
        self.corpus_size = len(self.corpus)

        self.nd: Counter = Counter()
        self.idf: Dict[str, float] = {}

        for doc in tqdm(self.corpus, desc="Word counting for IDF calculation"):
            for word in set(doc):
                self.nd[word] += 1

        self._rebuild_idf()

    def _rebuild_idf(self) -> None:
        self.avgdl = (
            self.total_doc_len / self.corpus_size
            if self.corpus_size > 0
            else 0
        )

        self.idf = {}
        for word, freq in tqdm(self.nd.items(), desc="Calculating IDF"):
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def remove_documents(self, indices: List[int]) -> None:
        """Removes documents from the corpus based on their indices.

        Args:
            indices: A list of indices of the documents to be removed.
        """
        if not indices:
            return

        for idx in sorted(set(indices), reverse=True):
            doc = self.corpus.pop(idx)
            doc_len = self.doc_lens.pop(idx)
            self.doc_freqs.pop(idx)

            self.total_doc_len -= doc_len
            self.corpus_size -= 1

            for word in set(doc):
                self.nd[word] -= 1
                if self.nd[word] <= 0:
                    del self.nd[word]

        self._rebuild_idf()

    def add_documents(self, documents: List[List[str]]) -> None:
        """Adds documents to the corpus.

        Args:
            documents: A list of documents,
            where each document is a list of tokens.
        """
        if not documents:
            return

        for doc in documents:
            doc_copy = list(doc)
            doc_len = len(doc_copy)
            doc_freq = Counter(doc_copy)

            self.corpus.append(doc_copy)
            self.doc_lens.append(doc_len)
            self.doc_freqs.append(doc_freq)

            self.total_doc_len += doc_len
            self.corpus_size += 1

            for word in set(doc_copy):
                self.nd[word] += 1

        self._rebuild_idf()

    def get_scores(self, query: List[str]) -> List[float]:
        """Returns the BM25 score for every document given a query.

        Args:
            query: A list of tokens representing the query.

        Returns:
            A list of BM25 scores for each document in the corpus.
        """
        scores = [0.0] * self.corpus_size

        for q in tqdm(query, desc="Calculating BM25 scores"):
            if q not in self.idf:
                continue

            q_idf = self.idf[q]

            for idx, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq[q]
                if freq == 0:
                    continue

                doc_len = self.doc_lens[idx]

                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avgdl))

                scores[idx] += q_idf * (numerator / denominator)

        return scores
