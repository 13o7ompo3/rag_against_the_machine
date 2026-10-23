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

        self.docs_lens = [len(doc) for doc in corpus]
        self.avg_len = sum(self.docs_lens) / len(self.docs_lens)
        self.docs = [Counter(doc) for doc in corpus]
        self.nd = Counter(word for doc in self.docs for word in doc)
        self.idf: Dict[str, float] = {}

        self._rebuild_idf()

    def _rebuild_idf(self) -> None:
        if self.docs_lens:
            self.avg_len = sum(self.docs_lens) / len(self.docs_lens)
        else:
            self.avg_len = 0.0

        doc_len = len(self.docs_lens)
        self.idf = {}
        for word, freq in tqdm(self.nd.items(), desc="Calculating IDF"):
            self.idf[word] = math.log(
                (doc_len - freq + 0.5) / (freq + 0.5) + 1.0)

    def remove_documents(self, indices: List[int]) -> None:
        """Removes documents from the corpus based on their indices.

        Args:
            indices: A list of indices of the documents to be removed.
        """
        if not indices:
            return

        sorted_indices = sorted(indices, reverse=True)
        del_docs: list[Counter[str]] = []
        for idx in sorted_indices:
            self.docs_lens.pop(idx)
            del_docs.append(self.docs.pop(idx))

        self.nd.subtract(word for doc in del_docs for word in doc)
        self.nd += Counter()

    def add_documents(self, corpus: List[List[str]]) -> None:
        """Adds documents to the corpus.

        Args:
            documents: A list of documents,
            where each document is a list of tokens.
        """
        if not corpus:
            return

        new_lens = [len(doc) for doc in corpus]
        new_docs = [Counter(doc) for doc in corpus]

        self.docs_lens.extend(new_lens)
        self.docs.extend(new_docs)
        self.nd.update(word for doc in new_docs for word in doc)

    def get_scores(self, query: List[str]) -> List[float]:
        """Returns the BM25 score for every document given a query.

        Args:
            query: A list of tokens representing the query.

        Returns:
            A list of BM25 scores for each document in the corpus.
        """
        scores = [0.0] * len(self.docs_lens)

        for q in tqdm(query, desc="Calculating BM25 scores"):
            if q not in self.idf:
                continue

            q_idf = self.idf[q]

            for idx, doc_freq in enumerate(self.docs):
                freq = doc_freq[q]
                if freq == 0:
                    continue

                doc_len = self.docs_lens[idx]

                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avg_len))

                scores[idx] += q_idf * (numerator / denominator)

        return scores
