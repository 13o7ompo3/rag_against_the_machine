import math
from tqdm import tqdm
from collections import Counter
from typing import List, Dict


class BM25:
    def __init__(self,
                 corpus: List[List[str]],
                 k1: float = 1.5, b: float = 0.75) -> None:
        """Initializes the BM25 model.

        Args:
            corpus: A list of documents, where each document is a list of tokens.
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)

        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(
            self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0

        self.doc_freqs = [Counter(doc) for doc in corpus]

        self.nd: Counter = Counter()
        for doc in tqdm(corpus, desc="Word counting for IDF calculation"):
            for word in set(doc):
                self.nd[word] += 1

        self.idf: Dict[str, float] = {}
        for word, freq in tqdm(self.nd.items(), desc="Calculating IDF"):
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

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
