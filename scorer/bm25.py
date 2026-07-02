import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\+\.]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.doc_tokens: list[list[str]] = [tokenize(d) for d in documents]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0
        self.tf: list[Counter] = [Counter(tokens) for tokens in self.doc_tokens]
        df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] += 1
        self.df = df
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        scores = [0.0] * self.N
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None or idf <= 0:
                continue
            for i in range(self.N):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / (denom or 1)
        return scores

    def top_k(self, query: str, k: int = 200) -> list[tuple]:
        scores = self.score(query)
        ranked = sorted(range(self.N), key=lambda i: -scores[i])[:k]
        return [(i, scores[i]) for i in ranked]
