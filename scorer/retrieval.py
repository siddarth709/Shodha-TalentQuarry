from scorer.embeddings import DenseEmbedder, candidate_text
from scorer.bm25 import BM25

def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rank_lists:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused

class HybridRetriever:
    def __init__(self, n_components: int = 128, dense_top_k: int | None = None, bm25_top_k: int | None = None):
        self._dense_top_k_override = dense_top_k
        self._bm25_top_k_override = bm25_top_k
        self.embedder = DenseEmbedder(n_components=n_components)
        self.bm25 = None
        self.candidates: list[dict] = []
        self._fitted = False

    def fit(self, candidates: list[dict]) -> HybridRetriever:
        self.candidates = candidates
        texts = [candidate_text(c) for c in candidates]
        self.embedder.fit(texts)
        self.bm25 = BM25(texts)
        self._fitted = True
        return self

    def retrieve(self, query_text: str, top_k: int = 500) -> list[tuple]:
        if not self._fitted:
            raise RuntimeError("HybridRetriever.fit() must be called before retrieve()")

        n = len(self.candidates)
        dense_top_k = self._dense_top_k_override or min(max(top_k, 1000), n)
        bm25_top_k = self._bm25_top_k_override or min(max(top_k, 1000), n)

        query_vec = self.embedder.transform([query_text])[0]
        dense_sims = self.embedder.embeddings_ @ query_vec
        dense_rank = list(dense_sims.argsort()[::-1][:dense_top_k])

        bm25_rank = [i for i, _ in self.bm25.top_k(query_text, k=bm25_top_k)]

        fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return ranked
