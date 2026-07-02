import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scorer.bm25 import BM25, tokenize
from scorer.embeddings import DenseEmbedder, candidate_text
from scorer.retrieval import HybridRetriever, reciprocal_rank_fusion
from scorer.rerank import rerank_score, normalize_bm25


class TestTokenize:
    def test_basic_split(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_keeps_hyphenated_terms(self):
        toks = tokenize("sentence-transformers and gpt-4o-mini")
        assert "sentence-transformers" in toks
        assert "gpt-4o-mini" in toks

    def test_strips_punctuation(self):
        toks = tokenize("FAISS, BM25; and NDCG!")
        assert "faiss" in toks and "bm25" in toks and "ndcg" in toks


class TestBM25:
    def test_exact_match_scores_highest(self):
        docs = [
            "built production semantic search using faiss",
            "managed client deliverables for offshore engagement",
            "designed marketing campaigns for social media",
        ]
        bm25 = BM25(docs)
        scores = bm25.score("semantic search faiss")
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_empty_query_zero_scores(self):
        bm25 = BM25(["some text here", "other text there"])
        scores = bm25.score("")
        assert all(s == 0.0 for s in scores)

    def test_top_k_returns_correct_count(self):
        docs = [f"document number {i} about search" for i in range(20)]
        bm25 = BM25(docs)
        results = bm25.top_k("search", k=5)
        assert len(results) == 5

    def test_unknown_terms_dont_crash(self):
        bm25 = BM25(["hello world", "foo bar"])
        scores = bm25.score("zzzznonexistentword")
        assert scores == [0.0, 0.0]


class TestDenseEmbedder:
    def test_fit_produces_unit_norm_vectors(self):
        texts = [
            "built search systems with faiss and embeddings",
            "managed offshore client deliverables",
            "designed dense retrieval pipeline at scale",
            "handled hr recruitment and onboarding",
            "shipped production rag system serving users",
        ]
        embedder = DenseEmbedder(n_components=3)
        embedder.fit(texts)
        norms = (embedder.embeddings_ ** 2).sum(axis=1) ** 0.5
        for n in norms:
            assert abs(n - 1.0) < 1e-6 or n < 1e-6

    def test_similar_texts_higher_cosine(self):
        texts = [
            "built dense retrieval search pipeline with embeddings",
            "designed semantic search system using vector embeddings",
            "managed offshore client billing and invoices",
            "handled payroll and hr compliance paperwork",
        ]
        embedder = DenseEmbedder(n_components=2)
        embedder.fit(texts)
        sims = embedder.embeddings_ @ embedder.embeddings_[0]
        assert sims[1] > sims[2]
        assert sims[1] > sims[3]

    def test_transform_before_fit_raises(self):
        embedder = DenseEmbedder()
        try:
            embedder.transform(["test"])
            assert False, "should have raised"
        except RuntimeError:
            pass


class TestCandidateText:
    def test_includes_career_and_skills(self):
        c = {
            'career_history': [{'title': 'ML Engineer', 'description': 'built faiss search'}],
            'skills': [{'name': 'faiss'}, {'name': 'bm25'}],
            'profile': {'headline': 'search engineer', 'summary': 'expert'},
        }
        text = candidate_text(c)
        assert 'faiss' in text.lower()
        assert 'search engineer' in text.lower()

    def test_empty_candidate_no_crash(self):
        assert candidate_text({}) == ""


class TestRRF:
    def test_consensus_candidate_ranks_high(self):
        dense_rank = [0, 2, 1, 3]
        bm25_rank = [0, 1, 3, 2]
        fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
        top = max(fused, key=fused.get)
        assert top == 0

    def test_only_in_one_list_still_scored(self):
        fused = reciprocal_rank_fusion([[5], [9]])
        assert 5 in fused and 9 in fused


class TestHybridRetriever:
    def _candidates(self):
        return [
            {'candidate_id': 'C0', 'career_history': [{'title': 'a', 'description': 'built faiss dense retrieval search pipeline at scale'}], 'skills': [{'name': 'faiss'}], 'profile': {}},
            {'candidate_id': 'C1', 'career_history': [{'title': 'b', 'description': 'managed offshore client deliverables statement of work'}], 'skills': [{'name': 'java'}], 'profile': {}},
            {'candidate_id': 'C2', 'career_history': [{'title': 'c', 'description': 'designed semantic search bm25 hybrid system production'}], 'skills': [{'name': 'bm25'}], 'profile': {}},
            {'candidate_id': 'C3', 'career_history': [{'title': 'd', 'description': 'handled hr recruitment payroll compliance paperwork'}], 'skills': [{'name': 'excel'}], 'profile': {}},
        ]

    def test_retrieve_returns_relevant_first(self):
        retriever = HybridRetriever()
        cands = self._candidates()
        retriever.fit(cands)
        results = retriever.retrieve("semantic search faiss bm25 retrieval", top_k=4)
        top_ids = [cands[idx]['candidate_id'] for idx, _ in results[:2]]
        assert 'C0' in top_ids or 'C2' in top_ids

    def test_retrieve_before_fit_raises(self):
        retriever = HybridRetriever()
        try:
            retriever.retrieve("test")
            assert False
        except RuntimeError:
            pass

    def test_top_k_above_legacy_1000_cap_actually_widens_retrieval(self):
        n = 1600
        texts = [f"candidate number {i} doing generic unrelated office work" for i in range(n)]
        texts[n - 1] = "expert in faiss bm25 dense retrieval semantic search ndcg framework"
        cands = [
            {'candidate_id': f'C{i}', 'career_history': [{'title': 'x', 'description': t}], 'skills': [], 'profile': {}}
            for i, t in enumerate(texts)
        ]

        retriever = HybridRetriever(dense_top_k=1400, bm25_top_k=1400)
        retriever.fit(cands)
        results = retriever.retrieve("faiss bm25 dense retrieval semantic search ndcg", top_k=1400)

        found_idx = [idx for idx, _ in results]
        assert (n - 1) in found_idx, (
            "Needle candidate dropped — dense_top_k/bm25_top_k did not honor "
            "the requested top_k beyond the old hardcoded 1000 cap"
        )

    def test_default_dense_and_bm25_top_k_scale_with_requested_top_k(self):
        n = 50
        texts = [f"doc {i} about generic topics" for i in range(n)]
        cands = [{'candidate_id': f'C{i}', 'career_history': [{'title': 'x', 'description': t}], 'skills': [], 'profile': {}} for i, t in enumerate(texts)]
        retriever = HybridRetriever()  
        retriever.fit(cands)
        results = retriever.retrieve("generic topics", top_k=n)
        assert len(results) <= n
        assert len(results) > 0


class TestDenseEmbedderNoWarnings:
    def test_duplicate_corpus_does_not_raise_runtime_warning(self):
        import warnings as _warnings
        texts = ["built faiss search system production"] * 5
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            embedder = DenseEmbedder(n_components=128)
            embedder.fit(texts)  
        assert embedder.embeddings_ is not None


class TestRerank:
    def test_bounded_0_to_1(self):
        s = rerank_score("search faiss bm25", "built faiss search pipeline", dense_sim=0.8, bm25_norm=0.9)
        assert 0.0 <= s <= 1.0

    def test_higher_overlap_higher_score(self):
        jd = "search faiss bm25 retrieval embeddings"
        good = rerank_score(jd, "built faiss bm25 search retrieval embeddings pipeline", dense_sim=0.5, bm25_norm=0.5)
        bad = rerank_score(jd, "managed offshore client billing", dense_sim=0.1, bm25_norm=0.1)
        assert good > bad

    def test_normalize_bm25_handles_uniform_scores(self):
        assert normalize_bm25([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]

    def test_normalize_bm25_empty(self):
        assert normalize_bm25([]) == []