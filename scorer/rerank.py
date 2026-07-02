import re
from scorer.bm25 import tokenize


def _term_overlap_features(jd_terms: set, cand_text: str) -> dict:
    cand_terms = set(tokenize(cand_text))
    overlap = jd_terms & cand_terms
    return {
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / max(len(jd_terms), 1),
    }


def _proximity_feature(jd_terms: set, cand_text: str, window: int = 8) -> float:
    tokens = tokenize(cand_text)
    positions: dict[str, list[int]] = {}
    for i, token in enumerate(tokens):
        if token in jd_terms:
            positions.setdefault(token, []).append(i)
    if len(positions) < 2:
        return 0.0

    hits = 0
    terms = list(positions.keys())
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            for pi in positions[terms[i]]:
                for pj in positions[terms[j]]:
                    if abs(pi - pj) <= window:
                        hits += 1
                        break
                else:
                    continue
                break
    return min(hits / 5.0, 1.0)


def rerank_score(
    jd_text: str, cand_text: str, dense_sim: float, bm25_norm: float
) -> float:
    jd_terms = set(tokenize(jd_text))
    feats = _term_overlap_features(jd_terms, cand_text)
    proximity = _proximity_feature(jd_terms, cand_text)
    score = (
        0.35 * max(dense_sim, 0.0)
        + 0.25 * bm25_norm
        + 0.25 * feats["overlap_ratio"]
        + 0.15 * proximity
    )
    return min(max(score, 0.0), 1.0)


def normalize_bm25(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]
