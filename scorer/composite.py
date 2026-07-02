from datetime import date

from scorer.filters import should_eliminate
from scorer.career import career_evidence_score, extract_best_evidence_sentence
from scorer.skills import skills_trust_score, TIER1_SKILLS
from scorer.experience import experience_fit_score
from scorer.location import location_score
from scorer.education import education_score
from scorer.behavioural import behavioural_multiplier
from scorer.embeddings import candidate_text
from scorer.retrieval import HybridRetriever
from scorer.rerank import rerank_score, normalize_bm25


def score_candidate(candidate: dict) -> dict | None:
    if not candidate.get("candidate_id"):
        return None
    if candidate.get("redrob_signals") is None:
        candidate = {**candidate, "redrob_signals": {}}
    eliminated, reason = should_eliminate(candidate)
    if eliminated:
        return None

    A = career_evidence_score(candidate)
    B = skills_trust_score(candidate)
    C = experience_fit_score(candidate, A)
    D = location_score(candidate)
    E = education_score(candidate)

    profile_score = A * 0.40 + B * 0.25 + C * 0.15 + D * 0.12 + E * 0.08

    mult = behavioural_multiplier(candidate.get("redrob_signals", {}), date.today())
    raw = profile_score * mult
    final = min(raw, 1.0)

    return {
        "candidate_id": candidate["candidate_id"],
        "score": final,
        "raw_score": raw,
        "dimensions": {"A": A, "B": B, "C": C, "D": D, "E": E},
        "multiplier": mult,
        "candidate": candidate,
    }


def reasoning_string(result: dict) -> str:
    from datetime import datetime

    c = result["candidate"]
    p = c.get("profile", {})
    sig = c.get("redrob_signals", {})

    title = p.get("current_title", "Engineer")
    yoe = p.get("years_of_experience", 0)

    career_lead = extract_best_evidence_sentence(c.get("career_history", []))

    if not career_lead:
        top_skills = [
            s["name"]
            for s in c.get("skills", [])
            if s.get("name", "").lower() in TIER1_SKILLS
            and s.get("proficiency") in ("advanced", "expert")
            and s.get("duration_months", 0) > 0
        ][:2]
        career_lead = (
            ", ".join(top_skills) + " practitioner" if top_skills else "ML background"
        )

    raw_date = sig.get("last_active_date")
    days = 999
    if raw_date:
        try:
            last = datetime.fromisoformat(raw_date).date()
            days = (date.today() - last).days
        except (ValueError, TypeError):
            pass

    rr = sig.get("recruiter_response_rate", 0.0)
    np_d = sig.get("notice_period_days", 60)

    if days <= 7 and rr > 0.60:
        beh = f"active {days}d ago, {rr:.0%} response rate"
    elif np_d <= 30:
        beh = f"available in {np_d}d notice"
    elif rr < 0.15:
        beh = f"low response rate ({rr:.0%}) — follow-up advised"
    else:
        beh = f"{days}d since last login, {np_d}d notice"

    return f"{title} ({yoe:.1f} yrs): {career_lead}; {beh}."


SEMANTIC_WEIGHT = 0.5

RULE_BASED_WEIGHT = 0.5


def score_candidates_v2(candidates: list[dict], jd_text: str, top_k_retrieve: int = 2000) -> list[dict]:
    survivors = []
    for c in candidates:
        if not c.get("candidate_id"):
            continue
        if c.get("redrob_signals") is None:
            c = {**c, "redrob_signals": {}}
        eliminated, _ = should_eliminate(c)
        if not eliminated:
            survivors.append(c)

    if not survivors:
        return []

    effective_top_k = min(top_k_retrieve, len(survivors))
    retriever = HybridRetriever(dense_top_k=effective_top_k, bm25_top_k=effective_top_k)
    retriever.fit(survivors)
    retrieved = retriever.retrieve(jd_text, top_k=effective_top_k)
    retrieved_idx = [idx for idx, _ in retrieved]

    bm25_raw = retriever.bm25.score(jd_text)
    bm25_for_retrieved = [bm25_raw[i] for i in retrieved_idx]
    bm25_norm = normalize_bm25(bm25_for_retrieved)
    bm25_norm_by_idx = dict(zip(retrieved_idx, bm25_norm))

    query_vec = retriever.embedder.transform([jd_text])[0]
    dense_sims = retriever.embedder.embeddings_ @ query_vec

    results = []
    for idx in retrieved_idx:
        candidate = survivors[idx]
        c_text = candidate_text(candidate)

        semantic_score = rerank_score(
            jd_text=jd_text,
            cand_text=c_text,
            dense_sim=float(dense_sims[idx]),
            bm25_norm=bm25_norm_by_idx.get(idx, 0.0),
        )

        A = career_evidence_score(candidate)
        B = skills_trust_score(candidate)
        C = experience_fit_score(candidate, A)
        D = location_score(candidate)
        E = education_score(candidate)

        rule_score = A * 0.40 + B * 0.25 + C * 0.15 + D * 0.12 + E * 0.08

        blended_profile = SEMANTIC_WEIGHT * semantic_score + RULE_BASED_WEIGHT * rule_score

        mult = behavioural_multiplier(candidate.get("redrob_signals", {}), date.today())
        raw = blended_profile * mult
        final = min(raw, 1.0)

        results.append({
            "candidate_id": candidate["candidate_id"],
            "score": final,
            "raw_score": raw,
            "semantic_score": semantic_score,
            "rule_score": rule_score,
            "dimensions": {"A": A, "B": B, "C": C, "D": D, "E": E},
            "multiplier": mult,
            "candidate": candidate,
        })

    
    results.sort(key=lambda x: -x["raw_score"])

    
    max_raw = max((r["raw_score"] for r in results), default=1.0)
    rescale = max_raw if max_raw > 1.0 else 1.0
    for r in results:
        r["display_score"] = r["raw_score"] / rescale

    return results


def reasoning_string_v2(result: dict, jd_text: str) -> str:
    
    from scorer.composite import reasoning_string

    base = reasoning_string({
        "candidate": result["candidate"],
        "dimensions": result["dimensions"],
        "multiplier": result["multiplier"],
        "score": result["rule_score"],
    })

    base = base.rstrip(".")
    return f"{base}; semantic relevance to JD: {result['semantic_score']:.2f}."