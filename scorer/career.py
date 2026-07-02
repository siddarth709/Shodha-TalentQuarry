import json, os
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

with open(os.path.join(_DATA_DIR, 'skill_taxonomy.json')) as f:
    _tax = json.load(f)
with open(os.path.join(_DATA_DIR, 'career_evidence_keywords.json')) as f:
    _kw = json.load(f)

CONSULTING_FIRMS = set(_tax['consulting_firms'])
TIER1_EVIDENCE = _kw['tier1_evidence']
CONSULTING_ANTI = _kw['consulting_anti_signals']

def career_evidence_score(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    if not history:
        return 0.0

    total_months = sum(r.get("duration_months", 0) for r in history) or 1
    consulting_months = 0
    weighted_score = 0.0

    for role in history:
        desc = role.get("description", "").lower()
        dur = role.get("duration_months", 0)
        weight = dur / total_months
        company_low = role.get("company", "").lower()

        tier1_hits = sum(1 for kw in TIER1_EVIDENCE if kw in desc)
        anti_hits = sum(1 for kw in CONSULTING_ANTI if kw in desc)
        role_score = min(tier1_hits/6.0, 1.0)

        if anti_hits >= 3 and tier1_hits < 2:
            role_score *= 0.25

        if any(cf in company_low for cf in CONSULTING_FIRMS):
            role_score *= 0.30
            consulting_months += dur
        weighted_score += role_score * weight
    if consulting_months / total_months > 0.8:
        weighted_score *= 0.3
    return min(weighted_score, 1.0)

def extract_best_evidence_sentence(history: list) -> str:
    best, best_score = "", 0.0
    for role in history:
        desc = role.get('description', '')
        for sentence in desc.split('.'):
            s = sentence.lower()
            hits = sum(1 for kw in TIER1_EVIDENCE if kw in s)
            if hits > best_score and len(sentence.strip()) > 25:
                best_score = hits
                best = sentence.strip()
    if not best:
        return ''
    if len(best) > 150:
        truncated = best[:146]
        last_space = truncated.rfind(' ')
        if last_space > 75:
            return best[:last_space] + " ..."
        return truncated + " ..."
    return best

