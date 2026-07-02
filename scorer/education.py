TIER_SCORES = {
    "tier_1": 1,
    "tier_2": 0.80,
    "tier_3": 0.60,
    "tier_4": 0.40,
    "unknown": 0.55,
}

GOOD_FIELDS = {
    "computer science",
    "information technology",
    "electronics",
    "electrical",
    "ai",
    "machine learning",
    "data science",
    "data analytics",
    "statistics",
    "mathematics",
    "information systems",
}

BAD_FIELDS = {
    "mechanical",
    "civil",
    "chemical",
    "arts",
    "commerce",
    "law",
}


def education_score(candidate: dict) -> float:
    edu = candidate.get("education", [])
    if not edu:
        base = 0.50
    else:
        best = max(edu, key=lambda e: TIER_SCORES.get(e.get("tier", "unknown"), 0.55))
        tier_s = TIER_SCORES.get(best.get("tier", "unknown"), 0.55)
        field = best.get("field_of_study", "").lower()
        if any(f in field for f in GOOD_FIELDS):
            field_s = 1.00
        elif any(f in field for f in BAD_FIELDS):
            field_s = 0.40
        else:
            field_s = 0.65

        base = tier_s * 0.6 + field_s * 0.4

    gh = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if gh > 60:
        base = min(base * 1.15, 1.0)
    elif gh == -1:
        pass
    elif gh < 10:
        base *= 0.95
    return base
