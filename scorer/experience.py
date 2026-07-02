def experience_fit_score(candidate: dict, career_evidence: float) -> float:
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0))
    if 5.0 <= yoe <= 9.0:
        base = 1.0
    elif 4.0 <= yoe <5.0:
        base = 0.85
    elif 9 < yoe <= 12.0:
        base = 0.80
    elif 3.0 <= yoe < 4.0:
        base = 0.65
    elif yoe > 12.0:
        base = 0.70
    else:
        base = 0.30

    if career_evidence < 0.3:
        base *= 0.5
    return base

