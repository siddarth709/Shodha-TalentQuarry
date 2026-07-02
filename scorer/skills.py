import json
import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

with open(os.path.join(_DATA_DIR, "skill_taxonomy.json")) as f:
    _tax = json.load(f)

TIER1_SKILLS = set(_tax["tier1_skills"])
TIER2_SKILLS = set(_tax["tier2_skills"])
NEGATIVE_SKILLS = set(_tax["negative_skills"])

PROF_W = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}


def skills_trust_score(candidate: dict) -> float:
    assessments = candidate.get("redrob_signals", {}).get("skills_assessment_scores")
    if assessments is None:
        assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    descriptions = " ".join(
        r.get("description", "").lower() for r in candidate.get("career_history", [])
    )
    t1_total = t2_total = neg_hits = 0.0

    for skill in candidate.get("skills", []):
        name = skill.get("name", "").lower()
        prof = PROF_W.get(skill.get("proficiency", "beginner"), 0.3)
        dur = min(skill.get("duration_months", 0) / 12, 2.0)

        trust = prof * 0.60 + min(dur / 2.0, 1.0) * 0.40

        first_word = name.split(" ")[0] if name else ""
        if name in descriptions or (first_word and first_word in descriptions):
            trust *= 1.30
        elif skill.get("duration_months", 0) == 0:
            trust *= 0.15

        if name in assessments:
            assessed = assessments[name] / 100
            trust = 0.35 * trust + 0.65 * assessed

        if skill.get("endorsements", 0) > 20:
            trust = min(trust * 1.10, 1.0)

        if name in TIER1_SKILLS:
            t1_total += trust
        elif name in TIER2_SKILLS:
            t2_total += trust * 0.5
        elif name in NEGATIVE_SKILLS:
            neg_hits += 0.12

    raw = min(t1_total / 4.0, 1.0) * 0.8 + min(t2_total / 3.0, 1.0) * 0.2

    return max(raw - min(neg_hits, 0.3), 0.0)
