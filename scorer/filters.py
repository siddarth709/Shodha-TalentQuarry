from datetime import date, datetime
import json, os

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

with open(os.path.join(_DATA_DIR, 'skill_taxonomy.json')) as f:
    _tax = json.load(f)
with open(os.path.join(_DATA_DIR, 'career_evidence_keywords.json')) as f:
    _kw = json.load(f)

CONSULTING_FIIRMS = set(_tax['consulting_firms'])
TIER1_SKILLS_SET = set(_tax['tier1_skills'])
NON_TECH_TITLES = set(_kw['non_tech_titles'])
TECH_KEYWORDS = set(_kw['tech_keywords'])

def is_honeypot(candidate: dict) -> bool:
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    stated_months = float(profile.get("years_of_experience", 0)) * 12

    zero_dur_experts = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced") 
        and s.get("duration_months", 0) == 0
    ]
    if len(zero_dur_experts) >= 4:
        return True
    
    total_career = sum(r.get("duration_months", 0) for r in history)
    signal_2 = stated_months > 0 and total_career > stated_months * 1.4

    ai_skill_months = sum(
        s.get("duration_months", 0)
        for s in skills
        if s.get("name", "").lower() in TIER1_SKILLS_SET
    )
    signal_3 = stated_months > 0 and ai_skill_months > stated_months * 2.5

    assessments = candidate.get("redrob_signals", {}).get("skills_assessment_scores")
    if assessments is None:
        assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    tier1_experts = [
        s for s in skills
        if s.get("name", "").lower() in TIER1_SKILLS_SET
        and s.get("proficiency") == 'expert'
    ]
    signal_4 = len(tier1_experts) >= 6 and len(assessments) == 0

    return sum([signal_2 ,signal_3 ,signal_4]) >= 2

def is_ghost(candidate: dict, today: date | None = None) -> bool:
    today = today or date.today()
    sig = candidate.get("redrob_signals", {})

    raw_date = sig.get("last_active_date")
    if not raw_date:
        return False
    try:
        last = datetime.fromisoformat(raw_date).date()
    except (ValueError, TypeError):
        return False
    days_inactive = (today - last).days

    if days_inactive > 300 and not sig.get('open_to_work_flag', False):
        return True
    
    if (
        not sig.get('verified_email', True)
        and not sig.get('verified_phone', True)
        and days_inactive > 90
    ):
        return True
    return False

def is_wrong_domain(candidate: dict) -> bool:
    history = candidate.get("career_history", [])
    if not history:
        return False
    all_titles = [r.get("title", "").lower() for r in history]
    non_tech_count = sum(1 for t in all_titles if any(nt in t for nt in NON_TECH_TITLES))
    if non_tech_count< len(all_titles):
        return False
    
    all_desc = ' '.join(r.get("description", "").lower() for r in history)
    tech_hits = sum(1 for kw in TECH_KEYWORDS if kw in all_desc)
    return tech_hits < 3

def is_pure_consulting(candidate: dict) -> bool:
    history = candidate.get('career_history', [])
    if not history:
        return False
    total_months = sum(r.get("duration_months", 0) for r in history) or 1
    consulting = sum(
        r.get("duration_months", 0)
        for r in history
        if any(cf in r.get("company", "").lower() for cf in CONSULTING_FIIRMS)
    )
    return consulting / total_months > 0.8

def should_eliminate(candidate: dict) -> tuple[bool, str]:
    if is_honeypot(candidate):
        return True, "honeypot"
    if is_ghost(candidate):
        return True, "ghost"
    if is_wrong_domain(candidate):
        return True, "wrong_domain"
    if is_pure_consulting(candidate):
        return True, "pure_consulting"
    return False, ""
