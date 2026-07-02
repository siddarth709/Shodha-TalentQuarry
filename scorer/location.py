import json, os
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

with open(os.path.join(_DATA_DIR, 'location_tiers.json')) as f:
    _locs = json.load(f)

TIER1_LOCS = set(_locs['tier1_locations'])
TIER2_LOCS = set(_locs['tier2_locations'])

def location_score(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    loc = profile.get("location", "").lower()
    country = profile.get("country", "India")
    relocate = sig.get("willing_to_relocate", False)
    notice = sig.get("notice_period_days", 60)
    mode = sig.get("preferred_work_mode", "hybrid")

    if any(x in loc for x in TIER1_LOCS):
        base = 1.0
    elif any(x in loc for x in TIER2_LOCS):
        base = 0.85 if relocate else 0.65
    elif country == "India":
        base = 0.6 if relocate else 0.35
    else:
        base = 0.2

    if notice<=15:
        base = min(base * 1.10, 1.0)
    elif notice <= 30:
        base = min(base * 1.05, 1.0)
    elif notice > 90:
        base *= 0.85
    

    if mode == "remote" and not relocate and base < 0.7:
        base  *= 0.85
    
    return base