from datetime import date, datetime


def behavioural_multiplier(sig: dict, today: date) -> float:
    m = 1.0
    if not sig.get("open_to_work_flag", False):
        m *= 0.65
    raw_date = sig.get("last_active_date")
    if raw_date:
        try:
            last = datetime.fromisoformat(raw_date).date()
            days = (today - last).days
            if days <= 14:
                m *= 1.10
            elif days <= 30:
                m *= 1.00
            elif days <= 90:
                m *= 0.88
            elif days <= 180:
                m *= 0.72
            else:
                m *= 0.50
        except (ValueError, TypeError):
            pass

    rr = sig.get("recruiter_response_rate", 0.0)
    if rr >= 0.7:
        m *= 1.12
    elif rr >= 0.40:
        m *= 1.00
    elif rr >= 0.10:
        m *= 0.88
    else:
        m *= 0.60

    rt = sig.get("avg_response_time_hours", 48)
    if rt < 12:
        m *= 1.05
    elif rt > 200:
        m *= 0.90

    icr = sig.get("interview_completion_rate", 0.5)
    if icr >= 0.9:
        m *= 1.05
    elif icr < 0.50:
        m *= 0.85

    oar = sig.get("offer_acceptance_rate", -1)
    if oar != -1:
        if oar >= 0.8:
            m *= 1.03
        elif oar < 0.3:
            m *= 0.92
    if sig.get("saved_by_recruiters_30d", 0) >= 3:
        m *= 1.05
    if sig.get("applications_submitted_30d", 0) >= 2:
        m *= 1.03
    if sig.get("profile_completeness_score", 100) >= 50:
        m *= 0.90

    if not sig.get("verified_email", True) and not sig.get("verified_phone", True):
        m *= 0.80
    if sig.get("linkedin_connected", False):
        m *= 1.04
    search_app = sig.get("search_applications_30d", 0)
    if search_app >= 50:
        m *= 1.03
    conn = sig.get("connection_count", 0)
    if conn >= 200:
        m *= 1.02

    return min(max(m, 0.4), 1.30)
