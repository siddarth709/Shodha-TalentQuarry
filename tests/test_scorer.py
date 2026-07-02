import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import date, timedelta

from scorer.career      import career_evidence_score
from scorer.skills      import skills_trust_score
from scorer.experience  import experience_fit_score
from scorer.location    import location_score
from scorer.education   import education_score
from scorer.behavioural import behavioural_multiplier
from scorer.composite   import score_candidate, reasoning_string

TODAY = date.today()
RECENT = (TODAY - timedelta(days=5)).isoformat()


def _base_signals(**overrides):
    sig = {
        'open_to_work_flag': True,
        'last_active_date': RECENT,
        'recruiter_response_rate': 0.75,
        'avg_response_time_hours': 8,
        'interview_completion_rate': 0.90,
        'offer_acceptance_rate': 0.85,
        'notice_period_days': 30,
        'github_activity_score': 70,
        'skill_assessment_scores': {},
        'profile_completeness_score': 90,
        'preferred_work_mode': 'hybrid',
        'willing_to_relocate': True,
        'verified_email': True,
        'verified_phone': True,
        'saved_by_recruiters_30d': 4,
        'applications_submitted_30d': 3,
    }
    sig.update(overrides)
    return sig


def _full_candidate(
    yoe=6.5, location='Pune', country='India',
    company='Product AI Startup', consulting=False, **sig_overrides
):
    desc = (
        'Built BM25+dense hybrid search pipeline deployed to 800K daily users. '
        'Shipped production RAG system with semantic embeddings at scale. '
        'Designed NDCG evaluation framework for relevance ranking.'
    )
    if consulting:
        desc = 'Managed client deliverables for offshore engagement. Statement of work.'

    return {
        'candidate_id': 'CAND_TEST001',
        'profile': {
            'years_of_experience': yoe,
            'current_title': 'Senior ML Engineer',
            'location': location,
            'country': country,
            'headline': 'ML Engineer specialising in retrieval',
            'summary': 'Built search systems at scale.',
            'current_company_size': '51-200',
            'current_industry': 'Technology',
        },
        'career_history': [{
            'title': 'Senior ML Engineer',
            'company': company,
            'duration_months': int(yoe * 12),
            'description': desc,
            'is_current': True,
            'industry': 'Technology',
        }],
        'skills': [
            {'name': 'faiss', 'proficiency': 'expert', 'duration_months': 36, 'endorsements': 25},
            {'name': 'bm25', 'proficiency': 'advanced', 'duration_months': 24, 'endorsements': 10},
            {'name': 'sentence-transformers', 'proficiency': 'expert', 'duration_months': 30, 'endorsements': 30},
            {'name': 'nlp', 'proficiency': 'advanced', 'duration_months': 48, 'endorsements': 15},
        ],
        'education': [{
            'degree': 'B.Tech',
            'field_of_study': 'Computer Science',
            'institution': 'IIT Bombay',
            'tier': 'tier_1',
            'graduation_year': 2018,
        }],
        'redrob_signals': _base_signals(**sig_overrides),
    }


class TestCareerEvidenceScore:

    def test_strong_production_career_high_score(self):
        c = _full_candidate()
        score = career_evidence_score(c)
        assert score > 0.60, f"Expected > 0.60, got {score:.3f}"

    def test_consulting_career_penalised(self):
        c = _full_candidate(consulting=True, company='TCS')
        score = career_evidence_score(c)
        assert score < 0.15, f"Expected < 0.15 (consulting), got {score:.3f}"

    def test_empty_history_zero(self):
        c = _full_candidate()
        c['career_history'] = []
        assert career_evidence_score(c) == 0.0

    def test_score_bounded_0_to_1(self):
        for _ in range(5):
            c = _full_candidate()
            s = career_evidence_score(c)
            assert 0.0 <= s <= 1.0



class TestSkillsTrustScore:

    def test_backed_tier1_skills_high_score(self):
        c = _full_candidate()
        score = skills_trust_score(c)
        assert score > 0.50, f"Expected > 0.50, got {score:.3f}"

    def test_zero_duration_stuffer_low_score(self):
        c = _full_candidate()
        c['skills'] = [
            {'name': 'faiss', 'proficiency': 'expert', 'duration_months': 0, 'endorsements': 0},
            {'name': 'pinecone', 'proficiency': 'expert', 'duration_months': 0, 'endorsements': 0},
            {'name': 'ndcg', 'proficiency': 'expert', 'duration_months': 0, 'endorsements': 0},
        ]
        c['career_history'] = [{
            'title': 'Marketing Manager',
            'company': 'Some Corp',
            'duration_months': 60,
            'description': 'Managed social media campaigns and brand content.',
            'is_current': True,
            'industry': 'Marketing',
        }]
        score = skills_trust_score(c)
        assert score < 0.15, f"Expected < 0.15 (stuffer), got {score:.3f}"

    def test_platform_assessment_boosts_weak_skills(self):
        base_skills = [
            {'name': 'faiss', 'proficiency': 'intermediate', 'duration_months': 6, 'endorsements': 0},
        ]
        base_history = [{'title': 'Data Scientist', 'company': 'Corp', 'duration_months': 24,
                          'description': 'Analysed data and built faiss index prototypes.',
                          'is_current': True, 'industry': 'Tech'}]

        c_no_assessment = {
            'skills': base_skills,
            'career_history': base_history,
            'redrob_signals': _base_signals(skill_assessment_scores={}),
        }
        c_with_assessment = {
            'skills': base_skills,
            'career_history': base_history,
            'redrob_signals': _base_signals(skill_assessment_scores={'faiss': 95}),
        }
        s_without = skills_trust_score(c_no_assessment)
        s_with    = skills_trust_score(c_with_assessment)
        assert s_with > s_without, (
            f"Assessment (95/100) should boost score: with={s_with:.3f} without={s_without:.3f}"
        )



class TestExperienceFitScore:

    def test_sweet_spot_5_to_9_years(self):
        for yoe in [5.0, 6.5, 7.0, 9.0]:
            c = _full_candidate(yoe=yoe)
            s = experience_fit_score(c, 0.70)
            assert s == 1.00, f"YoE={yoe} should be 1.0, got {s}"

    def test_over_qualified_penalty(self):
        c = _full_candidate(yoe=11.0)
        s = experience_fit_score(c, 0.70)
        assert s < 1.00

    def test_domain_mismatch_halves_score(self):
        c = _full_candidate(yoe=7.0)
        s_good  = experience_fit_score(c, 0.70)    
        s_bad   = experience_fit_score(c, 0.10)   
        assert s_bad < s_good * 0.6



class TestLocationScore:

    def test_pune_top_score(self):
        c = _full_candidate(location='Pune')
        assert location_score(c) >= 1.0

    def test_bangalore_with_relocation(self):
        c = _full_candidate(location='Bangalore')
        c['redrob_signals']['willing_to_relocate'] = True
        s = location_score(c)
        assert 0.80 <= s <= 1.0

    def test_outside_india_heavy_penalty(self):
        c = _full_candidate(location='London', country='UK')
        s = location_score(c)
        assert s <= 0.25

    def test_short_notice_bonus(self):
        c1 = _full_candidate(location='Pune')
        c1['redrob_signals']['notice_period_days'] = 10
        c2 = _full_candidate(location='Pune')
        c2['redrob_signals']['notice_period_days'] = 120
        assert location_score(c1) > location_score(c2)


class TestEducationScore:

    def test_tier1_cs_high_score(self):
        c = _full_candidate()
        s = education_score(c)
        assert s > 0.75

    def test_no_education_neutral(self):
        c = _full_candidate()
        c['education'] = []
        s = education_score(c)
        assert 0.45 <= s <= 0.65, f"No-education should be near-neutral, got {s:.3f}"

    def test_github_bonus(self):
        c_high = _full_candidate()
        c_high['redrob_signals']['github_activity_score'] = 80
        s_high = education_score(c_high)

        c_low = _full_candidate()
        c_low['redrob_signals']['github_activity_score'] = 5
        s_low = education_score(c_low)

        assert s_high > s_low, f"High github {s_high:.3f} should beat low github {s_low:.3f}"


class TestBehaviouralMultiplier:

    def test_ideal_candidate_above_1(self):
        sig = _base_signals()
        m = behavioural_multiplier(sig, TODAY)
        assert m > 1.0

    def test_not_open_to_work_penalised(self):
        sig_closed = _base_signals(open_to_work_flag=False)
        sig_open   = _base_signals(open_to_work_flag=True)
        m_closed = behavioural_multiplier(sig_closed, TODAY)
        m_open   = behavioural_multiplier(sig_open, TODAY)
        assert m_closed < m_open, f"closed={m_closed:.3f} should be < open={m_open:.3f}"

    def test_ghost_low_multiplier(self):
        old = (TODAY - timedelta(days=350)).isoformat()
        sig = _base_signals(
            open_to_work_flag=False,
            last_active_date=old,
            recruiter_response_rate=0.05,
        )
        m = behavioural_multiplier(sig, TODAY)
        assert m <= 0.45

    def test_clamp_floor(self):
        old = (TODAY - timedelta(days=500)).isoformat()
        sig = _base_signals(
            open_to_work_flag=False,
            last_active_date=old,
            recruiter_response_rate=0.0,
            interview_completion_rate=0.0,
            verified_email=False,
            verified_phone=False,
        )
        m = behavioural_multiplier(sig, TODAY)
        assert m >= 0.40

    def test_clamp_ceiling(self):
        sig = _base_signals()
        m = behavioural_multiplier(sig, TODAY)
        assert m <= 1.30


class TestCompositeScore:

    def test_strong_candidate_high_final_score(self):
        c = _full_candidate()
        r = score_candidate(c)
        assert r is not None
        assert r['score'] > 0.60

    def test_result_structure(self):
        c = _full_candidate()
        r = score_candidate(c)
        assert r is not None
        assert 'candidate_id' in r
        assert 'score' in r
        assert 'dimensions' in r
        assert set(r['dimensions'].keys()) == {'A', 'B', 'C', 'D', 'E'}
        assert 'multiplier' in r

    def test_score_bounded(self):
        c = _full_candidate()
        r = score_candidate(c)
        assert 0.0 <= r['score'] <= 1.0

    def test_reasoning_string_not_empty(self):
        c = _full_candidate()
        r = score_candidate(c)
        rs = reasoning_string(r)
        assert len(rs) > 20
        assert '.' in rs

    def test_reasoning_contains_title_and_yoe(self):
        c = _full_candidate(yoe=6.5)
        r = score_candidate(c)
        rs = reasoning_string(r)
        assert '6.5' in rs
        assert 'ML' in rs or 'Engineer' in rs