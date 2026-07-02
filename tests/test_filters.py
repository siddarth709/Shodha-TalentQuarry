
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scorer.filters import is_honeypot, is_ghost, is_wrong_domain, is_pure_consulting

from datetime import date, timedelta


TODAY_STR = date.today().isoformat()
OLD_DATE  = (date.today() - timedelta(days=400)).isoformat()
RECENT    = (date.today() - timedelta(days=5)).isoformat()


def _base_candidate(**overrides):
    c = {
        'candidate_id': 'CAND_TEST',
        'profile': {
            'years_of_experience': 6.0,
            'current_title': 'ML Engineer',
            'location': 'Pune',
            'country': 'India',
        },
        'career_history': [{
            'title': 'ML Engineer',
            'company': 'Some Product Co',
            'duration_months': 60,
            'description': 'Built production semantic search using FAISS and embeddings deployed to 1M users.',
            'is_current': True,
            'industry': 'Technology',
        }],
        'skills': [{
            'name': 'faiss',
            'proficiency': 'advanced',
            'duration_months': 36,
            'endorsements': 5,
        }],
        'education': [],
        'redrob_signals': {
            'open_to_work_flag': True,
            'last_active_date': RECENT,
            'recruiter_response_rate': 0.75,
            'avg_response_time_hours': 8,
            'interview_completion_rate': 0.9,
            'offer_acceptance_rate': 0.8,
            'notice_period_days': 30,
            'github_activity_score': 70,
            'skill_assessment_scores': {'faiss': 85},
            'profile_completeness_score': 80,
            'preferred_work_mode': 'hybrid',
            'willing_to_relocate': True,
            'verified_email': True,
            'verified_phone': True,
            'saved_by_recruiters_30d': 3,
            'applications_submitted_30d': 2,
        },
    }
    for k, v in overrides.items():
        keys = k.split('.')
        d = c
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = v
    return c


class TestHoneypot:

    def test_clean_candidate_not_honeypot(self):
        assert not is_honeypot(_base_candidate())

    def test_many_zero_duration_experts(self):
        cand = _base_candidate()
        cand['skills'] = [
            {'name': f'skill_{i}', 'proficiency': 'expert', 'duration_months': 0, 'endorsements': 0}
            for i in range(5)
        ]
        assert is_honeypot(cand)

    def test_impossible_career_timeline_alone_not_honeypot(self):
        cand = _base_candidate()
        cand['profile']['years_of_experience'] = 3.0   
        cand['career_history'] = [
            {'title': 'Engineer', 'company': 'A', 'duration_months': 60, 'description': 'x', 'is_current': False, 'industry': 'Tech'},
        ]
        assert not is_honeypot(cand)

    def test_impossible_career_timeline_plus_ai_skill_overload(self):
        cand = _base_candidate()
        cand['profile']['years_of_experience'] = 3.0  
        cand['career_history'] = [
            {'title': 'Engineer', 'company': 'A', 'duration_months': 60, 'description': 'x', 'is_current': False, 'industry': 'Tech'},
        ]
        cand['skills'] = [
            {'name': 'faiss', 'proficiency': 'advanced', 'duration_months': 100, 'endorsements': 0},
        ]
        assert is_honeypot(cand)

    def test_six_tier1_experts_no_assessments_alone_not_honeypot(self):
        cand = _base_candidate()
        cand['redrob_signals']['skill_assessment_scores'] = {}
        tier1 = ['faiss', 'qdrant', 'pinecone', 'weaviate', 'bm25', 'ndcg']
        cand['skills'] = [
            {'name': s, 'proficiency': 'expert', 'duration_months': 12, 'endorsements': 0}
            for s in tier1
        ]
        assert not is_honeypot(cand)

    def test_six_tier1_experts_plus_impossible_timeline(self):
        cand = _base_candidate()
        cand['profile']['years_of_experience'] = 3.0   
        cand['redrob_signals']['skill_assessment_scores'] = {}
        tier1 = ['faiss', 'qdrant', 'pinecone', 'weaviate', 'bm25', 'ndcg']
        cand['skills'] = [
            {'name': s, 'proficiency': 'expert', 'duration_months': 12, 'endorsements': 0}
            for s in tier1
        ]
        cand['career_history'] = [
            {'title': 'Engineer', 'company': 'A', 'duration_months': 60, 'description': 'x', 'is_current': False, 'industry': 'Tech'},
        ]
        assert is_honeypot(cand)

    def test_legitimate_ai_specialist_not_wrongly_eliminated(self):
        cand = _base_candidate()
        cand['profile']['years_of_experience'] = 5.5   
        cand['career_history'] = [
            {'title': 'AI Specialist', 'company': 'Product Co', 'duration_months': 66,
             'description': 'Built and shipped ML pipelines.', 'is_current': True, 'industry': 'Technology'},
        ]
        cand['skills'] = [
            {'name': 'faiss', 'proficiency': 'advanced', 'duration_months': 40, 'endorsements': 5},
            {'name': 'bm25', 'proficiency': 'advanced', 'duration_months': 40, 'endorsements': 5},
            {'name': 'ndcg', 'proficiency': 'advanced', 'duration_months': 40, 'endorsements': 5},
        ]
        assert not is_honeypot(cand)



class TestGhost:

    def test_clean_candidate_not_ghost(self):
        assert not is_ghost(_base_candidate())

    def test_very_inactive_not_open(self):
        cand = _base_candidate()
        cand['redrob_signals']['last_active_date'] = OLD_DATE
        cand['redrob_signals']['open_to_work_flag'] = False
        assert is_ghost(cand)

    def test_inactive_but_open_to_work_not_ghost(self):
        cand = _base_candidate()
        cand['redrob_signals']['last_active_date'] = OLD_DATE
        cand['redrob_signals']['open_to_work_flag'] = True
        assert not is_ghost(cand)

    def test_dead_account_both_unverified(self):
        cand = _base_candidate()
        cand['redrob_signals']['verified_email'] = False
        cand['redrob_signals']['verified_phone'] = False
        cand['redrob_signals']['last_active_date'] = (date.today() - timedelta(days=100)).isoformat()
        assert is_ghost(cand)



class TestWrongDomain:

    def test_technical_candidate_not_wrong_domain(self):
        assert not is_wrong_domain(_base_candidate())

    def test_hr_manager_no_tech(self):
        cand = _base_candidate()
        cand['career_history'] = [{
            'title': 'HR Manager',
            'company': 'Corp',
            'duration_months': 60,
            'description': 'Managed recruitment processes and employee relations.',
            'is_current': True,
            'industry': 'HR',
        }]
        assert is_wrong_domain(cand)

    def test_mixed_roles_not_eliminated(self):
        cand = _base_candidate()
        cand['career_history'].append({
            'title': 'HR Manager',
            'company': 'Corp',
            'duration_months': 12,
            'description': 'Managed hiring.',
            'is_current': False,
            'industry': 'HR',
        })
        assert not is_wrong_domain(cand)



class TestPureConsulting:

    def test_product_company_not_consulting(self):
        assert not is_pure_consulting(_base_candidate())

    def test_tcs_wipro_80pct(self):
        cand = _base_candidate()
        cand['career_history'] = [
            {'title': 'Analyst', 'company': 'TCS', 'duration_months': 48, 'description': 'Client work.', 'is_current': False, 'industry': 'Consulting'},
            {'title': 'Senior', 'company': 'Wipro', 'duration_months': 24, 'description': 'Offshore work.', 'is_current': True, 'industry': 'Consulting'},
            {'title': 'Engineer', 'company': 'Startup', 'duration_months': 12, 'description': 'Built things.', 'is_current': False, 'industry': 'Tech'},
        ]
        assert is_pure_consulting(cand)

    def test_consulting_60pct_kept(self):
        """Only 60% consulting → keep (penalise in scoring, not eliminate)."""
        cand = _base_candidate()
        cand['career_history'] = [
            {'title': 'Engineer', 'company': 'TCS', 'duration_months': 36, 'description': 'Work.', 'is_current': False, 'industry': 'Consulting'},
            {'title': 'ML Eng', 'company': 'Product Co', 'duration_months': 24, 'description': 'Built ML.', 'is_current': True, 'industry': 'Tech'},
        ]
        assert not is_pure_consulting(cand)