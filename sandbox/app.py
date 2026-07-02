import streamlit as st
import json
import os
import sys
import io
import csv
import gzip
from pathlib import Path
from datetime import date, datetime

# Setup project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scorer.composite import (
    score_candidate,
    reasoning_string
)
from scorer.filters import should_eliminate
from scorer.skills import TIER1_SKILLS, TIER2_SKILLS, PROF_W
from scorer.location import location_score

# Page Configuration
st.set_page_config(
    page_title="Shodha - TalentQuarry",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Premium Font and Color Styling */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* App Title Gradient styling */
    .title-gradient {
        background: linear-gradient(135deg, #FF4B4B 0%, #7E22CE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    /* Premium Card Design */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #7E22CE;
    }
    .metric-val {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #7E22CE;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
        margin-top: 0.3rem;
    }

    /* Candidate Detail Glassmorphism Card */
    .candidate-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s;
    }
    .candidate-card:hover {
        border-color: rgba(126, 34, 206, 0.3);
    }
    
    /* Badges */
    .tag-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 50px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-primary { background-color: rgba(126, 34, 206, 0.15); color: #A855F7; border: 1px solid rgba(126, 34, 206, 0.3); }
    .badge-secondary { background-color: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-success { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-danger { background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* Custom division lines */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper Function: Locate Candidates Database Path (local or cloud)
def get_candidate_file_info():
    path = Path("candidates.jsonl")
    if not path.exists():
        path = Path("candidates.jsonl.gz")
    if not path.exists():
        path = Path("candidates_sample.jsonl")
    return path if path.exists() else None

# Cached Candidate Ranker function to prevent OOM and re-execution on every render
@st.cache_data(ttl=600, show_spinner="Ranking candidates...")
def rank_candidates_cached(file_path_str: str, jd_text: str):
    import json
    # Load precomputed database if available to prevent long startup delays
    precomputed_path = Path("precomputed_top_candidates.json")
    if precomputed_path.exists():
        with open(precomputed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["top_results"], data["total_processed"], data["eliminated_count"]

    import gzip
    import heapq
    
    file_path = Path(file_path_str)
    top_heap = []
    total_processed = 0
    eliminated_count = 0
    counter = 0
    
    open_fn = gzip.open if file_path.suffix == ".gz" else open
    with open_fn(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_processed += 1
            c = json.loads(line)
            scored = score_candidate(c)
            if scored is None:
                eliminated_count += 1
            else:
                raw_score = scored["raw_score"]
                heap_item = (raw_score, counter, scored)
                counter += 1
                
                if len(top_heap) < 100:
                    heapq.heappush(top_heap, heap_item)
                else:
                    if raw_score > top_heap[0][0]:
                        heapq.heappushpop(top_heap, heap_item)
                        
    # Extract results from heap
    top_results = [item[2] for item in top_heap]
    top_results.sort(key=lambda x: (-x["raw_score"], x["candidate_id"]))
    if top_results:
        max_raw = max((r["raw_score"] for r in top_results), default=1.0)
        scale = max(max_raw, 1.0)
        for r in top_results:
            r["display_score"] = r["raw_score"] / scale
        top_results.sort(key=lambda x: (-round(x["display_score"], 4), x["candidate_id"]))
        
    return top_results, total_processed, eliminated_count

# Default Job Description Loader
@st.cache_data
def load_default_jd():
    path = Path("jd.txt")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "Job Description text not found. Paste your JD here."

# CSV Converter for Download
def convert_to_csv(top_results):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, res in enumerate(top_results, 1):
        display_score = res.get("display_score", res["score"])
        reason = reasoning_string(res)
        writer.writerow([
            res["candidate_id"],
            rank,
            f"{display_score:.4f}",
            reason
        ])
    return output.getvalue()

# App Header
st.markdown('<div class="title-gradient">Shodha - TalentQuarry</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Advanced Intelligence meets Elite Recruitment</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://static.streamlit.io/badges/streamlit_badge_black_white.svg", width=120)
st.sidebar.header("Sandbox Configurations")

# Locate Candidates database file
file_path = get_candidate_file_info()
precomputed_active = Path("precomputed_top_candidates.json").exists()
if precomputed_active:
    st.sidebar.success("Loaded database successfully from precomputed_top_candidates.json.")
elif file_path:
    if file_path.name == "candidates.jsonl":
        st.sidebar.success("Loaded database successfully from candidates.jsonl.")
    elif file_path.name == "candidates.jsonl.gz":
        st.sidebar.success("Loaded database successfully from candidates.jsonl.gz.")
    else:
        st.sidebar.warning("Using candidates_sample.jsonl (Streamlit Cloud sandbox mode).")
else:
    st.sidebar.error("Could not locate candidates database. Please verify the files are present.")

# Application Tabs
tab_ranker, tab_simulator, tab_jd = st.tabs([
    "Candidate Ranker",
    "What-If Simulator",
    "Edit Job Description"
])

# Maintain JD in session state
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = load_default_jd()

# Edit Job Description Tab
with tab_jd:
    st.subheader("Edit Job Description Text")
    jd_input = st.text_area(
        "Job Description (JD)",
        value=st.session_state["jd_text"],
        height=400,
        help="Paste a job description here."
    )
    if st.button("Apply Changes"):
        st.session_state["jd_text"] = jd_input
        st.success("Job description updated successfully!")

# Candidate Ranker Tab
with tab_ranker:
    if not file_path:
        st.warning("Please verify that candidates database is present in the workspace root.")
    else:
        st.subheader("Discover and Rank Candidates")
        
        # Scoring Execution
        top_results, total_processed, eliminated_count = rank_candidates_cached(
            str(file_path),
            st.session_state["jd_text"]
        )
        scored_count = total_processed - eliminated_count
        
        # Render Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{total_processed:,}</div>
                <div class="metric-lbl">Total Candidates</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            pct_elim = (eliminated_count / total_processed) if total_processed > 0 else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{eliminated_count:,} ({pct_elim:.1%})</div>
                <div class="metric-lbl">Eliminated (Pass 1 Filters)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{scored_count:,}</div>
                <div class="metric-lbl">Scored & Qualified</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # CSV Download Button
        csv_data = convert_to_csv(top_results)
        st.download_button(
            label="Download Top 100 as CSV",
            data=csv_data,
            file_name="team-Siddarth.NS.csv",
            mime="text/csv",
            help="Download the top 100 candidate ranking in the standard CSV submission format."
        )
        
        st.write("")
        
        # Display Rankings Table
        st.markdown("### Top Ranked Matches")
        
        table_rows = []
        for idx, res in enumerate(top_results, 1):
            profile = res["candidate"].get("profile", {})
            title = profile.get("current_title", "N/A")
            company = profile.get("current_company", "N/A")
            yoe = profile.get("years_of_experience", 0.0)
            display_score = res.get("display_score", res["score"])
            reason = reasoning_string(res)
                
            table_rows.append({
                "Rank": idx,
                "Candidate ID": res["candidate_id"],
                "Score": f"{display_score:.4f}",
                "Current Title": title,
                "Company": company,
                "YoE": f"{yoe:.1f} yrs",
                "Reasoning Summary": reason
            })
            
        st.table(table_rows)
        
        # Expandable Detail Viewer
        st.markdown("### Detailed Profile Inspector")
        for idx, res in enumerate(top_results, 1):
            display_score = res.get("display_score", res["score"])
            profile = res["candidate"].get("profile", {})
            title = profile.get("current_title", "N/A")
            yoe = profile.get("years_of_experience", 0.0)
            
            with st.expander(f"Rank #{idx} — {res['candidate_id']} | {title} (Score: {display_score:.4f})"):
                col_det1, col_det2 = st.columns([1, 1])
                
                with col_det1:
                    st.markdown("#### Profile Information")
                    st.write(f"**Current Title:** {title}")
                    st.write(f"**Experience:** {yoe:.1f} years")
                    st.write(f"**Current Company:** {profile.get('current_company', 'N/A')} (Size: {profile.get('current_company_size', 'N/A')})")
                    st.write(f"**Location:** {profile.get('location', 'N/A')}, {profile.get('country', 'N/A')}")
                    st.write(f"**Headline:** {profile.get('headline', 'N/A')}")
                    st.write(f"**Summary:** {profile.get('summary', 'N/A')}")
                    
                    st.markdown("#### Education")
                    for edu in res["candidate"].get("education", []):
                        st.markdown(f"- **{edu.get('degree', 'Degree')} in {edu.get('field_of_study', 'Field')}**")
                        st.markdown(f"  {edu.get('institution', 'N/A')} ({edu.get('graduation_year', 'N/A')}) | Tier: `{edu.get('tier', 'N/A')}`")
                        
                with col_det2:
                    st.markdown("#### Scoring Dimensions Breakdown")
                    dims = res["dimensions"]
                    
                    st.write(f"**Pass 2 Rule-Based Score Breakdown:**")
                    st.write(f"- Career Evidence (40%): **{dims['A']:.2f}**")
                    st.progress(float(dims['A']))
                    st.write(f"- Skills Trust (25%): **{dims['B']:.2f}**")
                    st.progress(float(dims['B']))
                    st.write(f"- Experience Fit (15%): **{dims['C']:.2f}**")
                    st.progress(float(dims['C']))
                    st.write(f"- Location Tier & Relo (12%): **{dims['D']:.2f}**")
                    st.progress(float(dims['D']))
                    st.write(f"- Education Background (8%): **{dims['E']:.2f}**")
                    st.progress(float(dims['E']))
                    
                    sig = res["candidate"].get("redrob_signals", {})
                    st.markdown("#### Behavioural Signals")
                    st.write(f"- **Notice Period:** {sig.get('notice_period_days', 60)} days")
                    st.write(f"- **Last Active Date:** {sig.get('last_active_date', 'N/A')}")
                    st.write(f"- **Recruiter Response Rate:** {sig.get('recruiter_response_rate', 0.0):.1%}")
                    st.write(f"- **Behavioural Multiplier (Pass 3):** `{res['multiplier']:.2f}x`")

                # Work History Detail
                st.markdown("#### Career History")
                for exp in res["candidate"].get("career_history", []):
                    st.markdown(f"**{exp.get('title', 'N/A')}** at *{exp.get('company', 'N/A')}* ({exp.get('duration_months', 0)} months)")
                    st.caption(exp.get('description', 'No description provided.'))
                    st.markdown("---")

                # Skills list
                st.markdown("#### Skills Inventory")
                skills_md = ""
                for s in res["candidate"].get("skills", []):
                    endorse = f" ({s.get('endorsements', 0)} endorsements)" if s.get('endorsements', 0) > 0 else ""
                    skills_md += f"<span class='tag-badge badge-primary'>{s.get('name')}: {s.get('proficiency')}{endorse}</span>"
                st.markdown(skills_md, unsafe_allow_html=True)


# What-If Simulator Tab
with tab_simulator:
    st.subheader("What-If Scoring Simulator")
    st.write("Construct a mock candidate profile and check how the filters and algorithm score it in real-time.")
    
    with st.form("what_if_profile"):
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### Candidate Basics")
            sim_title = st.text_input("Current Job Title", value="Senior ML Engineer")
            sim_yoe = st.slider("Stated Years of Experience", min_value=0.0, max_value=25.0, value=6.5, step=0.5)
            sim_company = st.text_input("Current Company", value="Tech Startup")
            sim_industry = st.text_input("Current Industry", value="Technology")
            
            st.markdown("#### Career History Evidence")
            sim_description = st.text_area(
                "Experience Description (Used for Career Evidence keyword verification)",
                value="Built semantic retrieval systems using dense embeddings. Deployed Milvus and Elasticsearch hybrid search. Implemented NDCG evaluation."
            )
            
            st.markdown("#### Location & Relo")
            sim_location = st.selectbox("Location (City)", options=["Pune", "Noida", "Bangalore", "Hyderabad", "Delhi", "Mumbai", "London", "New York"])
            sim_country = st.selectbox("Country", options=["India", "United Kingdom", "United States"])
            
            st.markdown("#### Education")
            sim_degree = st.selectbox("Degree", options=["B.Tech", "M.Tech", "MS", "B.Sc", "Ph.D"])
            sim_field = st.text_input("Field of Study", value="Computer Science")
            sim_inst_tier = st.selectbox("Institution Tier", options=["tier_1", "tier_2", "tier_3", "other"])

        with col_s2:
            st.markdown("#### Skills Inventory")
            st.write("Add up to 3 candidate skills to score:")
            
            col_sk1, col_sk2, col_sk3 = st.columns([2, 1, 1])
            with col_sk1:
                sk1_name = st.text_input("Skill 1", value="sentence-transformers")
                sk2_name = st.text_input("Skill 2", value="faiss")
                sk3_name = st.text_input("Skill 3", value="python")
            with col_sk2:
                sk1_prof = st.selectbox("Prof 1", options=["expert", "advanced", "intermediate", "beginner"], index=0)
                sk2_prof = st.selectbox("Prof 2", options=["expert", "advanced", "intermediate", "beginner"], index=1)
                sk3_prof = st.selectbox("Prof 3", options=["expert", "advanced", "intermediate", "beginner"], index=1)
            with col_sk3:
                sk1_dur = st.number_input("Months 1", min_value=0, value=24)
                sk2_dur = st.number_input("Months 2", min_value=0, value=12)
                sk3_dur = st.number_input("Months 3", min_value=0, value=36)
                
            st.markdown("#### Redrob Activity & Engagement Signals")
            sim_open = st.checkbox("Open To Work flag", value=True)
            sim_active = st.slider("Days since last active", min_value=0, max_value=400, value=3)
            sim_notice = st.slider("Notice Period (Days)", min_value=0, max_value=120, value=30)
            sim_rr = st.slider("Recruiter Response Rate", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
            sim_completion = st.slider("Interview Completion Rate", min_value=0.0, max_value=1.0, value=0.90, step=0.05)
            
            st.markdown("#### Platform Assessments (Verified Skill Scores)")
            sim_assess1 = st.slider("Skill 1 score", min_value=0, max_value=100, value=0)
            sim_assess2 = st.slider("Skill 2 score", min_value=0, max_value=100, value=90)
            
        submit_sim = st.form_submit_button("Compute Live Score")
        
    if submit_sim:
        # Build candidate mock dictionary
        import datetime as dt
        active_date = (date.today() - dt.timedelta(days=sim_active)).isoformat()
        
        sim_candidate = {
            "candidate_id": "CAND_SIMULATED",
            "profile": {
                "years_of_experience": sim_yoe,
                "current_title": sim_title,
                "location": sim_location,
                "country": sim_country,
                "headline": f"{sim_title} experienced in NLP",
                "summary": "Mock profile built in simulator"
            },
            "career_history": [
                {
                    "title": sim_title,
                    "company": sim_company,
                    "duration_months": int(sim_yoe * 12),
                    "description": sim_description,
                    "is_current": True,
                    "industry": sim_industry
                }
            ],
            "skills": [
                {"name": sk1_name, "proficiency": sk1_prof, "duration_months": sk1_dur, "endorsements": 10},
                {"name": sk2_name, "proficiency": sk2_prof, "duration_months": sk2_dur, "endorsements": 5},
                {"name": sk3_name, "proficiency": sk3_prof, "duration_months": sk3_dur, "endorsements": 15}
            ],
            "education": [
                {
                    "degree": sim_degree,
                    "field_of_study": sim_field,
                    "institution": "Mock University",
                    "tier": sim_inst_tier,
                    "graduation_year": 2020
                }
            ],
            "redrob_signals": {
                "open_to_work_flag": sim_open,
                "last_active_date": active_date,
                "recruiter_response_rate": sim_rr,
                "notice_period_days": sim_notice,
                "interview_completion_rate": sim_completion,
                "skills_assessment_scores": {
                    sk1_name.lower(): sim_assess1,
                    sk2_name.lower(): sim_assess2
                }
            }
        }
        
        # 1. Check elimination first
        eliminated, reason = should_eliminate(sim_candidate)
        
        if eliminated:
            st.error(f"Eliminated by Pass 1 Filters! Reason: {reason.upper()}")
        else:
            st.success("Passed Pass 1 Filters! Candidate is Qualified.")
            
            # Compute score
            res_v1 = score_candidate(sim_candidate)
            
            if res_v1:
                # Layout V1 results
                st.markdown("### Score Diagnostics")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown(f"#### Rule-Based Score: {res_v1['raw_score']:.4f}")
                    st.write("*(Note: displayed scores in list view are rescaled against the max raw score)*")
                    
                    st.markdown("##### Dimension Scores Breakdown")
                    dims = res_v1["dimensions"]
                    st.write(f"- Career Evidence (40%): **{dims['A']:.2f}**")
                    st.progress(float(dims['A']))
                    st.write(f"- Skills Trust (25%): **{dims['B']:.2f}**")
                    st.progress(float(dims['B']))
                    st.write(f"- Experience Fit (15%): **{dims['C']:.2f}**")
                    st.progress(float(dims['C']))
                    st.write(f"- Location Score (12%): **{dims['D']:.2f}**")
                    st.progress(float(dims['D']))
                    st.write(f"- Education Score (8%): **{dims['E']:.2f}**")
                    st.progress(float(dims['E']))
                
                with col_res2:
                    st.markdown(f"#### Behavioural Multiplier: {res_v1['multiplier']:.2f}x")
                    st.write(f"Raw profile score: {(res_v1['raw_score']/res_v1['multiplier']):.4f}")
                    
                    st.markdown("##### Reasoning Text")
                    st.info(reasoning_string(res_v1))
