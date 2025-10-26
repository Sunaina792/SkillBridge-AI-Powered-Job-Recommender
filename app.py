import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
import fitz  # PyMuPDF
from io import BytesIO
import base64

# Download stopwords
nltk.download('stopwords')

# --- Fixed Styling ---
st.set_page_config(page_title="Skill-Based Job Recommender", layout="wide")

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme toggle
theme_col1, theme_col2 = st.columns([9, 1])
with theme_col1:
    st.markdown(f"""
    <h1 style="font-size: 2.8rem; margin-bottom: 0.5rem; color: {'#ffffff' if st.session_state.theme == 'dark' else '#000000'}">
        Discover Your Ideal Career Based on Your Skills
    </h1>
    <p style="font-size: 1.2rem; color: {'#cccccc' if st.session_state.theme == 'dark' else '#666666'}; margin-top: 0">
        Get personalized job recommendations, identify skill gaps, and unlock curated learning paths to accelerate your career growth
    </p>
    """, unsafe_allow_html=True)
with theme_col2:
    if st.button("☀️" if st.session_state.theme == 'dark' else "🌙", key="theme_toggle"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# Apply theme-specific styles
if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: #ffffff;
        }
        .main-title {
            font-size: 2.8rem;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #cccccc;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #64b5f6;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #333333;
        }
        .dashboard-card {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            border: 1px solid #333333;
        }
        .dashboard-card:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
            transform: translateY(-3px);
        }
        .card-title {
            font-size: 1.3rem;
            font-weight: bold;
            color: #64b5f6;
            margin-bottom: 0.8rem;
        }
        .card-description {
            color: #bbbbbb;
            font-size: 1rem;
            margin-bottom: 1.2rem;
            line-height: 1.5;
        }
        .stButton>button {
            background-color: #1976d2;
            color: white;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s;
            border: none;
            width: 100%;
            margin: 0.3rem 0;
        }
        .stButton>button:hover {
            background-color: #1565c0;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(25, 118, 210, 0.3);
        }
        .stButton>button:active, .stButton>button:focus {
            background-color: #0d47a1;
            outline: 2px solid #64b5f6;
        }
        .nav-button {
            background-color: #2c2c2c;
            color: #ffffff;
            border: none;
            padding: 0.7rem 1.2rem;
            border-radius: 8px;
            margin: 0.3rem;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        .nav-button:hover {
            background-color: #3d3d3d;
        }
        .nav-button.active {
            background-color: #1976d2;
            color: white;
        }
        .match-score {
            font-weight: bold;
            color: #64b5f6;
        }
        .skill-tag {
            display: inline-block;
            background-color: #1565c0;
            color: white;
            padding: 0.3rem 0.7rem;
            border-radius: 20px;
            margin: 0.3rem;
            font-size: 0.9rem;
        }
        .matched { background-color: #1565c0; }
        .missing { background-color: #c62828; }
        div.stButton {
            margin: 0.8rem 0;
        }
        table, th, td {
            border: 1px solid #444444;
            border-collapse: collapse;
            padding: 8px;
            color: #ffffff;
        }
        th {
            background-color: #2c2c2c;
        }
        .metric-card {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            border: 1px solid #333333;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #64b5f6;
        }
        .metric-label {
            font-size: 1rem;
            color: #bbbbbb;
        }
        .stDataFrame {
            background-color: #1e1e1e;
        }
        .stDataFrame th, .stDataFrame td {
            color: #ffffff;
        }
        .stSelectbox, .stMultiselect, .stRadio {
            color: #ffffff;
        }
        .stSelectbox > div, .stMultiselect > div, .stRadio > div {
            background-color: #2c2c2c;
            border: 1px solid #444444;
            color: #ffffff;
        }
        .stTextInput > div {
            background-color: #2c2c2c;
            border: 1px solid #444444;
            color: #ffffff;
        }
        .stDataFrame {
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .main-title {
            font-size: 2.8rem;
            font-weight: bold;
            color: #000000;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #666666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #1976d2;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #eeeeee;
        }
        .dashboard-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 1px solid #eeeeee;
        }
        .dashboard-card:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            transform: translateY(-3px);
        }
        .card-title {
            font-size: 1.3rem;
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 0.8rem;
        }
        .card-description {
            color: #555555;
            font-size: 1rem;
            margin-bottom: 1.2rem;
            line-height: 1.5;
        }
        .stButton>button {
            background-color: #1976d2;
            color: white;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s;
            border: none;
            width: 100%;
            margin: 0.3rem 0;
        }
        .stButton>button:hover {
            background-color: #1565c0;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(25, 118, 210, 0.3);
        }
        .stButton>button:active, .stButton>button:focus {
            background-color: #0d47a1;
            outline: 2px solid #64b5f6;
        }
        .nav-button {
            background-color: #f5f5f5;
            color: #333333;
            border: none;
            padding: 0.7rem 1.2rem;
            border-radius: 8px;
            margin: 0.3rem;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }
        .nav-button:hover {
            background-color: #e0e0e0;
        }
        .nav-button.active {
            background-color: #1976d2;
            color: white;
        }
        .match-score {
            font-weight: bold;
            color: #1976d2;
        }
        .skill-tag {
            display: inline-block;
            background-color: #1976d2;
            color: white;
            padding: 0.3rem 0.7rem;
            border-radius: 20px;
            margin: 0.3rem;
            font-size: 0.9rem;
        }
        .matched { background-color: #1976d2; }
        .missing { background-color: #e57373; }
        div.stButton {
            margin: 0.8rem 0;
        }
        table, th, td {
            border: 1px solid #dddddd;
            border-collapse: collapse;
            padding: 8px;
        }
        th {
            background-color: #f8f9fa;
        }
        .metric-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            border: 1px solid #eeeeee;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #1976d2;
        }
        .metric-label {
            font-size: 1rem;
            color: #666666;
        }
    </style>
    """, unsafe_allow_html=True)

# Load learning resources
@st.cache_data
def load_learning_resources():
    try:
        df = pd.read_csv("learning_resources.csv")
        return df
    except Exception as e:
        st.error(f"Error loading learning resources: {e}")
        return pd.DataFrame()

# Load trending skills
@st.cache_data
def load_trending_skills():
    try:
        df = pd.read_csv("trending_skills.csv")
        return df.sort_values(by='Market Demand Score', ascending=False)
    except Exception as e:
        st.error(f"Error loading trending skills: {e}")
        return pd.DataFrame()

# Load career paths
@st.cache_data
def load_career_paths():
    try:
        df = pd.read_csv("career_paths.csv")
        return df
    except Exception as e:
        st.error(f"Error loading career paths: {e}")
        return pd.DataFrame()

# Load transferable skills
@st.cache_data
def load_transferable_skills():
    try:
        df = pd.read_csv("transferable_skills.csv")
        return df
    except Exception as e:
        st.error(f"Error loading transferable skills: {e}")
        return pd.DataFrame()

# Preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Load pickled dataset
@st.cache_data
def load_data():
    try:
        with open("job_roles_dataset.pkl", "rb") as f:
            df = pickle.load(f)
        # Ensure all values are strings before concatenation
        df['Technical Skills'] = df['Technical Skills'].fillna('').astype(str)
        df['Soft Skills'] = df['Soft Skills'].fillna('').astype(str)
        df['All Skills'] = df['Technical Skills'] + ', ' + df['Soft Skills']
        df['Processed Skills'] = df['All Skills'].apply(preprocess)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Extract text from uploaded PDF resume
def extract_text_from_pdf(uploaded_file) -> str:
    text: str = ""
    try:
        with st.spinner("📄 Extracting text from PDF..."):
            # Using fitz.open with stream parameter - this is the correct API
            # Type ignore comment to suppress static analysis warning
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # type: ignore
            for page in doc:
                # Explicitly convert to string to ensure type compatibility
                page_text: str = str(page.get_text())
                text += page_text
        return text
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""

# Generate skill tags HTML
def generate_skill_tags(skills, matched_skills=None):
    if matched_skills is None:
        matched_skills = set()
    html = ""
    for skill in skills.split(','):
        skill = skill.strip()
        tag_class = "skill-tag matched" if skill in matched_skills else "skill-tag"
        html += f'<span class="{tag_class}">{skill}</span>'
    return html

# Format score for display in plain number
def format_score(score):
    return round(score, 2)

# Helper function to get learning resources for skills
def get_learning_resources_for_skills(skills, resources_df):
    """Get learning resources for a list of skills"""
    if skills and not resources_df.empty:
        # Normalize skill names for matching
        skill_matches = []
        for skill in skills:
            # Look for exact or partial matches (case insensitive)
            matching_resources = resources_df[
                resources_df['Skill'].str.contains(skill, case=False, na=False)
            ]
            if not matching_resources.empty:
                skill_matches.extend(matching_resources.to_dict('records'))
        
        # Remove duplicates based on resource name
        seen = set()
        unique_resources = []
        for resource in skill_matches:
            if resource['Resource Name'] not in seen:
                seen.add(resource['Resource Name'])
                unique_resources.append(resource)
        
        return unique_resources[:5]  # Return top 5 resources
    return []

# --- Dashboard UI ---
# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

# Load and preprocess data
df = load_data()
learning_resources = load_learning_resources()
trending_skills = load_trending_skills()
career_paths = load_career_paths()
transferable_skills = load_transferable_skills()

# Navigation
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    if st.button("🏠 Dashboard", key="nav_dashboard", 
                 help="Return to main dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()
with col2:
    if st.button("🎯 Job Match", key="nav_job_match", 
                 help="Find job recommendations"):
        st.session_state.page = 'job_match'
        st.rerun()
with col3:
    if st.button("📊 Analytics", key="nav_analytics", 
                 help="View skill analytics and trends"):
        st.session_state.page = 'analytics'
        st.rerun()
with col4:
    if st.button("🧭 Career Paths", key="nav_career", 
                 help="Explore career progression"):
        st.session_state.page = 'career'
        st.rerun()
with col5:
    if st.button("🔄 Transfer", key="nav_transfer", 
                 help="Cross-industry skill transfer"):
        st.session_state.page = 'transfer'
        st.rerun()

st.markdown("---")

# Dashboard Page
if st.session_state.page == 'dashboard' or df is None:
    st.markdown('<div class="section-header">📋 Dashboard Overview</div>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">20+</div><div class="metric-label">Job Roles</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">100+</div><div class="metric-label">Skills</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">50+</div><div class="metric-label">Learning Resources</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">15+</div><div class="metric-label">Industries</div></div>', unsafe_allow_html=True)
    
    # Feature cards with functional navigation buttons
    st.markdown('<div class="section-header">🚀 Key Features</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🎯 Job Matching</div>
            <div class="card-description">Get personalized job recommendations based on your skills using advanced ML algorithms.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Jobs", key="explore_jobs"):
            st.session_state.page = 'job_match'
            st.rerun()
        
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🧭 Career Paths</div>
            <div class="card-description">Discover progression routes and required skills for your target roles.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Paths", key="view_paths"):
            st.session_state.page = 'career'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">📊 Skill Analytics</div>
            <div class="card-description">Analyze market demand for skills and identify learning priorities.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Analytics", key="view_analytics"):
            st.session_state.page = 'analytics'
            st.rerun()
        
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">🔄 Industry Transfer</div>
            <div class="card-description">See how your skills transfer across different industries.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore Transfer", key="explore_transfer"):
            st.session_state.page = 'transfer'
            st.rerun()

# Job Matching Page
elif st.session_state.page == 'job_match' and df is not None:
    st.markdown('<div class="section-header">🎯 Job Matching</div>', unsafe_allow_html=True)
    
    if df is not None and learning_resources is not None:
        skill_list = sorted(set(skill.strip() for skills in df['All Skills'] for skill in skills.split(',')))

        input_method = st.radio("Choose your input method:", ["📄 Upload Resume", "🛠️ Manual Skill Selection"], horizontal=True, index=0)
        user_selected_skills = []

        if input_method == "📄 Upload Resume":
            st.markdown('<div class="section-header">📄 Upload Your Resume (PDF only)</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload your resume:", type=["pdf"])

            if uploaded_file:
                with st.spinner("🔍 Analyzing your resume..."):
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    user_skills = set()
                    for known_skill in skill_list:
                        if known_skill.lower() in extracted_text.lower():
                            user_skills.add(known_skill)
                    user_selected_skills = sorted(list(user_skills))
                    st.success("✅ Skills extracted from resume!")
                    st.markdown("### Your Skills:")
                    st.markdown(generate_skill_tags(", ".join(user_selected_skills)), unsafe_allow_html=True)

        else:
            st.markdown('<div class="section-header">🛠️ Select Your Skills</div>', unsafe_allow_html=True)
            manual_skills = st.multiselect("Select your skills:", options=skill_list)
            user_selected_skills = manual_skills

        user_input = ", ".join(user_selected_skills)

        if user_input:
            user_input_clean = preprocess(user_input)

            with st.spinner("🤖 Finding matching jobs..."):
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(df['Processed Skills'].tolist() + [user_input_clean])
                user_vec = tfidf_matrix[-1]
                job_vecs = tfidf_matrix[:-1]
                similarities = cosine_similarity(user_vec, job_vecs).flatten()
                df['Match Score'] = similarities

            top_matches = df.sort_values(by='Match Score', ascending=False).head(10)
            top_matches['Formatted Match Score'] = top_matches['Match Score'].apply(format_score)
            st.markdown('<div class="section-header">🔎 Top Matching Job Roles</div>', unsafe_allow_html=True)
            st.dataframe(top_matches[['Job Role', 'Formatted Match Score']].rename(columns={'Formatted Match Score': 'Match Score'}))

            selected_role = st.selectbox("🎯 Select a job role to analyze skill gap:", top_matches['Job Role'].tolist())
            if selected_role:
                role_data = df[df['Job Role'] == selected_role].iloc[0]
                # Ensure skills are strings before concatenation
                tech_skills = str(role_data['Technical Skills']) if pd.notna(role_data['Technical Skills']) else ''
                soft_skills = str(role_data['Soft Skills']) if pd.notna(role_data['Soft Skills']) else ''
                role_skills = set([s.strip() for s in (tech_skills + ", " + soft_skills).split(',')])
                user_skills = set(user_selected_skills)
                matched = sorted(list(role_skills & user_skills))
                missing = sorted(list(role_skills - user_skills))

                st.markdown(f'<div class="section-header">🧩 Skill Gap Analysis for: {selected_role}</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Matched Skills:**")
                    st.markdown(generate_skill_tags(", ".join(matched), matched), unsafe_allow_html=True)
                    st.markdown(f"**Coverage: {len(matched)/len(role_skills)*100:.1f}%**")
                with col2:
                    st.markdown("**❌ Missing Skills:**")
                    st.markdown(generate_skill_tags(", ".join(missing), matched), unsafe_allow_html=True)
                    if missing:
                        st.markdown("📚 Consider learning these skills to improve your match!")

                # Show learning resources for missing skills
                if missing and not learning_resources.empty:
                    st.markdown("### 📚 Recommended Learning Resources")
                    st.markdown("Enhance your skills with these targeted learning resources:")
                    
                    resources = get_learning_resources_for_skills(missing, learning_resources)
                    if resources:
                        for i, resource in enumerate(resources, 1):
                            st.markdown(f"""
                            **{resource['Resource Name']}]({resource['URL']})** ({resource['Resource Type']})
                            - Platform: {resource['Platform']}
                            - Level: {resource['Difficulty Level']} | Duration: {resource['Duration']}
                            """)
                    else:
                        st.info("We're working on adding more learning resources for these skills. Check back soon!")
                
                # Show market demand for missing skills
                if missing and not trending_skills.empty:
                    st.markdown("### 📊 Market Demand for Missing Skills")
                    st.markdown("See how in-demand these skills are in the current job market:")
                    
                    # Find missing skills in trending data
                    missing_skill_info = []
                    for skill in missing:
                        # Try to find exact match first
                        match = trending_skills[trending_skills['Skill'].str.lower() == skill.lower()]
                        if match.empty:
                            # Try partial match
                            match = trending_skills[trending_skills['Skill'].str.contains(skill, case=False, na=False)]
                        
                        if not match.empty:
                            missing_skill_info.append({
                                'Skill': skill,
                                'Market Demand Score': match.iloc[0]['Market Demand Score'],
                                'Category': match.iloc[0]['Category']
                            })
                    
                    if missing_skill_info:
                        # Sort by market demand score
                        missing_skill_info = sorted(missing_skill_info, key=lambda x: x['Market Demand Score'], reverse=True)
                        
                        # Display as a table
                        demand_df = pd.DataFrame(missing_skill_info)
                        demand_df['Priority'] = demand_df['Market Demand Score'].apply(
                            lambda x: '🔴 High' if x >= 70 else '🟡 Medium' if x >= 40 else '🟢 Low'
                        )
                        
                        # Style the dataframe
                        def color_priority(val):
                            if 'High' in val:
                                return 'color: red; font-weight: bold'
                            elif 'Medium' in val:
                                return 'color: orange; font-weight: bold'
                            else:
                                return 'color: green; font-weight: bold'
                        
                        styled_demand = demand_df.style.applymap(color_priority, subset=['Priority'])
                        st.dataframe(styled_demand, use_container_width=True)
                        
                        # Recommendation
                        high_demand_missing = [s['Skill'] for s in missing_skill_info if s['Market Demand Score'] >= 70]
                        if high_demand_missing:
                            st.markdown(f"**💡 Recommendation:** Focus on learning these high-demand skills first: {', '.join(high_demand_missing[:3])}")
                    else:
                        st.info("Market demand information for your missing skills is currently unavailable.")

# Analytics Page
elif st.session_state.page == 'analytics':
    st.markdown('<div class="section-header">📊 Skill Analytics</div>', unsafe_allow_html=True)
    
    # Add a section for trending skills
    st.markdown('<div class="section-header">📈 Trending Skills in the Market</div>', unsafe_allow_html=True)
    st.markdown("Discover which skills are currently in high demand to optimize your career strategy.")
    
    # Display trending skills as a dataframe with color coding
    if not trending_skills.empty:
        # Create a styled dataframe
        trending_display = trending_skills.head(15).copy()
        trending_display['Rank'] = range(1, len(trending_display) + 1)
        trending_display = trending_display[['Rank', 'Skill', 'Market Demand Score', 'Category']]
        
        # Apply styling
        def color_score(val):
            """Color code the market demand score"""
            if val >= 80:
                color = '#4a86e8'  # High demand - blue
            elif val >= 60:
                color = '#6fa8dc'  # Medium demand - lighter blue
            elif val >= 40:
                color = '#9fc5e8'  # Moderate demand - even lighter blue
            else:
                color = '#cfe2f3'  # Lower demand - lightest blue
            return f'color: {color}; font-weight: bold'
        
        styled_df = trending_display.style.applymap(color_score, subset=['Market Demand Score'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Add a visualization
        st.markdown("### 🔥 Top 10 Trending Skills Visualization")
        top_trending = trending_skills.head(10)
        
        # Create a horizontal bar chart using Streamlit
        chart_data = top_trending[['Skill', 'Market Demand Score']].set_index('Skill')
        st.bar_chart(chart_data, use_container_width=True, height=400)
    else:
        st.info("Trending skills data is currently unavailable.")

# Career Paths Page
elif st.session_state.page == 'career':
    st.markdown('<div class="section-header">🧭 Career Path Visualization</div>', unsafe_allow_html=True)
    st.markdown("Explore potential career progression routes based on your current role and skills.")
    
    # Get unique roles for selection
    if not career_paths.empty:
        all_roles = sorted(list(set(career_paths['Current Role'].tolist() + career_paths['Next Role'].tolist())))
        selected_role = st.selectbox("Select your current or target role:", all_roles)
        
        if selected_role:
            # Find career paths from this role
            forward_paths = career_paths[career_paths['Current Role'] == selected_role]
            # Find career paths to this role
            backward_paths = career_paths[career_paths['Next Role'] == selected_role]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚀 Progression Paths")
                if not forward_paths.empty:
                    for _, path in forward_paths.iterrows():
                        st.markdown(f"""
                        **{path['Current Role']} → {path['Next Role']}**
                        - **Required Skills:** {path['Required Skills']}
                        - **Time to Achieve:** {path['Time to Achieve']}
                        - **Experience Needed:** {path['Years of Experience']}
                        """)
                        st.progress(0.5)  # Visual separator
                else:
                    st.info("No direct progression paths found from this role.")
            
            with col2:
                st.markdown("#### 🔙 Prerequisite Paths")
                if not backward_paths.empty:
                    for _, path in backward_paths.iterrows():
                        st.markdown(f"""
                        **{path['Current Role']} → {path['Next Role']}**
                        - **Required Skills:** {path['Required Skills']}
                        - **Time to Achieve:** {path['Time to Achieve']}
                        - **Experience Needed:** {path['Years of Experience']}
                        """)
                        st.progress(0.5)  # Visual separator
                else:
                    st.info("No prerequisite paths found for this role.")
            
            # Create a simple visualization of the career path
            st.markdown("#### 🗺️ Career Path Map")
            # Find all connected roles
            connected_roles = set()
            connected_roles.add(selected_role)
            
            # Add forward connections
            for _, path in forward_paths.iterrows():
                connected_roles.add(path['Next Role'])
            
            # Add backward connections
            for _, path in backward_paths.iterrows():
                connected_roles.add(path['Current Role'])
            
            # Display as a flow diagram using markdown
            roles_list = list(connected_roles)
            if len(roles_list) > 1:
                # Create a simple flow representation
                st.markdown("```\n" + " → ".join(roles_list) + "\n```")
    else:
        st.info("Career path data is currently unavailable.")

# Transfer Skills Page
elif st.session_state.page == 'transfer':
    st.markdown('<div class="section-header">🔄 Cross-Industry Skill Transfer</div>', unsafe_allow_html=True)
    st.markdown("Discover how your skills transfer across different industries and unlock new career opportunities.")
    
    if not transferable_skills.empty:
        # Get unique industries from the transferable skills data
        industries = list(set(transferable_skills['Industry 1'].tolist() + transferable_skills['Industry 2'].tolist()))
        industries = sorted([ind for ind in industries if pd.notna(ind)])
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_industry = st.selectbox("Select your current industry:", industries, key="current_industry")
        
        with col2:
            target_industry = st.selectbox("Select your target industry:", industries, key="target_industry")
        
        if current_industry and target_industry and current_industry != target_industry:
            # Find transferable skills between these industries
            direct_matches = transferable_skills[
                (transferable_skills['Industry 1'] == current_industry) & 
                (transferable_skills['Industry 2'] == target_industry)
            ]
            
            reverse_matches = transferable_skills[
                (transferable_skills['Industry 1'] == target_industry) & 
                (transferable_skills['Industry 2'] == current_industry)
            ]
            
            all_matches = pd.concat([direct_matches, reverse_matches])
            
            if not all_matches.empty:
                st.markdown(f"#### Transferable Skills: {current_industry} → {target_industry}")
                
                # Display transferable skills with similarity scores
                for _, match in all_matches.iterrows():
                    try:
                        similarity = float(match['Similarity Score'])
                    except:
                        similarity = 0
                    st.markdown(f"""
                    **{match['Core Skill']}**
                    - **Transfer Application:** {match['Transfer Application']}
                    - **Similarity Score:** {similarity:.2f}
                    """)
                    st.progress(similarity)
                    st.markdown("---")
            else:
                st.info(f"No direct transfer information found between {current_industry} and {target_industry}.")
        elif current_industry == target_industry:
            st.info("Please select different industries to see transferable skills.")
    else:
        st.info("Transferable skills data is currently unavailable.")

else:
    st.info("Please upload a resume or select skills to get recommendations")