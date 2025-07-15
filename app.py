
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
st.markdown("""
    <style>
        .main-title {
            font-size: 3em;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2em;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.4em;
            color: #2980b9;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #ecf0f1;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s;
            display: block;
            margin: 1rem auto;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .match-score {
            font-weight: bold;
        }
        .high-match { color: #2db4f7; }
        .medium-match { color: #3498db; }
        .low-match { color: #2980b9; }
        .skill-tag {
            display: inline-block;
            background-color: #2db4f7;
            padding: 0.2rem 0.5rem;
            border-radius: 15px;
            margin: 0.2rem;
            font-size: 0.9em;
        }
        .matched { background-color: #2db4f7; }
        .missing { background-color: #fadbd8; }
        div.stButton {
            margin: 1rem 0;
        }
        table, th, td {
            border: 1px solid #ddd;
            border-collapse: collapse;
            padding: 8px;
        }
    </style>
""", unsafe_allow_html=True)

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
        df['All Skills'] = df['Technical Skills'] + ', ' + df['Soft Skills']
        df['Processed Skills'] = df['All Skills'].apply(preprocess)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Extract text from uploaded PDF resume
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with st.spinner("📄 Extracting text from PDF..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
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

# --- Main UI ---
st.markdown('<div class="main-title">💼 Skill-Based Job Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find your perfect job match based on your skills</div>', unsafe_allow_html=True)

# Load and preprocess data
df = load_data()
if df is not None:
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
            role_skills = set([s.strip() for s in (role_data['Technical Skills'] + ", " + role_data['Soft Skills']).split(',')])
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

            progress = len(matched)/len(role_skills)
            st.progress(progress, text=f"Skill Match: {progress*100:.1f}%")
            st.markdown("---")
            st.markdown("**📋 Role Details:**")
            st.markdown(f"**Job Role:** {role_data['Job Role']}")
            st.markdown(f"**Technical Skills:** {role_data['Technical Skills']}")
            st.markdown(f"**Soft Skills:** {role_data['Soft Skills']}")
            st.markdown(f"**Match Score:** {format_score(role_data['Match Score'])}")
else:
    st.info("Please upload a resume or select skills to get recommendations")
