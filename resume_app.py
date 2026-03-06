import streamlit as st
from src.preprocess import clean_text
from src.similarity import compute_tfidf_similarity, compute_bert_similarity
from src.skill_extractor import extract_missing_skills
from src.matcher import skill_gap   
from utils.file_handler import extract_text_from_pdf

st.set_page_config(page_title="AI Resume Matcher", layout="wide")

st.title("AI Resume – Job Description Matcher")

resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
jd_text = st.text_area("Paste Job Description")

if resume_file and jd_text:
    resume_text = extract_text_from_pdf(resume_file)

    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    tfidf_score = compute_tfidf_similarity(cleaned_resume, cleaned_jd)
    bert_score = compute_bert_similarity(cleaned_resume, cleaned_jd)

    st.subheader("Match Scores")
    col1, col2 = st.columns(2)

    col1.metric("TF-IDF Score", f"{tfidf_score}%")
    col2.metric("Semantic Score (MiniLM)", f"{bert_score}%")

    missing_skills = extract_missing_skills(cleaned_resume, cleaned_jd)

    st.subheader("Skill Gap Analysis")
    st.subheader("Skill Gap Analysis")

if missing_skills:
    for skill in missing_skills:
        st.markdown(f"- {skill}")
else:
    st.success("No major skill gaps detected!")