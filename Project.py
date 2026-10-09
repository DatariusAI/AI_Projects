"""Data Science Resume Analyzer: skill coverage and a keyword word cloud."""
import io
import re

import matplotlib.pyplot as plt
import pandas as pd
import pdfplumber
import plotly.express as px
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader
from wordcloud import WordCloud

st.set_page_config(page_title="Resume Skill Analyzer", page_icon="📄", layout="wide")

MAX_MB = 5

# skill -> regex alternatives (matched on whole words, case-insensitive)
SKILL_GROUPS = {
    "Programming": {
        "Python": r"python", "SQL": r"sql", "R": r"r(?!\s*&\s*d)", "Spark": r"(?:py)?spark",
    },
    "Machine learning": {
        "Machine Learning": r"machine learning|\bml\b", "Deep Learning": r"deep learning", "Statistics": r"statistic(?:s|al)",
        "Predictive Modeling": r"predictive model(?:l)?ing", "Time Series Forecasting": r"time[- ]series|forecasting",
        "NLP": r"nlp|natural language processing", "Customer Segmentation": r"segmentation",
        "Recommendation Engine": r"recommend(?:ation|er) (?:engine|system)s?",
    },
    "Data and cloud": {
        "Data Analysis": r"data analy(?:sis|tics)", "Data Visualization": r"data visuali[sz]ation",
        "Data Engineering": r"data engineering", "ETL": r"etl", "Big Data": r"big data", "AWS": r"aws|amazon web services",
        "Azure": r"azure", "Tableau": r"tableau", "Power BI": r"power ?bi",
    },
    "Soft skills": {
        "Communication": r"communication", "Analytical Thinking": r"analytical", "Problem Solving": r"problem[- ]solving",
        "Teamwork": r"teamwork|team player|collaborat\w+", "Leadership": r"leadership|led a team|managed a team",
    },
}

SAMPLE_RESUME = """Jane Doe - Data Scientist
Summary: Data scientist with 4 years of experience in machine learning, statistics and data visualization.
Skills: Python, SQL, scikit-learn, Deep Learning (PyTorch), Tableau, AWS, ETL pipelines with Airflow.
Experience:
- Built predictive modeling pipelines for customer churn and customer segmentation, improving retention by 8%.
- Developed a recommendation engine serving 2M users; ran A/B tests and communicated results to leadership.
- Time series forecasting of weekly demand with gradient boosting.
Education: MSc Data Science. Strong communication, teamwork and problem-solving skills.
"""


# ---------------------------------------------------------------- extraction
def extract_pdf(data: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:  # fallback parser
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    except Exception:
        return ""


def extract_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


@st.cache_data(show_spinner="Reading file…")
def extract_text(name: str, data: bytes) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_pdf(data)
    if ext == "docx":
        return extract_docx(data)
    return data.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------- analysis
def analyse(text: str, groups: dict) -> pd.DataFrame:
    low = re.sub(r"\s+", " ", text.lower())
    rows = []
    for group, skills in groups.items():
        for skill, pattern in skills.items():
            hits = len(re.findall(rf"(?<![\w]){pattern}(?![\w])", low))
            rows.append({"Group": group, "Skill": skill, "Mentions": hits, "Found": hits > 0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- UI
st.title("📄 Data Science Resume Skill Analyzer")
st.write(
    "Upload a resume to see which common data-science skills it mentions, which are missing, "
    "and a word cloud weighted by how often each skill appears. A sample resume is loaded so you can try it right away."
)

with st.sidebar:
    st.header("Settings")
    selected_groups = st.multiselect("Skill groups to check", list(SKILL_GROUPS), default=list(SKILL_GROUPS))
    extra = st.text_input("Extra skills (comma separated)", placeholder="e.g. Docker, LangChain, MLflow")
    show_text = st.checkbox("Show extracted text", value=False)

groups = {g: SKILL_GROUPS[g] for g in selected_groups}
extra_skills = [s.strip() for s in extra.split(",") if s.strip()][:20]
if extra_skills:
    groups["Custom"] = {s: re.escape(s.lower()) for s in extra_skills}

resume_file = st.file_uploader("Upload your resume (.pdf, .docx or .txt)", type=["pdf", "docx", "txt"])

if resume_file is None:
    st.info("Showing a sample resume. Upload your own file above to analyse it. Files are processed in memory and not stored.")
    resume_text, source = SAMPLE_RESUME, "Sample resume"
else:
    data = resume_file.getvalue()
    if len(data) > MAX_MB * 1024 * 1024:
        st.error(f"The file is larger than {MAX_MB} MB. Please upload a smaller file.")
        st.stop()
    try:
        resume_text = extract_text(resume_file.name, data)
    except Exception:
        st.error("Could not read this file. It may be corrupted or password protected. Try another format.")
        st.stop()
    source = resume_file.name
    if not resume_text.strip():
        st.warning("No text could be extracted. Scanned (image-only) PDFs are not supported. Try a .docx or text-based PDF.")
        st.stop()

if not groups:
    st.warning("Select at least one skill group in the sidebar.")
    st.stop()

result = analyse(resume_text, groups)
found = result[result["Found"]]
coverage = len(found) / len(result) if len(result) else 0

st.subheader(f"Results for: {source}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Skill coverage", f"{coverage:.0%}")
c2.metric("Skills found", f"{len(found)} / {len(result)}")
c3.metric("Total mentions", int(result["Mentions"].sum()))
c4.metric("Words in resume", len(resume_text.split()))

left, right = st.columns([1.1, 1])
with left:
    by_group = result.groupby("Group")["Found"].mean().reset_index(name="Coverage")
    fig = px.bar(by_group, x="Coverage", y="Group", orientation="h", range_x=[0, 1], text_auto=".0%",
                 title="Coverage by skill group")
    fig.update_layout(height=300, margin=dict(t=40, b=10), xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Found:** " + (", ".join(found["Skill"]) or "none"))
    missing = result[~result["Found"]]["Skill"].tolist()
    st.markdown("**Missing:** " + (", ".join(missing) or "none, nice work"))
    if missing:
        st.caption("Only add a skill if you really have it. Concrete project examples help more than keyword lists.")

with right:
    st.markdown("**Skill word cloud** (size = number of mentions)")
    freqs = dict(zip(found["Skill"], found["Mentions"]))
    if freqs:
        wc = WordCloud(width=900, height=500, background_color="white", colormap="viridis",
                       max_words=40, min_font_size=10, max_font_size=110, random_state=7).generate_from_frequencies(freqs)
        fig_wc, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc, width="stretch")
        plt.close(fig_wc)
    else:
        st.info("None of the selected skills were found, so there is nothing to draw yet.")

with st.expander("Skill table"):
    st.dataframe(result.sort_values(["Found", "Mentions"], ascending=False), hide_index=True, width="stretch")

if show_text:
    with st.expander("Extracted text", expanded=True):
        st.text(resume_text[:20000])

with st.expander("How it works"):
    st.markdown(
        """
1. Text is extracted with **pdfplumber** (falling back to **PyPDF2**) for PDFs, **python-docx** for Word files (including tables), or read directly for .txt.
2. Each skill is matched with a whole-word regular expression, so "R" does not match every letter r and "Power BI" matches "PowerBI".
3. Coverage = skills found ÷ skills checked. The word cloud is weighted by mention counts.
4. Nothing is uploaded to a third party. The file stays in the app's memory for your session.
"""
    )
