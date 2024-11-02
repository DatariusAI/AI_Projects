import streamlit as st
import pdfplumber  # Use pdfplumber instead of PyPDF2 or fitz
from docx import Document
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re

# List of keywords to check in the resume for data science relevance
KEY_SKILLS = [
    "Python", "Machine Learning", "Deep Learning", "SQL", "Data Visualization", 
    "Statistics", "Data Analysis", "Data Science", "Big Data", "Predictive Modeling",
    "R", "ETL", "Data Engineering", "Time Series Forecasting", "Customer Segmentation", 
    "Recommendation Engine", "AWS", "Spark", "Tableau", "PowerBI", "Communication",
    "Analytical Thinking", "Problem Solving", "Teamwork", "Leadership"
]

# Helper function for extracting text from PDFs using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function for extracting text from .docx files
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Preprocess text by lowering case and removing extra whitespace
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return text.lower().strip()

# Function to capture keywords present in the resume text
def capture_keywords(resume_text, key_skills):
    found_keywords = [skill for skill in key_skills if skill.lower() in resume_text]
    return found_keywords

# Streamlit App Interface
st.title("Data Science Resume Word Cloud Analyzer")

resume_file = st.file_uploader("Upload your resume (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

if resume_file is not None:
    # Determine file type and extract text
    file_type = resume_file.name.split('.')[-1].lower()
    if file_type == "pdf":
        resume_text = preprocess_text(extract_text_from_pdf(resume_file))
    elif file_type == "docx":
        resume_text = preprocess_text(extract_text_from_docx(resume_file))
    elif file_type == "txt":
        resume_text = preprocess_text(resume_file.getvalue().decode("utf-8"))
    else:
        st.error("Unsupported file format.")
        st.stop()
    
    # Display uploaded resume content
    st.write("Uploaded Resume Content:")
    st.write(resume_text)

    # Filter resume content to include only relevant keywords for word cloud
    found_keywords = capture_keywords(resume_text, KEY_SKILLS)
    filtered_text = " ".join(found_keywords)  # Only include captured keywords

    # Generate and display refined word cloud
    st.subheader("Refined Word Cloud of Relevant Data Science Keywords")
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=1000, height=500, background_color="white", colormap="viridis",
        stopwords=stopwords, max_words=30, contour_width=1, contour_color='steelblue',
        min_font_size=10, max_font_size=100
    ).generate(filtered_text)
    
    plt.figure(figsize=(
