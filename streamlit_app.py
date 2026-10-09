"""DatariusAI portfolio index: one page that links to every live demo."""
import streamlit as st

st.set_page_config(page_title="DatariusAI · AI Projects", page_icon="✨", layout="wide")

GITHUB_REPO = "https://github.com/DatariusAI/AI_Projects"
HF_PROFILE = "https://huggingface.co/DatariusAI"

# (name, url, one-line description, tags)
STREAMLIT_APPS = {
    "Machine learning for finance and customers": [
        ("Fraud Risk Scoring", "https://kibsfraudrisk.streamlit.app/", "Upload transactions and score each one for fraud risk.", "classification"),
        ("Loan Acceptance Prediction", "https://kibsloanacceptance.streamlit.app/", "Predict loan acceptance from applicant features.", "classification"),
        ("Customer Segmentation", "https://kibssegments.streamlit.app/", "Group customers into segments with k-means clustering.", "clustering"),
        ("ChurnGuard", "https://churnguard2025.streamlit.app/", "Churn prediction dashboard with explainable drivers.", "classification · SHAP"),
        ("Melbourne House Prices", "https://deakinhousepricesmalbourne.streamlit.app/", "Estimate a property price from its features.", "regression"),
        ("AAPL vs MSFT (BQL) Explorer", "https://bqntbql.streamlit.app/", "Interactive investment comparison dashboard.", "finance · time series"),
        ("Ambiscions Case Study", "https://kewlfunky2023-blank-app-ambiscions-case-study-teoy6m.streamlit.app/", "Retail analytics: descriptive, diagnostic, predictive and prescriptive.", "analytics · XGBoost"),
    ],
    "Recommender systems": [
        ("GCNN Recommender", "https://gcnnrecommender01.streamlit.app/", "Matrix factorisation vs. graph propagation, with Hit-Rate and NDCG.", "graph ML"),
        ("Movie Recommender", "https://blank-app-kpkm6rp3yrcu8mpp7tk3yj.streamlit.app/", "Rate a few films and get recommendations.", "collaborative filtering"),
    ],
    "NLP, LLMs and assistants": [
        ("Arabic Assistant", "https://arabic-assitant.streamlit.app/", "Arabic NLP toolkit: summary, sentiment, dialect and retrieval.", "Arabic NLP"),
        ("Arabic Assistant 2025", "https://arabic-assistant202.streamlit.app/", "Arabic chat assistant with optional LLM answers.", "Arabic NLP · LLM"),
        ("Insurance Claims Assistant (Arabic)", "https://medicalchatbot2025.streamlit.app/", "Claims helpdesk: FAQ, hospital finder and claim status.", "chatbot · Arabic"),
        ("AI Knowledge Repo", "https://medicalchatbotfinal2025.streamlit.app/", "Knowledge assistant demo.", "chatbot"),
        ("Nutrition Disorder Agent", "https://nutritiondisorderagent.streamlit.app/", "LLM chat assistant for nutrition-related questions.", "LLM · FastAPI"),
        ("Resume Keyword Analyzer", "https://blank-app-wx9ckhjjtpzen6zs4qqfum.streamlit.app/", "Check a resume against data-science skills, with a word cloud.", "text mining"),
        ("Dubai Space Travel", "https://dubaispacetravel.streamlit.app/", "Hackathon booking app with a travel assistant.", "prototype"),
    ],
}

HF_SPACES = [
    ("Pneumonia Detection AI", "PneumoniaDetectionCV", "Deep learning on chest X-rays."),
    ("Chest X-ray", "chest_X-ray", "Chest X-ray image classification."),
    ("AUB Admissions Assistant (RAG)", "Multimodal_Virtual_Assistant_with_RAG_Project", "Multimodal RAG assistant for admissions questions."),
    ("Multimodal Virtual Assistant with RAG", "Multimodal_Virtual_Assistant_with_RAG", "English/Arabic RAG with OCR and voice, open models only."),
    ("Advanced Backtesting Platform", "advance_backtesting_platform", "Backtest trading strategies."),
    ("ChurnGuard (HF)", "churn", "Churn prediction on Hugging Face."),
    ("FoodHub Chatbot", "FoodHub_Chatbot", "Customer-support chatbot for a food-delivery app."),
    ("Sentiment Analysis", "sentiment-analysis-gradio", "Text sentiment with a Gradio UI."),
    ("AzureML Pizza Sales Analysis", "AzureML_Pizza_Sales_Analysis", "Sales analysis built on Azure ML."),
    ("Intelligent Reporting on Azure", "IntelligentreportingonAzure", "Automated reporting on Azure."),
    ("Python for Generative AI", "PythonforGenerativeAI", "Generative AI exercises in Python."),
    ("GreatLearning LLM Projects", "GreatLearning", "Course projects with LLMs."),
    ("AI Projects (HF)", "AI_Projects", "Hugging Face mirror of this repository."),
]


def card(name, url, desc, tag=None, key=None):
    with st.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(desc)
        if tag:
            st.markdown(f":gray-badge[{tag}]")
        st.link_button("Open app ↗", url, width="stretch", key=key)


st.title("✨ DatariusAI · AI Projects")
st.write(
    "Live demos of machine learning, NLP and LLM apps built with Python and Streamlit. "
    "Every app runs in the browser. Most work without an account or API key, and the source is on GitHub."
)

c1, c2, c3 = st.columns(3)
c1.metric("Streamlit apps", sum(len(v) for v in STREAMLIT_APPS.values()))
c2.metric("Hugging Face Spaces", len(HF_SPACES))
c3.metric("Focus areas", "ML · NLP · LLM")

b1, b2, _ = st.columns([1, 1, 2])
b1.link_button("GitHub repo: AI_Projects", GITHUB_REPO, icon="💻", width="stretch")
b2.link_button("Hugging Face profile", HF_PROFILE, icon="🤗", width="stretch")

search = st.text_input("Filter apps", placeholder="Try: Arabic, fraud, recommender, LLM").strip().lower()


def matches(*fields):
    return not search or any(search in str(f).lower() for f in fields)


tab_st, tab_hf = st.tabs(["Streamlit apps", "Hugging Face Spaces"])

with tab_st:
    shown = 0
    for section, apps in STREAMLIT_APPS.items():
        apps = [a for a in apps if matches(section, *a)]
        if not apps:
            continue
        st.subheader(section)
        cols = st.columns(3)
        for i, (name, url, desc, tag) in enumerate(apps):
            with cols[i % 3]:
                card(name, url, desc, tag, key=f"st_{section}_{i}")
        shown += len(apps)
    if not shown:
        st.info("No Streamlit app matches that filter.")
    st.caption("Apps on the free Streamlit tier sleep when idle. If one shows a wake-up screen, click the button and give it about 30 seconds.")

with tab_hf:
    spaces = [s for s in HF_SPACES if matches(*s)]
    if not spaces:
        st.info("No Space matches that filter.")
    cols = st.columns(3)
    for i, (name, sid, desc) in enumerate(spaces):
        with cols[i % 3]:
            card(name, f"https://huggingface.co/spaces/DatariusAI/{sid}", desc, key=f"hf_{i}")

with st.expander("How this repo is organised"):
    st.markdown(
        f"""
- Each app is a single Streamlit script in the root of [{GITHUB_REPO.split('/')[-1]}]({GITHUB_REPO}), deployed separately on Streamlit Community Cloud.
- Trained models are stored as `.joblib` files next to the scripts, and small sample datasets are included so every demo works out of the box.
- LLM features read their API keys from Streamlit secrets or environment variables. Without a key the app shows a notice and falls back to an offline mode.
- To run one locally: `pip install -r requirements.txt` then `streamlit run <app>.py`.
"""
    )

st.caption("Built by DatariusAI · [GitHub](https://github.com/DatariusAI) · [Hugging Face](https://huggingface.co/DatariusAI)")
