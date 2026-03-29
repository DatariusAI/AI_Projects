<div align="center">

# :sparkles: AI Projects

### A collection of production-ready AI and ML applications built with Streamlit, FastAPI, and HuggingFace

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

</div>

---

## Overview

A curated collection of end-to-end AI and machine learning applications, each built as a fully interactive Streamlit web app. Projects span natural language processing (Arabic), fraud detection, financial modeling, customer analytics, computer vision, and intelligent recommendation systems. Most apps are containerized with Docker and deployable to HuggingFace Spaces.

## Projects

| # | Project | Description | Tech | File |
|---|---------|-------------|------|------|
| 1 | **Arabic Assistant** | Arabic-language AI assistant with conversational capabilities | LangChain, OpenAI | `Arabic_Assistant.py`, `Arabic_Assistant2025.py` |
| 2 | **Fraud Risk Detector** | Real-time fraud risk scoring using trained ML models | scikit-learn, joblib | `Fraud_Risk.py` |
| 3 | **Loan Acceptance Predictor** | Predict loan approval probability based on applicant features | scikit-learn, XGBoost | `Loan_Acceptance.py` |
| 4 | **Customer Churn Predictor** | Identify customers likely to churn with explainable predictions | scikit-learn, SHAP | `churn_app.py` |
| 5 | **House Price Estimator** | Predict house prices with interactive feature inputs | Random Forest, XGBoost | `HousePrices streamlit app.py` |
| 6 | **Nutrition Agent** | AI-powered nutrition analysis and meal planning agent | LangChain, OpenAI | `NutritionAgent_HF_Docker_FIXED.zip` |
| 7 | **Recommender System (GCNN)** | Graph-based recommendation engine using Graph Convolutional Networks | PyTorch, GCN | `Graded_Assignment_Project_Recommender_GCNN.py` |
| 8 | **Customer Segmentation** | K-Means clustering for customer segmentation with visualization | scikit-learn, K-Means | `segementation.py` |
| 9 | **Insurance Claims System** | Arabic-language insurance claims: FAQ, hospital finder, claims submission | Streamlit, pandas | `1_...py`, `2_...py`, `3_...py`, `4_...py` |
| 10 | **Dubai Space App** | Dubai-themed space exploration data application | Streamlit | `DubaiSpace.py` |
| 11 | **Ambiscions Case Study** | Business analytics case study with interactive dashboard | Streamlit, pandas | `Ambiscions_Case_Study.py` |
| 12 | **BQL Analytics** | Business Query Language analytics tool | Streamlit | `BQL.py` |

## Pre-Trained Models

| Model | File | Algorithm |
|-------|------|-----------|
| Fraud Detection | `fraud_model.joblib` | Classification |
| Loan Approval | `loan_model.joblib` | Classification |
| House Prices (RF) | `rf_final.joblib` | Random Forest |
| House Prices (XGB) | `xgb_final.joblib` | XGBoost |
| Customer Segments | `kmeans_model.joblib` | K-Means Clustering |
| Segmentation Scaler | `segmentation_scaler.joblib` | StandardScaler |

## Tech Stack

| Category | Tools |
|----------|-------|
| **Languages** | Python |
| **Web Framework** | Streamlit, FastAPI |
| **ML Libraries** | scikit-learn, XGBoost, LightGBM |
| **LLM / NLP** | LangChain, OpenAI |
| **Data Processing** | pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Docker, HuggingFace Spaces |
| **Model Persistence** | joblib |

## Project Structure

```
AI_Projects/
|-- Arabic_Assistant.py             # Arabic AI assistant
|-- Arabic_Assistant2025.py         # Updated Arabic assistant
|-- Fraud_Risk.py                   # Fraud detection app
|-- Loan_Acceptance.py              # Loan prediction app
|-- churn_app.py                    # Churn prediction app
|-- HousePrices streamlit app.py    # House price predictor
|-- segementation.py                # Customer segmentation
|-- recommender_app.py              # Recommendation engine
|-- Graded_Assignment_Project_Recommender_GCNN.py  # GCN recommender
|-- DubaiSpace.py                   # Dubai space app
|-- Ambiscions_Case_Study.py        # Business case study
|-- BQL.py                          # BQL analytics
|-- 1_..._FAQ.py                    # Insurance FAQ (Arabic)
|-- 2_..._Hospital.py               # Hospital finder (Arabic)
|-- 3_..._Claims.py                 # Claims submission (Arabic)
|-- 4_..._Status.py                 # Claims status (Arabic)
|-- pages/                          # Streamlit multi-page components
|-- *.joblib                        # Pre-trained model files
|-- *.csv                           # Data files
|-- Dockerfile                      # Container configuration
|-- requirements.txt                # Python dependencies
|-- .huggingface.yaml               # HuggingFace Spaces config
|-- runtime.txt                     # Runtime specification
`-- README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/DatariusAI/AI_Projects.git
cd AI_Projects

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run any Streamlit app
streamlit run Fraud_Risk.py
streamlit run churn_app.py
streamlit run Loan_Acceptance.py
```

### Run with Docker

```bash
docker build -t ai-projects .
docker run -p 8501:8501 ai-projects
```

## Deployment

Most apps are configured for deployment on **HuggingFace Spaces** via the included `.huggingface.yaml` and `Dockerfile`. Push to a HuggingFace Space repository to deploy automatically.

## Roadmap

- [ ] Add unit tests for model prediction endpoints
- [ ] Consolidate requirements.txt (remove duplicates)
- [ ] Add demo screenshots and GIFs
- [ ] Deploy all apps to HuggingFace Spaces with live links
- [ ] Add API documentation for FastAPI endpoints

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Part of the [DatariusAI](https://github.com/DatariusAI) portfolio**

</div>
