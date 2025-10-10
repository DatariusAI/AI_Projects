import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load model and feature info
# ----------------------------
model = joblib.load("/content/rf_final.joblib")

try:
    MODEL_FEATURES = list(model.feature_names_in_)
except AttributeError:
    MODEL_FEATURES = [
        "Tenure","Complain_ly","cashback",
        "CC_Agent_Score","CC_Contacted_LY",
        "rev_growth_yoy","Service_Score"
    ]

THRESHOLDS = {"low": 0.3, "medium": 0.6, "high": 0.9}

# ----------------------------
# Data preparation helper
# ----------------------------
def prepare_input(df):
    df = df.copy()
    if "Complain_ly" in df.columns:
        df["Complain_ly"] = df["Complain_ly"].replace({"Yes": 1, "No": 0})
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[MODEL_FEATURES]
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df

# ----------------------------
# Classification helper
# ----------------------------
def classify_risk(prob):
    if prob < THRESHOLDS["low"]:
        return "🟢 Safe", "Customer loyalty strong — minimal churn risk.", "Low"
    elif prob < THRESHOLDS["medium"]:
        return "🟠 Caution", "Moderate churn risk — monitor satisfaction indicators.", "Medium"
    else:
        return "🔴 High Risk", "Customer likely to churn — immediate retention action needed.", "High"

# ----------------------------
# Single Prediction
# ----------------------------
def predict_single(tenure, complain, cashback, agent_score, contact_count, rev_growth, service_score):
    complain_value = 1 if complain == "Yes" else 0
    df = pd.DataFrame([{
        "Tenure": tenure,
        "Complain_ly": complain_value,
        "cashback": cashback,
        "CC_Agent_Score": agent_score,
        "CC_Contacted_LY": contact_count,
        "rev_growth_yoy": rev_growth,
        "Service_Score": service_score
    }])
    df_prepared = prepare_input(df)
    prob = model.predict_proba(df_prepared)[0][1]
    indicator, advice, level = classify_risk(prob)
    return indicator, prob, advice, level

# ----------------------------
# Batch Prediction
# ----------------------------
def predict_batch(file):
    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    df_prepared = prepare_input(df)
    probs = model.predict_proba(df_prepared)[:, 1]
    risk_labels, advice_list, levels = [], [], []
    for p in probs:
        indicator, advice, level = classify_risk(p)
        risk_labels.append(indicator)
        advice_list.append(advice)
        levels.append(level)

    df["Churn_Probability"] = probs.round(2)
    df["Risk_Level"] = levels
    df["Risk_Indicator"] = risk_labels
    df["Business_Advice"] = advice_list
    df = df.sort_values("Churn_Probability", ascending=False).reset_index(drop=True)
    return df

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Customer Churn Prediction Dashboard", page_icon="📊", layout="wide")

st.markdown("<h1 style='text-align:center; color:#004aad;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Predict churn likelihood with visual risk indicators and business guidance.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔹 Single Prediction", "📁 Batch Prediction"])

with tab1:
    st.subheader("Single Customer Prediction")
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.number_input("Tenure (months)", min_value=0, value=12)
        complain = st.radio("Complain Last Year?", ["Yes", "No"], index=1)
        cashback = st.number_input("Cashback Amount", min_value=0.0, value=25.0)
        agent_score = st.number_input("Agent Score (1–10)", min_value=0.0, value=8.0)
    with c2:
        contact_count = st.number_input("Contact Count Last Year", min_value=0, value=3)
        rev_growth = st.number_input("Revenue Growth YoY (%)", value=5.0)
        service_score = st.number_input("Service Score (1–10)", value=9.0)

    if st.button("🔍 Predict Churn Risk"):
        indicator, prob, advice, level = predict_single(
            tenure, complain, cashback, agent_score, contact_count, rev_growth, service_score
        )
        st.markdown(f"### {indicator} ({level} Risk)")
        st.markdown(f"**Churn Probability:** {prob:.2f}")
        st.info(advice)

with tab2:
    st.subheader("Batch Prediction (CSV or Excel)")
    st.markdown("""
    Upload a CSV or Excel file containing customer data.  
    Risk levels:
    - 🟢 **Safe:** Probability < 0.3  
    - 🟠 **Caution:** 0.3 ≤ Probability < 0.6  
    - 🔴 **High Risk:** ≥ 0.6
    """)
    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df_results = predict_batch(uploaded_file)
        st.dataframe(df_results)

st.markdown("<hr><center><small>Built with ❤️ using Streamlit — AI-Powered Business Retention Analytics</small></center>", unsafe_allow_html=True)
