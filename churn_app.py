import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# ==============================================================
#                   MODEL LOADING
# ==============================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_final.joblib")
GITHUB_RAW_URL = "https://raw.githubusercontent.com/DatariusAI/AI_Projects/main/rf_final.joblib"


@st.cache_resource(show_spinner="Loading churn model…")
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    response = requests.get(GITHUB_RAW_URL, timeout=60)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))


try:
    model = load_model()
except Exception as exc:
    st.error(f"The churn model could not be loaded ({exc}). Please try again later.")
    st.stop()

MODEL_FEATURES = list(model.feature_names_in_)
THRESHOLDS = {"low": 0.3, "medium": 0.6}

# The model was trained on preprocessed data: Tenure and Complain_ly are raw, the categorical
# fields are label-encoded integers, and the remaining numeric fields are standardised (z-scores).
# City tier and agent score scales below were recovered from the model's own split points.
CITY_TIER_MEAN, CITY_TIER_STD = 1.65, 0.917
AGENT_SCORE_MEAN, AGENT_SCORE_STD = 3.07, 1.372

TYPICAL_CUSTOMER = {
    "Tenure": 10, "Payment": 2, "Account_user_count": 0.0, "account_segment": 3,
    "Marital_Status": 1, "Login_device": 1, "cashback": 0.0, "Day_Since_CC_connect": 0.0,
    "rev_per_month": 0.0, "rev_growth_yoy": 0.0,
    "City_Tier": (1 - CITY_TIER_MEAN) / CITY_TIER_STD, "CC_Contacted_LY": 0.0,
    "CC_Agent_Score": 0.0, "Complain_ly": 0, "Engagement_Intensity": 0.0,
}
PRETTY = {
    "Tenure": "Tenure (months)", "Payment": "Payment method", "Account_user_count": "Users on account",
    "account_segment": "Account segment", "Marital_Status": "Marital status", "Login_device": "Login device",
    "cashback": "Cashback", "Day_Since_CC_connect": "Days since last support contact",
    "rev_per_month": "Revenue per month", "rev_growth_yoy": "Revenue growth YoY", "City_Tier": "City tier",
    "CC_Contacted_LY": "Support contacts last year", "CC_Agent_Score": "Agent score",
    "Complain_ly": "Complained last year", "Engagement_Intensity": "Engagement intensity",
}
LEVEL_OPTIONS = {"Much lower": -1.5, "Lower": -0.75, "Average": 0.0, "Higher": 0.75, "Much higher": 1.5}

# ==============================================================
#                   HELPERS
# ==============================================================


def prepare_input(df):
    """Map Yes/No complaints, fill any missing feature with the typical value, order columns."""
    df = df.copy()
    if "Complain_ly" in df.columns and df["Complain_ly"].dtype == object:
        yes_no = df["Complain_ly"].map({"Yes": 1, "No": 0})
        df["Complain_ly"] = yes_no.fillna(pd.to_numeric(df["Complain_ly"], errors="coerce"))
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = TYPICAL_CUSTOMER[col]
    df = df[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(pd.Series(TYPICAL_CUSTOMER))
    return df


def classify_risk(prob):
    if prob < THRESHOLDS["low"]:
        return "🟢 Safe", "Customer loyalty strong — minimal churn risk.", "Low"
    if prob < THRESHOLDS["medium"]:
        return "🟠 Caution", "Moderate churn risk — monitor satisfaction indicators.", "Medium"
    return "🔴 High Risk", "Customer likely to churn — immediate retention action needed.", "High"


def drivers(row):
    """Change in churn probability when each feature is reset to the typical customer's value."""
    base_prob = model.predict_proba(row)[0, 1]
    variants = pd.concat([row] * len(MODEL_FEATURES), ignore_index=True)
    for i, col in enumerate(MODEL_FEATURES):
        variants.loc[i, col] = TYPICAL_CUSTOMER[col]
    probs = model.predict_proba(variants)[:, 1]
    return pd.Series(base_prob - probs, index=MODEL_FEATURES)


@st.cache_data
def sample_customers(n=250, seed=21):
    """Synthetic customers already in the model's preprocessed feature space."""
    rng = np.random.default_rng(seed)
    tiers = rng.choice([1, 2, 3], n, p=[0.6, 0.1, 0.3])
    agent = rng.integers(1, 6, n)
    return pd.DataFrame({
        "CustomerID": [f"CUST{20000 + i}" for i in range(n)],
        "Tenure": rng.choice(np.r_[0:6, 0:37], n),
        "Payment": rng.integers(0, 5, n),
        "Account_user_count": rng.normal(0, 1, n).round(2),
        "account_segment": rng.integers(0, 7, n),
        "Marital_Status": rng.integers(0, 3, n),
        "Login_device": rng.integers(0, 3, n),
        "cashback": rng.normal(0, 1, n).round(2),
        "Day_Since_CC_connect": rng.normal(0, 1, n).round(2),
        "rev_per_month": rng.normal(0, 1, n).round(2),
        "rev_growth_yoy": rng.normal(0, 1, n).round(2),
        "City_Tier": ((tiers - CITY_TIER_MEAN) / CITY_TIER_STD).round(3),
        "CC_Contacted_LY": rng.normal(0, 1, n).round(2),
        "CC_Agent_Score": ((agent - AGENT_SCORE_MEAN) / AGENT_SCORE_STD).round(3),
        "Complain_ly": rng.choice([0, 1], n, p=[0.72, 0.28]),
        "Engagement_Intensity": rng.normal(0, 1, n).round(2),
    })


def predict_batch(df):
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    probs = model.predict_proba(prepare_input(df))[:, 1]
    out = df.copy()
    out["Churn_Probability"] = probs.round(3)
    risk = [classify_risk(p) for p in probs]
    out["Risk_Level"] = [r[2] for r in risk]
    out["Risk_Indicator"] = [r[0] for r in risk]
    out["Business_Advice"] = [r[1] for r in risk]
    return out.sort_values("Churn_Probability", ascending=False).reset_index(drop=True), missing


# ==============================================================
#                   STREAMLIT DASHBOARD
# ==============================================================

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    "Predicts how likely a customer is to leave, using a **Random Forest** trained on telecom/DTH "
    "customer data, and turns the score into a risk level with a suggested action. Adjust the customer "
    "profile below, or score a whole portfolio in the batch tab."
)

with st.expander("How it works"):
    st.markdown(
        f"""
- **Algorithm:** scikit-learn `RandomForestClassifier` with {model.n_estimators} trees and
  balanced class weights.
- **Inputs:** {len(MODEL_FEATURES)} features. Tenure and complaints are raw values, categorical fields
  are label-encoded codes, and the other numeric fields were standardised, so the form asks for them
  relative to an average customer.
- **Risk levels:** 🟢 Safe < {THRESHOLDS['low']}, 🟠 Caution {THRESHOLDS['low']}–{THRESHOLDS['medium']},
  🔴 High Risk ≥ {THRESHOLDS['medium']}.
- **Drivers:** each feature is reset to the typical customer's value and the change in churn
  probability is measured. Bars to the right push this customer towards churn.
"""
    )

tab1, tab2, tab3 = st.tabs(["🔹 Single prediction", "📁 Batch prediction", "📈 Model insights"])

# ---------------- Single Prediction ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 60, 2)
        complain = st.radio("Complained last year?", ["Yes", "No"], index=0, horizontal=True)
        city_tier = st.radio("City tier", [1, 2, 3], horizontal=True)
        agent_score = st.slider("Support agent score (1–5)", 1, 5, 3)
    with c2:
        cashback = st.select_slider("Cashback received", list(LEVEL_OPTIONS), "Lower")
        contacts = st.select_slider("Support contacts last year", list(LEVEL_OPTIONS), "Average")
        days_since = st.select_slider("Days since last support contact", list(LEVEL_OPTIONS), "Average")
        users = st.select_slider("Users on the account", list(LEVEL_OPTIONS), "Average")
    with c3:
        rev_month = st.select_slider("Revenue per month", list(LEVEL_OPTIONS), "Average")
        rev_growth = st.select_slider("Revenue growth YoY", list(LEVEL_OPTIONS), "Average")
        engagement = st.select_slider("Engagement intensity", list(LEVEL_OPTIONS), "Average")

    with st.expander("Advanced: label-encoded categorical fields"):
        a1, a2, a3, a4 = st.columns(4)
        payment = a1.number_input("Payment method code", 0, 4, TYPICAL_CUSTOMER["Payment"])
        segment = a2.number_input("Account segment code", 0, 6, TYPICAL_CUSTOMER["account_segment"])
        marital = a3.number_input("Marital status code", 0, 2, TYPICAL_CUSTOMER["Marital_Status"])
        device = a4.number_input("Login device code", 0, 2, TYPICAL_CUSTOMER["Login_device"])

    customer = prepare_input(pd.DataFrame([{
        "Tenure": tenure, "Payment": payment, "Account_user_count": LEVEL_OPTIONS[users],
        "account_segment": segment, "Marital_Status": marital, "Login_device": device,
        "cashback": LEVEL_OPTIONS[cashback], "Day_Since_CC_connect": LEVEL_OPTIONS[days_since],
        "rev_per_month": LEVEL_OPTIONS[rev_month], "rev_growth_yoy": LEVEL_OPTIONS[rev_growth],
        "City_Tier": (city_tier - CITY_TIER_MEAN) / CITY_TIER_STD,
        "CC_Contacted_LY": LEVEL_OPTIONS[contacts],
        "CC_Agent_Score": (agent_score - AGENT_SCORE_MEAN) / AGENT_SCORE_STD,
        "Complain_ly": complain, "Engagement_Intensity": LEVEL_OPTIONS[engagement],
    }]))
    prob = float(model.predict_proba(customer)[0, 1])
    indicator, advice, level = classify_risk(prob)

    r1, r2 = st.columns([1, 2])
    with r1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": "Churn probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                   "steps": [{"range": [0, 30], "color": "#C8E6C9"},
                             {"range": [30, 60], "color": "#FFE0B2"},
                             {"range": [60, 100], "color": "#FFCDD2"}]},
        ))
        gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, width="stretch")
        st.markdown(f"### {indicator}")
        st.info(advice)
    with r2:
        d = drivers(customer)
        d = d[d.abs() >= 0.005].sort_values()
        if d.empty:
            st.info("This customer looks like the typical customer, so no single feature stands out.")
        else:
            ddf = pd.DataFrame({"feature": [PRETTY[i] for i in d.index], "effect": d.values})
            fig_d = px.bar(ddf, x="effect", y="feature", orientation="h", color=ddf["effect"] > 0,
                           color_discrete_map={True: "#C62828", False: "#2E7D32"},
                           title="What drives this score (change vs. a typical customer)")
            fig_d.update_layout(showlegend=False, height=340, margin=dict(t=40, b=10),
                                xaxis_title="Change in churn probability", yaxis_title="")
            fig_d.update_xaxes(tickformat="+.0%")
            st.plotly_chart(fig_d, width="stretch")

# ---------------- Batch Prediction ----------------
with tab2:
    st.markdown(
        f"""
Upload a CSV or Excel file with the model's {len(MODEL_FEATURES)} preprocessed features
(`{"`, `".join(MODEL_FEATURES)}`). Extra columns such as a customer ID are kept.
No file? The app scores 250 synthetic sample customers.
"""
    )
    sample = sample_customers()
    st.download_button("Download CSV template (sample data)", sample.to_csv(index=False).encode("utf-8"),
                       file_name="churn_sample.csv", mime="text/csv")
    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

    raw, source = sample, "built-in sample"
    if uploaded_file is not None:
        try:
            raw = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
            source = uploaded_file.name
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
            st.stop()

    if raw.empty:
        st.error("The file has no rows.")
        st.stop()

    df_results, missing = predict_batch(raw)
    if len(missing) == len(MODEL_FEATURES):
        st.error("None of the model's feature columns were found. Download the template to see the expected format.")
        st.stop()
    if missing:
        st.warning(f"Missing columns filled with typical-customer values: {', '.join(missing)}.")

    st.caption(f"Scoring: {source}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(df_results):,}")
    k2.metric("🔴 High risk", f"{(df_results['Risk_Level'] == 'High').sum():,}")
    k3.metric("🟠 Caution", f"{(df_results['Risk_Level'] == 'Medium').sum():,}")
    k4.metric("Average churn probability", f"{df_results['Churn_Probability'].mean():.0%}")

    g1, g2 = st.columns(2)
    with g1:
        fig_r = px.pie(df_results, names="Risk_Level", hole=0.45, title="Customers by risk level",
                       color="Risk_Level",
                       color_discrete_map={"Low": "#66BB6A", "Medium": "#FFA726", "High": "#EF5350"})
        st.plotly_chart(fig_r, width="stretch")
    with g2:
        fig_t = px.box(df_results, x="Complain_ly", y="Churn_Probability", points="outliers",
                       title="Churn probability by complaint status",
                       labels={"Complain_ly": "Complained last year (1 = yes)"})
        st.plotly_chart(fig_t, width="stretch")

    st.dataframe(df_results, width="stretch", hide_index=True)
    st.download_button("Download scored customers", df_results.to_csv(index=False).encode("utf-8"),
                       file_name="churn_predictions.csv", mime="text/csv")

# ---------------- Model insights ----------------
with tab3:
    imp = pd.DataFrame({"feature": [PRETTY[f] for f in MODEL_FEATURES],
                        "importance": model.feature_importances_}).sort_values("importance")
    fig_i = px.bar(imp, x="importance", y="feature", orientation="h",
                   title="Global feature importance (mean decrease in impurity)")
    fig_i.update_layout(height=480, yaxis_title="")
    st.plotly_chart(fig_i, width="stretch")
    st.caption("Tenure is by far the strongest signal: new customers churn much more often than long-standing ones.")

st.markdown("---")
st.caption("Built with Streamlit and scikit-learn · AI-powered business retention analytics")
