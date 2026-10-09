import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.tree import export_text

st.set_page_config(page_title="Fraud Risk Prediction", page_icon="🛡️", layout="wide")

NUMERIC_COLS = [
    "transaction_amount",
    "account_age_days",
    "device_trust_score",
    "num_prev_transactions",
    "location_match",
]
CATEGORICAL_COLS = ["time_of_day", "channel"]
# Levels seen in training. The first level of each was dropped by one-hot encoding (drop_first=True).
TIME_LEVELS = ["Afternoon", "Evening", "Morning", "Night"]
CHANNEL_LEVELS = ["ATM", "Branch", "Mobile", "Web"]


@st.cache_resource
def load_model():
    return joblib.load("fraud_model.joblib")


@st.cache_data
def sample_transactions(n=200, seed=7):
    """Synthetic transactions in the same schema the model was trained on."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "transaction_amount": rng.gamma(2.0, 150.0, n).round(2),
        "account_age_days": rng.integers(1, 3650, n),
        "device_trust_score": rng.beta(4, 1.5, n).round(2),
        "num_prev_transactions": rng.integers(0, 200, n),
        "location_match": rng.choice([0, 1], n, p=[0.25, 0.75]),
        "time_of_day": rng.choice(TIME_LEVELS, n),
        "channel": rng.choice(CHANNEL_LEVELS, n),
    })


def encode(df, expected_features):
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    return encoded.reindex(columns=expected_features, fill_value=0).astype(float)


def validate(df):
    missing = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c not in df.columns]
    if missing:
        return f"Missing required column(s): {', '.join(missing)}."
    bad = [c for c in NUMERIC_COLS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        return f"These columns must be numeric: {', '.join(bad)}."
    if df[NUMERIC_COLS].isna().any().any():
        return "Some numeric values are empty. Fill or drop those rows and upload again."
    return None


def score(df, model):
    probs = model.predict_proba(encode(df, model.feature_names_in_))[:, 1]
    out = df.copy()
    out["fraud_probability"] = np.round(probs, 2)
    out["prediction_label"] = np.where(probs > 0.5, "Fraudulent", "Legitimate")
    return out


model = load_model()
features = list(model.feature_names_in_)

st.title("🛡️ Fraud Risk Prediction")
st.markdown(
    "Scores card or account transactions for fraud risk with a **Decision Tree classifier** "
    "trained on labelled transactions. Try a single transaction below, or score a whole batch "
    "using the built-in sample or your own CSV."
)

with st.expander("How it works"):
    st.markdown(
        f"""
- **Algorithm:** scikit-learn `DecisionTreeClassifier` (max depth {model.get_params()['max_depth']}).
- **Inputs:** {len(NUMERIC_COLS)} numeric fields plus `time_of_day` and `channel`, one-hot encoded into
  {len(features)} model features.
- **Output:** probability that the transaction is fraudulent; above 0.5 is flagged.
- **What the tree learned:** it only needed two signals to separate the training data, a location
  mismatch and a low device trust score. The rules are below.
"""
    )
    st.code(export_text(model, feature_names=features), language="text")

imp = pd.Series(model.feature_importances_, index=features)
imp = imp[imp > 0].sort_values()

tab_single, tab_batch = st.tabs(["Single transaction", "Batch scoring"])

with tab_single:
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Transaction amount", min_value=0.0, value=250.0, step=10.0)
        age_days = st.number_input("Account age (days)", min_value=0, value=400, step=30)
    with c2:
        trust = st.slider("Device trust score", 0.0, 1.0, 0.65, 0.01,
                          help="0 = unknown or risky device, 1 = fully trusted device")
        prev_tx = st.number_input("Previous transactions", min_value=0, value=25)
    with c3:
        loc = st.radio("Location matches usual location?", ["Yes", "No"], index=1, horizontal=True)
        tod = st.selectbox("Time of day", TIME_LEVELS, index=3)
        channel = st.selectbox("Channel", CHANNEL_LEVELS, index=2)

    single = pd.DataFrame([{
        "transaction_amount": amount, "account_age_days": age_days,
        "device_trust_score": trust, "num_prev_transactions": prev_tx,
        "location_match": 1 if loc == "Yes" else 0, "time_of_day": tod, "channel": channel,
    }])
    res = score(single, model).iloc[0]
    flagged = res["prediction_label"] == "Fraudulent"

    m1, m2 = st.columns([1, 2])
    m1.metric("Fraud probability", f"{res['fraud_probability']:.0%}")
    with m2:
        if flagged:
            st.error("**Flagged as likely fraud.** Hold the transaction for review.")
        else:
            st.success("**Looks legitimate.** No action needed.")
        reasons = []
        if single["location_match"].iloc[0] == 0:
            reasons.append("location does not match the customer's usual location")
        else:
            reasons.append("location matches the usual location")
        reasons.append(f"device trust score is {trust:.2f} (the tree's cut-off is 0.80)")
        st.caption("Why: " + "; ".join(reasons) + ".")

    fig_imp = px.bar(imp, orientation="h", labels={"value": "Importance", "index": ""},
                     title="Features the model actually uses")
    fig_imp.update_layout(showlegend=False, height=260, margin=dict(t=40, b=10))
    st.plotly_chart(fig_imp, width="stretch")

with tab_batch:
    st.markdown(
        "Upload a CSV with columns: " + ", ".join(f"`{c}`" for c in NUMERIC_COLS + CATEGORICAL_COLS)
        + ". No file? The app scores 200 synthetic sample transactions."
    )
    sample = sample_transactions()
    st.download_button("Download CSV template (sample data)", sample.to_csv(index=False),
                       file_name="fraud_sample.csv", mime="text/csv")
    uploaded_file = st.file_uploader("Upload transactions (.csv)", type=["csv"])

    df, source = sample, "built-in sample"
    if uploaded_file is not None:
        try:
            df, source = pd.read_csv(uploaded_file), uploaded_file.name
        except Exception as exc:
            st.error(f"Could not read that file as CSV: {exc}")
            st.stop()

    error = validate(df)
    if error:
        st.error(error)
        st.stop()

    unknown = {c: sorted(set(df[c].astype(str)) - set(levels))
               for c, levels in [("time_of_day", TIME_LEVELS), ("channel", CHANNEL_LEVELS)]}
    for col, vals in unknown.items():
        if vals:
            st.warning(f"Unrecognised `{col}` values {vals} were treated as the baseline level.")

    results = score(df, model)
    total = len(results)
    fraud_count = int((results["prediction_label"] == "Fraudulent").sum())

    st.caption(f"Scoring: {source}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Transactions", f"{total:,}")
    k2.metric("Flagged as fraud", f"{fraud_count:,}")
    k3.metric("Fraud rate", f"{fraud_count / total:.1%}")
    k4.metric("Amount at risk",
              f"{results.loc[results['prediction_label'] == 'Fraudulent', 'transaction_amount'].sum():,.0f}")

    g1, g2 = st.columns(2)
    with g1:
        fig_pie = px.pie(results, names="prediction_label", title="Fraud vs legitimate", hole=0.45,
                         color="prediction_label",
                         color_discrete_map={"Legitimate": "#4C8BF5", "Fraudulent": "#E5484D"})
        st.plotly_chart(fig_pie, width="stretch")
    with g2:
        rate = (results.groupby("channel")["prediction_label"]
                .apply(lambda s: (s == "Fraudulent").mean()).reset_index(name="fraud_rate"))
        fig_ch = px.bar(rate, x="channel", y="fraud_rate", title="Flag rate by channel",
                        labels={"fraud_rate": "Share flagged", "channel": "Channel"})
        fig_ch.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ch, width="stretch")

    st.subheader("Results (highest risk first)")
    st.dataframe(results.sort_values("fraud_probability", ascending=False), width="stretch",
                 hide_index=True)
    st.download_button("Download prediction results", results.to_csv(index=False),
                       file_name="fraud_predictions.csv", mime="text/csv")
