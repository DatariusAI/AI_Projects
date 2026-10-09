import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

NUMERIC_COLS = ["age", "income", "employment_years", "credit_score", "loan_amount",
                "loan_term_months", "existing_loans_count"]
CATEGORICAL_COLS = ["marital_status", "education_level", "loan_purpose"]
# Levels seen in training. The alphabetically first level was dropped by one-hot encoding.
LEVELS = {
    "marital_status": ["Divorced", "Married", "Single"],
    "education_level": ["Bachelor", "High School", "Master", "PhD"],
    "loan_purpose": ["Business", "Car", "Home", "Personal"],
}
PRETTY = {
    "age": "Age", "income": "Annual income", "employment_years": "Years employed",
    "credit_score": "Credit score", "loan_amount": "Loan amount", "loan_term_months": "Loan term",
    "existing_loans_count": "Existing loans",
}


@st.cache_resource
def load_model():
    return joblib.load("loan_model.joblib")


@st.cache_data
def sample_applications(n=150, seed=11):
    rng = np.random.default_rng(seed)
    income = rng.normal(60000, 18000, n).clip(15000, 150000).round(-2)
    return pd.DataFrame({
        "age": rng.integers(21, 65, n),
        "income": income,
        "employment_years": rng.integers(0, 30, n),
        "credit_score": rng.normal(680, 60, n).clip(450, 850).round(),
        "loan_amount": (income * rng.uniform(0.15, 0.55, n)).round(-2),
        "loan_term_months": rng.choice([12, 24, 36, 48, 60], n),
        "existing_loans_count": rng.integers(0, 4, n),
        "marital_status": rng.choice(LEVELS["marital_status"], n),
        "education_level": rng.choice(LEVELS["education_level"], n),
        "loan_purpose": rng.choice(LEVELS["loan_purpose"], n),
    })


def encode(df, expected):
    enc = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])
    return enc.reindex(columns=expected, fill_value=0).astype(float)


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


def label(prob):
    if prob >= 0.95:
        return "Approve (very confident)"
    if prob >= 0.5:
        return "Approve (moderate confidence)"
    if prob <= 0.05:
        return "Reject (very confident)"
    return "Reject (moderate confidence)"


model = load_model()
expected_columns = list(model.feature_names_in_)
coefs = pd.Series(model.coef_[0], index=expected_columns)

st.title("🏦 Loan Acceptance Predictor")
st.markdown(
    "Estimates the probability that a loan application is **approved**, using a "
    "**Logistic Regression** model trained on historical applications. Change the applicant "
    "below to see the decision update, or score a whole file in the batch tab."
)

with st.expander("How it works"):
    st.markdown(
        f"""
- **Algorithm:** scikit-learn `LogisticRegression` (L2 penalty, C={model.C}).
- **Inputs:** {len(NUMERIC_COLS)} numeric fields plus marital status, education and loan purpose,
  one-hot encoded into {len(expected_columns)} features.
- **Output:** probability of approval. 0.5 or more means approve.
- **Explanation:** a logistic model adds up `coefficient x value` for each feature. The
  "what moved the score" chart shows each feature's contribution compared with a reference
  applicant (the app's default inputs), in log-odds.
- The model was trained on raw, unscaled values, so income and loan amount carry most of the weight.
"""
    )

tab_single, tab_batch = st.tabs(["Single applicant", "Batch scoring"])

REFERENCE = {"age": 35, "income": 55000, "employment_years": 6, "credit_score": 680,
             "loan_amount": 20000, "loan_term_months": 36, "existing_loans_count": 1,
             "marital_status": "Married", "education_level": "Bachelor", "loan_purpose": "Home"}

with tab_single:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 75, REFERENCE["age"])
        income = st.number_input("Annual income", 1000, 1000000, REFERENCE["income"], step=1000)
        employment_years = st.slider("Years employed", 0, 45, REFERENCE["employment_years"])
    with c2:
        credit_score = st.slider("Credit score", 300, 850, REFERENCE["credit_score"])
        loan_amount = st.number_input("Loan amount", 500, 500000, REFERENCE["loan_amount"], step=500)
        loan_term = st.select_slider("Loan term (months)", [12, 24, 36, 48, 60, 72, 84],
                                     REFERENCE["loan_term_months"])
    with c3:
        existing = st.slider("Existing loans", 0, 10, REFERENCE["existing_loans_count"])
        marital = st.selectbox("Marital status", LEVELS["marital_status"], index=1)
        education = st.selectbox("Education", LEVELS["education_level"], index=0)
        purpose = st.selectbox("Loan purpose", LEVELS["loan_purpose"], index=2)

    applicant = pd.DataFrame([{
        "age": age, "income": income, "employment_years": employment_years,
        "credit_score": credit_score, "loan_amount": loan_amount, "loan_term_months": loan_term,
        "existing_loans_count": existing, "marital_status": marital,
        "education_level": education, "loan_purpose": purpose,
    }])
    x = encode(applicant, expected_columns)
    prob = float(model.predict_proba(x)[0, 1])

    r1, r2 = st.columns([1, 2])
    with r1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": "Approval probability"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#2E7D32" if prob >= 0.5 else "#C62828"},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 50}},
        ))
        gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(gauge, width="stretch")
        verdict = label(prob)
        if prob >= 0.5:
            st.success(f"**{verdict}**")
        else:
            st.error(f"**{verdict}**")
        st.caption(f"Loan-to-income ratio: {loan_amount / income:.0%}")

    with r2:
        ref_x = encode(pd.DataFrame([REFERENCE]), expected_columns)
        contrib = (coefs * (x.iloc[0] - ref_x.iloc[0]))
        contrib = contrib[contrib.abs() > 1e-6]
        if contrib.empty:
            st.info("This is the reference applicant. Change any input to see what moves the score.")
        else:
            names = [PRETTY.get(i, i.replace("_", " ").capitalize()) for i in contrib.index]
            cdf = pd.DataFrame({"feature": names, "contribution": contrib.values}).sort_values("contribution")
            fig_c = px.bar(cdf, x="contribution", y="feature", orientation="h",
                           color=cdf["contribution"] > 0,
                           color_discrete_map={True: "#2E7D32", False: "#C62828"},
                           title="What moved the score vs. the reference applicant (log-odds)")
            fig_c.update_layout(showlegend=False, height=300, margin=dict(t=40, b=10),
                                yaxis_title="", xaxis_title="Contribution (+ helps approval)")
            st.plotly_chart(fig_c, width="stretch")

        amounts = np.linspace(max(500, loan_amount * 0.2), loan_amount * 2, 60)
        sweep = pd.concat([applicant] * len(amounts), ignore_index=True)
        sweep["loan_amount"] = amounts
        curve = model.predict_proba(encode(sweep, expected_columns))[:, 1]
        fig_s = px.line(x=amounts, y=curve, labels={"x": "Loan amount", "y": "Approval probability"},
                        title="What if the applicant asked for a different amount?")
        fig_s.add_vline(x=loan_amount, line_dash="dot")
        fig_s.update_yaxes(range=[0, 1], tickformat=".0%")
        fig_s.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_s, width="stretch")

with tab_batch:
    st.markdown(
        "Upload a CSV with columns: " + ", ".join(f"`{c}`" for c in NUMERIC_COLS + CATEGORICAL_COLS)
        + ". No file? The app scores 150 synthetic sample applications."
    )
    sample = sample_applications()
    st.download_button("Download CSV template (sample data)", sample.to_csv(index=False).encode("utf-8"),
                       file_name="loan_sample.csv", mime="text/csv")
    uploaded_file = st.file_uploader("Upload applications (.csv)", type=["csv"])

    new_data, source = sample, "built-in sample"
    if uploaded_file is not None:
        try:
            new_data, source = pd.read_csv(uploaded_file), uploaded_file.name
        except Exception as exc:
            st.error(f"Could not read that file as CSV: {exc}")
            st.stop()

    error = validate(new_data)
    if error:
        st.error(error)
        st.stop()

    probs = model.predict_proba(encode(new_data, expected_columns))[:, 1]
    results_df = new_data.copy()
    results_df["approval_probability"] = probs.round(3)
    results_df["prediction"] = (probs >= 0.5).astype(int)
    results_df["decision"] = [label(p) for p in probs]

    st.caption(f"Scoring: {source}")
    k1, k2, k3 = st.columns(3)
    k1.metric("Applications", f"{len(results_df):,}")
    k2.metric("Predicted approvals", f"{results_df['prediction'].mean():.0%}")
    k3.metric("Borderline (30-70%)", f"{((probs > 0.3) & (probs < 0.7)).sum():,}")

    g1, g2 = st.columns(2)
    with g1:
        fig_h = px.histogram(results_df, x="approval_probability", nbins=20,
                             title="Distribution of approval probabilities")
        st.plotly_chart(fig_h, width="stretch")
    with g2:
        fig_sc = px.scatter(results_df, x="income", y="loan_amount", color="approval_probability",
                            color_continuous_scale="RdYlGn", title="Income vs loan amount")
        st.plotly_chart(fig_sc, width="stretch")

    st.subheader("Prediction results")
    st.dataframe(results_df, width="stretch", hide_index=True)
    st.download_button("📥 Download predictions as CSV", results_df.to_csv(index=False).encode("utf-8"),
                       file_name="loan_predictions.csv", mime="text/csv")
