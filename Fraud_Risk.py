import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.joblib")

model = load_model()

st.title("Fraud Risk Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload transaction data (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Categorical columns
    categorical_cols = ['time_of_day', 'channel']
    st.subheader("Preprocessing Data")
    st.write("Categorical columns:")
    st.code(categorical_cols)

    # Encode and align features
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    try:
        model_features = model.feature_names_in_
    except AttributeError:
        st.error("Model missing feature names. Retrain using scikit-learn ≥ 1.0.")
        st.stop()

    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[model_features]

    # Prediction
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    def explain(pred, prob):
        if pred == 1:
            if prob > 0.9:
                return "High Risk - Likely Fraud"
            elif prob > 0.5:
                return "Moderate Risk - Investigate"
            else:
                return "Suspicious but Uncertain"
        else:
            return "Legitimate Transaction"

    df["Fraud_Prediction"] = preds
    df["Fraud_Probability"] = np.round(proba, 2)
    df["Prediction_Explanation"] = df.apply(lambda row: explain(row["Fraud_Prediction"], row["Fraud_Probability"]), axis=1)

    st.subheader("Fraud Prediction Results")
    st.dataframe(df)

    st.download_button("Download Prediction Results", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="fraud_predictions.csv", mime="text/csv")

    # Optional: Insights
    st.subheader("Segment-Level Insights")
    total = len(df)
    frauds = df[df["Fraud_Prediction"] == 1]
    legitimate = df[df["Fraud_Prediction"] == 0]

    st.markdown(f"""
    - Total Transactions: **{total}**
    - Predicted Fraud: **{len(frauds)} ({(len(frauds)/total)*100:.1f}%)**
    - Legitimate: **{len(legitimate)} ({(len(legitimate)/total)*100:.1f}%)**

    **Top 5 Highest Risk Transactions:**
    """)
    st.dataframe(df.sort_values(by="Fraud_Probability", ascending=False).head(5)[["transaction_id", "Fraud_Probability", "Prediction_Explanation"]])
