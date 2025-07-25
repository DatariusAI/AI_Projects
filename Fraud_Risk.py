import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and expected features
model = joblib.load("fraud_model.joblib")
expected_features = model.feature_names_in_

st.title("Fraud Risk Prediction App")
st.subheader("Upload transaction data (.csv)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # --- Internal Preprocessing (Hidden from UI) ---
    categorical_cols = ["time_of_day", "channel"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[expected_features]

    # --- Model Predictions ---
    probs = model.predict_proba(df_encoded)[:, 1]
    preds = (probs > 0.5).astype(int)

    df["fraud_probability"] = np.round(probs, 2)
    df["prediction"] = preds
    df["prediction_label"] = df["prediction"].map({1: "Fraudulent", 0: "Legitimate"})

    st.subheader("Fraud Prediction Results")
    st.dataframe(df)

    # --- Insights ---
    st.subheader("Prediction Insights")
    total = len(df)
    fraud_count = (df["prediction"] == 1).sum()
    legit_count = total - fraud_count
    avg_prob = df["fraud_probability"].mean()

    st.markdown(f"**Total Transactions:** {total}")
    st.markdown(f"**Predicted Fraudulent:** {fraud_count}")
    st.markdown(f"**Predicted Legitimate:** {legit_count}")
    st.markdown(f"**Average Fraud Probability:** {avg_prob:.2f}")

    # --- Visual Summary ---
    st.subheader("Visual Summary")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].pie([legit_count, fraud_count], labels=['Legitimate', 'Fraudulent'],
              autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    ax[0].set_title("Fraud vs Legitimate")

    ax[1].hist(df["fraud_probability"], bins=10, color='skyblue', edgecolor='black')
    ax[1].set_title("Fraud Probability Distribution")
    ax[1].set_xlabel("Fraud Probability")
    ax[1].set_ylabel("Count")

    st.pyplot(fig)

    # --- Download Option ---
    st.download_button("Download Prediction Results",
                       df.to_csv(index=False),
                       file_name="fraud_predictions.csv",
                       mime="text/csv")
