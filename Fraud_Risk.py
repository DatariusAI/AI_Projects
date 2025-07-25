import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("fraud_model.joblib")

# App title
st.title("Fraud Risk Prediction App")
st.markdown("Upload transaction data (.csv)")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Select categorical columns that exist in uploaded file
    expected_categoricals = ['time_of_day', 'channel']
    categorical_cols = [col for col in expected_categoricals if col in df.columns]

    st.subheader("Preprocessing Data")
    st.markdown("**Categorical columns:**")
    st.json(categorical_cols)

    # Encode categoricals
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Make prediction
    probs = model.predict_proba(df_encoded)[:, 1]  # Probability of class 1 (fraud)
    preds = (probs >= 0.5).astype(int)

    df_results = df.copy()
    df_results["Fraud_Probability"] = probs.round(2)
    df_results["Prediction"] = preds
    df_results["Prediction_Label"] = df_results["Prediction"].map({1: "Fraudulent", 0: "Legitimate"})

    st.subheader("Fraud Prediction Results")
    st.dataframe(df_results)

    # Download button
    st.download_button("Download Prediction Results", df_results.to_csv(index=False), "fraud_predictions.csv")

    # Insights
    st.subheader("Prediction Insights")
    total = len(df_results)
    fraud = df_results["Prediction"].sum()
    legit = total - fraud
    avg_prob = df_results["Fraud_Probability"].mean().round(2)

    st.markdown(f"""
    - Total Transactions: **{total}**
    - Predicted Fraudulent: **{fraud}**
    - Predicted Legitimate: **{legit}**
    - Average Fraud Probability: **{avg_prob}**
    """)

    # Visualizations
    st.subheader("Visual Insights")

    # Fraud distribution barplot
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_results, x="Prediction_Label", palette=["green", "red"], ax=ax1)
    ax1.set_title("Transaction Classification")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Prediction")
    st.pyplot(fig1)

    # Scatter plot: amount vs fraud
    if "transaction_amount" in df_results.columns:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_results, x="transaction_amount", y="Fraud_Probability",
                        hue="Prediction_Label", palette=["green", "red"], ax=ax2)
        ax2.set_title("Transaction Amount vs Fraud Probability")
        ax2.set_xlabel("Transaction Amount")
        ax2.set_ylabel("Fraud Probability")
        st.pyplot(fig2)
