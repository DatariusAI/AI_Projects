import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("fraud_model.joblib")

# Title
st.title("Fraud Risk Prediction App")
st.caption("Upload transaction data (.csv)")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Categorical columns to encode
    categorical_cols = ['time_of_day', 'channel']
    st.subheader("Preprocessing Data")
    st.write("Categorical columns:")
    st.json(categorical_cols)

    # Check presence of categorical columns
    for col in categorical_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align encoded columns with model input
    expected_features = model.feature_names_in_
    missing_cols = [col for col in expected_features if col not in df_encoded.columns]
    for col in missing_cols:
        df_encoded[col] = 0  # Add missing columns with 0
    df_encoded = df_encoded[expected_features]  # Ensure correct order

    # Predictions
    probs = model.predict_proba(df_encoded)[:, 1]  # Prob. of class 1 (fraud)
    preds = (probs >= 0.5).astype(int)

    # Results
    df_results = df.copy()
    df_results["fraud_probability"] = probs.round(2)
    df_results["prediction"] = preds
    label_map = {0: "Legitimate", 1: "Fraudulent"}
    df_results["prediction_label"] = df_results["prediction"].map(label_map)

    st.subheader("Fraud Prediction Results")
    st.dataframe(df_results.head(10))

    # Summary insights
    total = len(df_results)
    fraud_count = df_results["prediction"].sum()
    legit_count = total - fraud_count
    avg_prob = df_results["fraud_probability"].mean()

    st.subheader("Prediction Insights")
    st.write(f"**Total Transactions:** {total}")
    st.write(f"**Predicted Fraudulent:** {fraud_count}")
    st.write(f"**Predicted Legitimate:** {legit_count}")
    st.write(f"**Average Fraud Probability:** {avg_prob:.2f}")

    # Visualization
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df_results, x="prediction_label", palette="Set2", ax=ax1)
    ax1.set_title("Prediction Counts")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.histplot(df_results["fraud_probability"], bins=10, kde=True, ax=ax2)
    ax2.set_title("Fraud Probability Distribution")
    st.pyplot(fig2)

    # Download link
    csv_output = df_results.to_csv(index=False).encode("utf-8")
    st.download_button("Download Prediction Results", csv_output, file_name="fraud_predictions.csv")
