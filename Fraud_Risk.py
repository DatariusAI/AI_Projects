import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.joblib")

model = load_model()

st.title("Fraud Risk Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload transaction data CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(data.head())

    # Step 1: Preprocess
    st.subheader("🔄 Preprocessing Data")
    categorical_cols = ['time_of_day', 'channel']
    st.write("Categorical columns:", categorical_cols)

    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Align features to model input
    try:
        X_columns = model.feature_names_in_
    except AttributeError:
        st.error("⚠️ Model missing feature names. Ensure it was trained using scikit-learn ≥ 1.0.")
        st.stop()

    for col in X_columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    X_new = data_encoded[X_columns]

    # Step 2: Predict
    predictions = model.predict(X_new)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_new)[:, 1]
    else:
        probabilities = np.zeros(len(predictions))

    # Step 3: Explain Predictions
    def explain(pred, prob):
        if pred == 1:
            return "Potential Fraud - Review required"
        else:
            return "Legitimate Transaction - No action needed"

    results_df = data.copy()
    results_df["Fraud_Prediction"] = predictions
    results_df["Fraud_Probability"] = np.round(probabilities, 3)
    results_df["Prediction_Explanation"] = results_df.apply(lambda row: explain(row["Fraud_Prediction"], row["Fraud_Probability"]), axis=1)

    # Show results
    st.subheader("✅ Fraud Prediction Results")
    st.dataframe(results_df)

    # Download button
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Prediction Results", data=csv, file_name="fraud_predictions.csv", mime="text/csv")
