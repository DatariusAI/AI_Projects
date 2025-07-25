import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.title("📊 Loan Acceptance Prediction App")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("loan_model.joblib")

model = load_model()

# Load training columns from the model input during training
expected_columns = list(model.feature_names_in_)

st.markdown("""
This app uses a trained Logistic Regression model to predict loan approval.
- Upload a CSV file with new loan applications
- The app will predict approval (1) or rejection (0)
- It will also provide a probability of approval
""")

uploaded_file = st.file_uploader("📤 Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.write(new_data.head())

    # Preprocessing: One-hot encode
    new_data_encoded = pd.get_dummies(new_data, drop_first=True)

    # Add any missing columns
    for col in expected_columns:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0

    # Align column order
    new_data_encoded = new_data_encoded[expected_columns]

    # Predict
    pred_labels = model.predict(new_data_encoded)
    pred_probs = model.predict_proba(new_data_encoded)[:, 1]

    # Attach predictions
    results_df = new_data.copy()
    results_df["prediction"] = pred_labels
    results_df["approval_probability"] = pred_probs

    # Display predictions
    st.subheader("🔍 Prediction Results")
    st.write(results_df[["prediction", "approval_probability"]].head())

    # Detailed interpretation
    st.subheader("📘 Interpretation of First 5 Predictions")
    for i, row in results_df.head().iterrows():
        pred = row["prediction"]
        prob = row["approval_probability"]
        st.markdown(f"**Row {i}** - Prediction: `{pred}` | Probability: `{prob:.2f}`")
        if pred == 1 and prob > 0.95:
            st.info("The model is VERY confident this loan will be APPROVED.")
        elif pred == 1:
            st.success("The model predicts approval, but with moderate confidence.")
        elif pred == 0 and prob < 0.05:
            st.error("The model is VERY confident this loan will be REJECTED.")
        else:
            st.warning("The model predicts rejection, but with moderate confidence.")

    # Option to download results
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="loan_predictions.csv",
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to get predictions.")
