import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_model.joblib")

st.title("Fraud Risk Prediction App")
st.markdown("Upload transaction data (.csv)")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df)

    # Define expected categorical columns
    categorical_cols = ['time_of_day', 'channel']

    st.subheader("🛠 Preprocessing Data")
    st.markdown("**Categorical columns:**")
    st.write(categorical_cols)

    # Check if columns exist before encoding
    existing_categoricals = [col for col in categorical_cols if col in df.columns]
    missing = list(set(categorical_cols) - set(df.columns))

    if missing:
        st.warning(f"⚠️ Missing expected categorical columns: {missing}")

    # Apply one-hot encoding only to existing columns
    df_encoded = pd.get_dummies(df, columns=existing_categoricals, drop_first=True)

    # Align features to model input shape
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    # Make predictions
    st.subheader("✅ Fraud Prediction Results")
    df['Fraud_Prediction'] = model.predict(df_encoded)
    df['Fraud_Probability'] = model.predict_proba(df_encoded)[:, 1]

    # Explain predictions
    df['Prediction_Explanation'] = df['Fraud_Prediction'].apply(
        lambda x: 'Potential Fraud - Review required' if x == 1 else 'Legitimate Transaction - No action needed'
    )

    st.dataframe(df)

    # 💡 Insights Section
    st.subheader("🧠 Prediction Insights")
    total = len(df)
    fraud = df['Fraud_Prediction'].sum()
    legit = total - fraud
    avg_prob = df['Fraud_Probability'].mean()

    st.markdown(f"- Total Transactions: **{total}**")
    st.markdown(f"- 🚨 Predicted Fraudulent: **{fraud}**")
    st.markdown(f"- ✅ Predicted Legitimate: **{legit}**")
    st.markdown(f"- 📊 Average Fraud Probability: **{avg_prob:.2f}**")

    st.download_button("⬇️ Download Prediction Results", data=df.to_csv(index=False), file_name="fraud_predictions.csv", mime="text/csv")
