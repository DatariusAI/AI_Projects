import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and training feature names
model = joblib.load("kmeans_model.joblib")
scaler = joblib.load("segmentation_scaler.joblib")
feature_names = joblib.load("features_used.pkl")  # <-- FIXED HERE

# Set page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🧠 Customer Segmentation App")
st.markdown("Upload a CSV file with customer data to get cluster predictions.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 1: Read data
    new_data = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(new_data.head())

    # Step 2: Identify and encode categorical columns
    st.subheader("🔄 Processing Data")

    categorical_cols = ["channel_preference", "region", "gender"]
    st.markdown("**Categorical columns:**")
    st.code(categorical_cols)

    try:
        # One-hot encode (drop first to match training)
        new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

        # Step 3: Align columns to training structure
        for col in feature_names:
            if col not in new_data_encoded.columns:
                new_data_encoded[col] = 0

        # Reorder columns
        X_new = new_data_encoded[feature_names]

        # Step 4: Scale data
        X_scaled = scaler.transform(X_new)

        # Step 5: Predict clusters
        cluster_labels = model.predict(X_scaled)

        # Step 6: Output predictions
        st.subheader("🎯 Predicted Customer Segments")
        new_data["Segment"] = cluster_labels
        st.dataframe(new_data)

        # Download button
        csv_output = new_data.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Segmentation Results", data=csv_output, file_name="segmentation_output.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ An error occurred while processing: {e}")
