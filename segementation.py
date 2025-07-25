# streamlit_segmentation_app.py

import streamlit as st
import pandas as pd
import joblib

# Load models
scaler = joblib.load("segmentation_scaler.joblib")
model = joblib.load("kmeans_model.joblib")

# App title
st.title("🏷️ Customer Segmentation App")
st.write("Upload a CSV file to predict customer segments.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Step 1: Load and display input data
    st.subheader("📄 Uploaded Data Preview")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # Step 2: Preprocess data
    st.subheader("🔄 Processing Data")
    categorical_cols = ['channel_preference', 'region', 'gender']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Step 3: Align columns to model input
    try:
        # Ensure input columns match what the model expects
        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Add missing cols as 0

        df_encoded = df_encoded[expected_cols]  # Align column order

        # Step 4: Apply scaler and predict
        X_scaled = scaler.transform(df_encoded)
        predictions = model.predict(X_scaled)

        # Step 5: Output results
        st.subheader("📊 Segmentation Results")
        df['Segment'] = predictions
        st.dataframe(df)

        # Download segmented data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

    except AttributeError:
        st.error("❌ Could not determine feature names. Ensure model was trained with scikit-learn ≥ 1.0 and scaler was saved using `.fit()`.")
