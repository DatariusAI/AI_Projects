import streamlit as st
import pandas as pd
import joblib

# Load trained scaler and clustering model
scaler = joblib.load("segmentation_scaler.joblib")
model = joblib.load("kmeans_model.joblib")

# Set page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation App")

st.markdown("Upload customer data to classify them into meaningful segments.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Segment details
segment_descriptions = {
    0: "Segment 0 – Digitally Comfortable Veterans: High digital engagement, prefers online and mobile channels.",
    1: "Segment 1 – Wealthy Traditionalists: Traditional users, prefer branch interaction, possibly older demographic.",
    2: "Segment 2 – Product-Rich Hybrids: Medium-income, high product ownership, mixed channel usage.",
    3: "Segment 3 – Low-Value Starters: New or low-value customers, fewer products, lower engagement."
}

segment_colors = {
    0: "#91C8E4",  # blue
    1: "#FFCF81",  # orange
    2: "#97DECE",  # green
    3: "#FEC7B4"   # red
}

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    st.subheader("Processing Data")
    categorical_cols = ['channel_preference', 'region', 'gender']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    try:
        # Align with training features
        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[expected_cols]

        # Scale and Predict
        X_scaled = scaler.transform(df_encoded)
        df['Segment'] = model.predict(X_scaled)

        st.subheader("Segmentation Results")
        st.dataframe(df)

        # Segment Explanation
        st.markdown("### Segment Insights")
        for seg_id, description in segment_descriptions.items():
            st.markdown(f"""
                <div style="
                    background-color: {segment_colors[seg_id]};
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border-radius: 0.5rem;
                    color: black;
                    font-weight: bold;
                ">
                    {description}
                </div>
            """, unsafe_allow_html=True)

        # Download segmented data
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

    except AttributeError:
        st.error("Model/scaler missing `.feature_names_in_`. Retrain with scikit-learn ≥ 1.0 to fix.")
