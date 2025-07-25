import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    model = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("segmentation_scaler.joblib")
    return model, scaler

# Define segment explanations
SEGMENT_EXPLANATIONS = {
    0: "Digitally Comfortable Veterans: High digital engagement, prefers online and mobile channels.",
    1: "Wealthy Traditionalists: Traditional users, prefer branch interaction, possibly older demographic.",
    2: "Product-Rich Hybrids: Medium-income, high product ownership, mixed channel usage.",
    3: "Low-Value Starters: New or low-value customers, fewer products, lower engagement."
}

# App layout
st.title("Customer Segmentation App")

uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")
if uploaded_file:
    st.subheader("Preview of Uploaded Data")
    new_data = pd.read_csv(uploaded_file)
    st.write(new_data.head())

    # Step 1: Load Model and Scaler
    model, scaler = load_model_scaler()

    # Step 2: One-hot encode categorical features
    categorical_cols = ['gender', 'occupation', 'channel_preference', 'time_of_day']
    new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

    # Step 3: Align with training columns
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[expected_columns]

    # Step 4: Scale the data
    scaled_data = scaler.transform(new_data_encoded)

    # Step 5: Predict segments
    predictions = model.predict(scaled_data)

    # Show predictions with explanations
    st.subheader("Predicted Segments")
    new_data["Predicted Segment"] = predictions
    new_data["Segment Description"] = new_data["Predicted Segment"].map(SEGMENT_EXPLANATIONS)
    st.write(new_data[["Predicted Segment", "Segment Description"]].head(10))

    # Optional download
    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Prediction", csv, "segmented_customers.csv", "text/csv")
