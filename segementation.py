import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances_argmin_min

# --- Load model and scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("segmentation_scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

# --- App Title ---
st.title("Customer Segmentation App")
st.write("Upload a CSV file with new customer data to predict their segments using KMeans.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
if uploaded_file is not None:
    # Load uploaded data
    new_data = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(new_data.head())

    # Identify and one-hot encode categorical features
    st.subheader("🔢 Processing Data")
    categorical_cols = new_data.select_dtypes(include='object').columns.tolist()
    st.write("Categorical columns:", categorical_cols)
    new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

    # Align columns with training features
    st.write("📏 Aligning features to match training structure...")
    try:
        X_columns = model.feature_names_in_  # requires sklearn >=1.0
    except:
        st.error("❌ Could not determine feature names. Ensure model was trained with scikit-learn >= 1.0")
        st.stop()

    for col in X_columns:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[X_columns]

    # Scale new data
    X_scaled = scaler.transform(new_data_encoded)

    # Predict segments
    predicted_segments = model.predict(X_scaled)

    # Compute distance to cluster centers (for confidence estimate)
    closest, distances = pairwise_distances_argmin_min(X_scaled, model.cluster_centers_)
    max_distance = distances.max()
    probabilities = 1 - (distances / max_distance)

    # Combine results
    results = new_data.copy()
    results['Predicted_Segment'] = predicted_segments
    results['Segment_Probability'] = probabilities.round(4)

    # Segment descriptions (based on 4-cluster setup)
    segment_descriptions = {
        0: "Segment 0: Digitally Comfortable Veterans – High digital engagement",
        1: "Segment 1: Wealthy Traditionalists – Branch-preferred users, possibly older",
        2: "Segment 2: Product-Rich Hybrids – Mixed channels, high product ownership",
        3: "Segment 3: Low-Value Starters – New, few products, low engagement"
    }

    results['Explanation'] = results['Predicted_Segment'].apply(
        lambda seg: segment_descriptions.get(seg, "⚠️ Unknown segment (check model training)")
    )

    # Show output
    st.subheader("✅ Segment Predictions")
    st.dataframe(results[["Predicted_Segment", "Segment_Probability", "Explanation"]].head())

    # Option to download results
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=results.to_csv(index=False).encode('utf-8'),
        file_name='segment_predictions.csv',
        mime='text/csv'
    )
