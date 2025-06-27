import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load your trained model and feature names ---
# Save your model first with: joblib.dump(model, 'rf_model.pkl')
model = joblib.load('rf_model.pkl')

# List your top N features used in the model
top_features = [
    'Regionname_Southern Metropolitan',
    'Rooms',
    'Distance',
    'Type_u',
    'Landsize'
    # ... add more as used in your model!
]

# --- Sidebar for user inputs ---
st.sidebar.header('Input Property Features')

def user_input_features():
    data = {}
    # Example input types, change as per your features and data!
    data['Regionname_Southern Metropolitan'] = st.sidebar.selectbox(
        'Regionname_Southern Metropolitan', [0, 1])
    data['Rooms'] = st.sidebar.slider('Rooms', 1, 10, 3)
    data['Distance'] = st.sidebar.number_input('Distance (km to CBD)', 0.0, 50.0, 5.0)
    data['Type_u'] = st.sidebar.selectbox('Unit/House Indicator (1=Unit, 0=House)', [0, 1])
    data['Landsize'] = st.sidebar.number_input('Land Size (m2)', 0, 2000, 500)
    # Add additional fields as needed, matching your model's features
    return pd.DataFrame([data])

input_df = user_input_features()

# --- Main page ---
st.title('Melbourne Housing Price Prediction')
st.markdown('This simple demo predicts house prices based on key features. Enter details in the sidebar and get an instant price estimate.')

st.subheader('Your Input')
st.write(input_df)

# --- Make prediction ---
if st.button('Predict Price'):
    prediction = model.predict(input_df)[0]
    st.subheader('Predicted House Price (AUD)')
    st.success(f"${prediction:,.0f}")

st.markdown("---")
st.caption("Model: Random Forest | Demo by [Your Name]")

# (Optional) Add extra info, visuals, or links as needed.
