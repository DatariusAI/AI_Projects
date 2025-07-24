import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Melbourne Housing Price Prediction")
st.markdown("Enter details below to get an estimated house price.")

# Load model
model, feature_cols = joblib.load("final_rf_model.pkl")

# Sidebar inputs
region = st.sidebar.selectbox("Regionname_Southern Metropolitan", [0, 1])
rooms = st.sidebar.slider("Rooms", 1, 10, 3)
distance = st.sidebar.number_input("Distance (km to CBD)", 0.0, 50.0, 5.0)
type_u = st.sidebar.selectbox("Unit/House Indicator (1=Unit, 0=House)", [0, 1])
landsize = st.sidebar.number_input("Land Size (m2)", 0, 2000, 500)

input_data = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
input_data["Regionname_Southern Metropolitan"] = region
input_data["Rooms"] = rooms
input_data["Distance"] = distance
input_data["Type_u"] = type_u
input_data["Landsize"] = landsize

st.subheader("Input Summary")
st.write(input_data)

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price (AUD): ${prediction:,.0f}")
