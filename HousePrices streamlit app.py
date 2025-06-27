import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- Load data ---
@st.cache_data
def load_data():
    # Minimal synthetic sample (replace with your real data or CSV load)
    data = pd.DataFrame({
        'Regionname_Southern Metropolitan': [1, 0, 1, 0, 1],
        'Rooms': [3, 2, 4, 3, 5],
        'Distance': [5.0, 10.0, 3.5, 12.0, 7.0],
        'Type_u': [0, 1, 0, 1, 0],
        'Landsize': [500, 350, 600, 400, 900],
        'Price': [1000000, 750000, 1200000, 850000, 1500000]
    })
    return data

df = load_data()
X = df.drop('Price', axis=1)
y = df['Price']

# --- Train model on the fly ---
model = RandomForestRegressor(random_state=42, n_estimators=50)
model.fit(X, y)

top_features = list(X.columns)

# --- Sidebar for user inputs ---
st.sidebar.header('Input Property Features')

def user_input_features():
    data = {}
    data['Regionname_Southern Metropolitan'] = st.sidebar.selectbox('Regionname_Southern Metropolitan', [0, 1])
    data['Rooms'] = st.sidebar.slider('Rooms', 1, 10, 3)
    data['Distance'] = st.sidebar.number_input('Distance (km to CBD)', 0.0, 50.0, 5.0)
    data['Type_u'] = st.sidebar.selectbox('Unit/House Indicator (1=Unit, 0=House)', [0, 1])
    data['Landsize'] = st.sidebar.number_input('Land Size (m2)', 0, 2000, 500)
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
