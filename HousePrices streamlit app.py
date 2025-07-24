import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- Custom CSS ---
st.markdown("""
    <style>
        .main {
            background-color: #111827;
            color: #f3f4f6;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #ef4444;
            color: white;
            padding: 0.5em 1em;
            border: none;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #dc2626;
        }
        .stDataFrame {
            background-color: #1f2937;
        }
        .stTextInput>div>div>input {
            color: white;
        }
        h1, h2, h3, h4 {
            color: #facc15;
        }
        .stMarkdown p {
            color: #d1d5db;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load data ---
@st.cache_data
def load_data():
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

model = RandomForestRegressor(random_state=42, n_estimators=50)
model.fit(X, y)

# --- Sidebar: User Input ---
st.sidebar.header('🧮 Input Property Features')

def user_input_features():
    region = st.sidebar.selectbox('Regionname_Southern Metropolitan', [0, 1])
    rooms = st.sidebar.slider('Rooms', 1, 10, 3)
    distance = st.sidebar.number_input('Distance (km to CBD)', 0.0, 50.0, 5.0)
    type_u = st.sidebar.selectbox('Unit/House Indicator (1 = Unit, 0 = House)', [0, 1])
    landsize = st.sidebar.number_input('Land Size (m²)', 0, 2000, 500)

    data = {
        'Regionname_Southern Metropolitan': region,
        'Rooms': rooms,
        'Distance': distance,
        'Type_u': type_u,
        'Landsize': landsize
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# --- Main Layout ---
st.title('🏠 Melbourne Housing Price Prediction')
st.markdown("Welcome! This app uses a machine learning model to estimate **house prices in Melbourne** based on selected features. Adjust the inputs in the sidebar and click predict.")

with st.container():
    st.subheader("🔎 Your Input Summary")
    st.dataframe(input_df.style.format())

# --- Prediction Section ---
if st.button("💰 Predict Price"):
    input_df = input_df[X.columns]
    prediction = model.predict(input_df)[0]

    st.markdown("## 🎯 Predicted House Price (AUD)")
    st.success(f"**${prediction:,.0f}**")

st.markdown("---")
st.caption("Model: Random Forest | UI enhanced by [Your Name]")
