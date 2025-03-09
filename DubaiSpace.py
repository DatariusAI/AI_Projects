import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from transformers import pipeline

# ---- App Configuration ---- #
st.set_page_config(
    page_title="Dubai Space Travel",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Session State Initialization ---- #
if 'bookings' not in st.session_state:
    st.session_state.bookings = []
    
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        "name": "Guest Explorer",
        "loyalty_points": 3500,
        "membership_level": "Silver Voyager"
    }

# ---- Helper Functions ---- #
def generate_price(seat_class):
    """Generate price based on seat class."""
    base_price = 500000  # Base price in USD
    multipliers = {'Economy': 1, 'Luxury': 2, 'VIP Zero-Gravity': 3.5}
    return base_price * multipliers.get(seat_class, 1)

# ---- AI Chatbot Using Free Hugging Face Model ---- #
def get_ai_travel_tip():
    """Generate an AI travel tip using a free model."""
    try:
        generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=-1)  # Use CPU
        response = generator("Provide a futuristic space travel tip.", max_length=50)
        return response[0]["generated_text"].strip()
    except Exception as e:
        return f"AI Assistant error: {str(e)}"

# ---- Streamlit UI ---- #
st.title("🚀 Dubai to the Stars – AI-Powered Space Travel Booking")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Book a Trip", "My Dashboard", "AI Travel Assistant", "About"])

if menu == "Book a Trip":
    st.header("🛸 Schedule Your Space Trip")
    destinations = ["International Space Station", "Lunar Hotel", "Mars Colony", "Orbital Space Yacht"]
    departure_date = st.date_input("Select Departure Date", min_value=datetime.today(), max_value=datetime.today() + timedelta(days=365))
    destination = st.selectbox("Choose Your Destination", destinations)
    seat_class = st.radio("Select Seat Class", ["Economy", "Luxury", "VIP Zero-Gravity"])
    price = generate_price(seat_class)
    st.write(f"💰 Estimated Price: **${price:,.2f}**")
    if st.button("Book Now! 🚀"):
        booking_id = f"BK-{random.randint(100000, 999999)}"
        st.session_state.bookings.append({
            "id": booking_id,
            "destination": destination,
            "departure_date": departure_date.strftime('%Y-%m-%d'),
            "seat_class": seat_class,
            "status": "Confirmed"
        })
        st.success(f"Your trip to {destination} on {departure_date.strftime('%Y-%m-%d')} is booked! Booking Reference: {booking_id}")
        st.balloons()

elif menu == "My Dashboard":
    st.header("🧑‍🚀 Your Space Travel Dashboard")
    if st.session_state.bookings:
        bookings_df = pd.DataFrame(st.session_state.bookings)
        st.table(bookings_df)
        next_trip_date = datetime.strptime(bookings_df.iloc[0]['departure_date'], '%Y-%m-%d')
        countdown = (next_trip_date - datetime.today()).days
        st.write(f"**{countdown} days** until your next space adventure! 🚀")
    else:
        st.info("You don't have any bookings yet. Book your first space journey now!")

elif menu == "AI Travel Assistant":
    st.header("🤖 AI Space Travel Assistant")
    if st.button("Get AI Travel Tip 🛰️"):
        ai_tip = get_ai_travel_tip()
        st.info(ai_tip)

elif menu == "About":
    st.header("🌌 About the Platform")
    st.write("This futuristic booking platform allows users to schedule and manage their space journeys from Dubai.")
    st.write("Designed for the Dubai Space Travel Hackathon 🚀")
    st.write("Developed with **Streamlit** for an intuitive and engaging user experience.")
