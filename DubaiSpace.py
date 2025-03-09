import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
import requests
from transformers import pipeline

def generate_price(seat_class):
    base_price = 500000  # Base price in USD
    multipliers = {'Economy': 1, 'Luxury': 2, 'VIP Zero-Gravity': 3.5}
    return base_price * multipliers[seat_class]

def get_ai_travel_tip():
    try:
        generator = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha")
        response = generator("Provide a futuristic space travel tip.", max_length=50)
        return response[0]["generated_text"].strip()
    except Exception as e:
        return "AI is unavailable. Try again later."

# Title
st.title("🚀 Dubai to the Stars – Space Travel Booking Platform")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Book a Trip", "My Dashboard", "About", "AI Travel Assistant"])

if menu == "Book a Trip":
    st.header("🛸 Schedule Your Space Trip")
    
    # User input
    destinations = ["International Space Station", "Lunar Hotel", "Mars Colony", "Orbital Space Yacht"]
    departure_date = st.date_input("Select Departure Date", min_value=datetime.today(), max_value=datetime.today() + timedelta(days=365))
    destination = st.selectbox("Choose Your Destination", destinations)
    seat_class = st.radio("Select Seat Class", ["Economy", "Luxury", "VIP Zero-Gravity"])
    
    # Pricing
    price = generate_price(seat_class)
    st.write(f"💰 Estimated Price: **${price:,.2f}**")
    
    if st.button("Book Now! 🚀"):
        st.success(f"Your trip to {destination} on {departure_date.strftime('%Y-%m-%d')} is booked!")
        st.balloons()
    
elif menu == "My Dashboard":
    st.header("🧑‍🚀 Your Space Travel Dashboard")
    
    # Dummy booking data
    bookings = pd.DataFrame({
        "Destination": ["Lunar Hotel", "Orbital Space Yacht"],
        "Departure Date": [(datetime.today() + timedelta(days=random.randint(10, 90))).strftime('%Y-%m-%d') for _ in range(2)],
        "Seat Class": ["Luxury", "VIP Zero-Gravity"],
        "Status": ["Confirmed", "Upcoming"]
    })
    
    st.table(bookings)
    
    st.subheader("🚀 Countdown to Next Trip")
    next_trip_date = datetime.strptime(bookings.iloc[0]['Departure Date'], '%Y-%m-%d')
    countdown = next_trip_date - datetime.today()
    st.write(f"**{countdown.days} days** until your next space adventure!")
    
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
