import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_price(seat_class):
    base_price = 500000  # Base price in USD
    multipliers = {'Economy': 1, 'Luxury': 2, 'VIP Zero-Gravity': 3.5}
    return base_price * multipliers[seat_class]

# Title
st.title("🚀 Dubai to the Stars – Space Travel Booking Platform")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Book a Trip", "My Dashboard", "About"])

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
    
    st.subheader("🤖 AI Travel Tip")
    travel_tips = [
        "Pack light, but don't forget your space socks!",
        "Hydration is key in microgravity. Drink plenty of water!",
        "Gaze at Earth from orbit - it's a life-changing view!",
        "Floating in zero-G is fun, but secure your belongings!"
    ]
    st.info(random.choice(travel_tips))
    
elif menu == "About":
    st.header("🌌 About the Platform")
    st.write("This futuristic booking platform allows users to schedule and manage their space journeys from Dubai.")
    st.write("Designed for the Dubai Space Travel Hackathon 🚀")
    st.write("Developed with **Streamlit** for an intuitive and engaging user experience.")
