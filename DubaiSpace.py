import streamlit as st
import datetime
import random
from pathlib import Path

# App Configurations
st.set_page_config(page_title="Dubai Space Travel Booking", layout="wide")

# Title & Intro Section
st.title("🚀 Dubai Space Travel Booking Platform")
st.markdown("### Book Your Futuristic Space Trip Now!")
st.image("https://source.unsplash.com/1600x900/?space,stars,galaxy", use_column_width=True)

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigation", ["Home", "Book a Trip", "Dashboard", "AI Travel Assistant"])

# Dynamic Pricing Function
def get_dynamic_price(destination, seat_class):
    base_prices = {"ISS": 500000, "Lunar Hotel": 1200000, "Orbital Space Station": 800000}
    class_multiplier = {"Economy": 1, "Luxury": 2, "VIP Zero-Gravity": 3}
    demand_factor = random.uniform(1.1, 2.0)
    return base_prices[destination] * class_multiplier[seat_class] * demand_factor

if menu == "Home":
    st.markdown("### Welcome to the Future of Space Travel!")
    st.write("Book a trip, track upcoming journeys, and explore AI-powered space travel insights.")

elif menu == "Book a Trip":
    st.subheader("🌌 Space Trip Booking")
    col1, col2 = st.columns(2)
    with col1:
        departure_date = st.date_input("Select Departure Date", datetime.date.today())
        destination = st.selectbox("Select Destination", ["ISS", "Lunar Hotel", "Orbital Space Station"])
        seat_class = st.selectbox("Select Seat Class", ["Economy", "Luxury", "VIP Zero-Gravity"])
    
    with col2:
        st.image("https://source.unsplash.com/400x300/?rocket,spaceship")
        st.write(f"📍 Destination: {destination}")
        st.write(f"💺 Seat Class: {seat_class}")
        price = get_dynamic_price(destination, seat_class)
        st.write(f"💰 Estimated Price: ${price:,.2f}")
        if st.button("Confirm Booking"):
            st.success("✅ Your trip is booked! Check the dashboard for details.")

elif menu == "Dashboard":
    st.subheader("📊 User Dashboard")
    st.write("View your upcoming and past space travel bookings.")
    countdown = datetime.datetime(2025, 6, 1) - datetime.datetime.now()
    st.write(f"🚀 Next Launch Countdown: {countdown.days} days")
    
    st.markdown("### Your Upcoming Trips")
    trips = [
        {"date": "2025-06-01", "destination": "ISS", "seat_class": "VIP Zero-Gravity"},
        {"date": "2025-08-15", "destination": "Lunar Hotel", "seat_class": "Luxury"}
    ]
    for trip in trips:
        st.write(f"🛰️ {trip['date']} - {trip['destination']} ({trip['seat_class']})")
    
    st.markdown("### Travel Rewards")
    st.write("⭐ You have **3,000 SpaceMiles** available for discounts on future trips!")

elif menu == "AI Travel Assistant":
    st.subheader("🤖 AI-Powered Travel Recommendations")
    user_pref = st.text_area("Tell us about your space travel preferences:")
    
    def local_ai_response(query):
        responses = {
            "luxury": "AI suggests Lunar Hotel for a premium stay experience!",
            "cheap": "AI recommends an affordable stay in the Orbital Space Station!",
            "fast": "AI suggests the next express trip to ISS!"
        }
        for key, value in responses.items():
            if key in query.lower():
                return value
        return "AI suggests an exciting adventure based on your input!"
    
    if st.button("Get Recommendations"):
        st.write("🔍 Analyzing... ✅ " + local_ai_response(user_pref))

# Footer
st.markdown("---")
st.markdown("Developed for Dubai Space Travel Competition 🚀")
