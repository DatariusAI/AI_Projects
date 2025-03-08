import streamlit as st
import datetime
import random
import requests

# App Configurations
st.set_page_config(page_title="Dubai Space Travel Booking", layout="wide")

# Title & Intro Section
st.title("🚀 Dubai Space Travel Booking Platform")
st.markdown("### Book Your Futuristic Space Trip Now!")
st.image("https://source.unsplash.com/1600x900/?space,stars,galaxy", use_container_width=True)

# Sidebar Navigation
menu = st.sidebar.selectbox("Navigation", ["Home", "Book a Trip", "Dashboard", "AI Travel Assistant"])

# Session State for Bookings
if "bookings" not in st.session_state:
    st.session_state["bookings"] = []

# Dynamic Pricing Function
def get_dynamic_price(destination, seat_class, departure_date):
    base_prices = {"ISS": 500000, "Lunar Hotel": 1200000, "Orbital Space Station": 800000}
    class_multiplier = {"Economy": 1, "Luxury": 2, "VIP Zero-Gravity": 3}
    
    days_until_departure = (departure_date - datetime.date.today()).days
    demand_factor = 1.5 if days_until_departure < 30 else random.uniform(1.1, 1.8)
    
    return base_prices[destination] * class_multiplier[seat_class] * demand_factor

# AI Model: Free Hugging Face API
def ai_travel_recommendation(query):
    """Uses Hugging Face's API for AI-based travel recommendations."""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
    headers = {"Authorization": f"Bearer hf_free_api_key"}  # Replace with a real key if required

    payload = {"inputs": f"Recommend a space travel destination based on this preference: {query}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "AI is unable to fetch recommendations at the moment. Try again later."

# Pages Navigation
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
        st.image("https://source.unsplash.com/400x300/?rocket,spaceship", use_container_width=True)
        st.write(f"📍 Destination: {destination}")
        st.write(f"💺 Seat Class: {seat_class}")
        price = get_dynamic_price(destination, seat_class, departure_date)
        st.write(f"💰 Estimated Price: ${price:,.2f}")
        
        if st.button("Confirm Booking"):
            new_booking = {
                "date": departure_date.strftime("%Y-%m-%d"),
                "destination": destination,
                "seat_class": seat_class,
                "price": f"${price:,.2f}"
            }
            st.session_state["bookings"].append(new_booking)
            st.success("✅ Your trip is booked! Check the dashboard for details.")

elif menu == "Dashboard":
    st.subheader("📊 User Dashboard")
    countdown = datetime.datetime(2025, 6, 1) - datetime.datetime.now()
    st.write(f"🚀 Next Launch Countdown: {countdown.days} days")

    st.markdown("### Your Upcoming Trips")
    if st.session_state["bookings"]:
        for trip in st.session_state["bookings"]:
            st.write(f"🛰️ {trip['date']} - {trip['destination']} ({trip['seat_class']}) - {trip['price']}")
    else:
        st.info("No upcoming trips yet. Book your first space journey!")

    st.markdown("### Travel Rewards")
    st.write("⭐ You have **3,000 SpaceMiles** available for discounts on future trips!")

elif menu == "AI Travel Assistant":
    st.subheader("🤖 AI-Powered Travel Recommendations")
    user_pref = st.text_area("Tell us about your space travel preferences:")

    if st.button("Get Recommendations"):
        st.write("🔍 AI is analyzing your preferences...")
        ai_response = ai_travel_recommendation(user_pref)
        st.write("🤖 AI Suggests: " + ai_response)

# Footer
st.markdown("---")
st.markdown("Developed for Dubai Space Travel Competition 🚀")
