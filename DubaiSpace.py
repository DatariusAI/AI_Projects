import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# ---- Helper Functions ---- #
def generate_price(seat_class):
    """Generate price based on seat class."""
    base_price = 500000  # Base price in USD
    multipliers = {'Economy': 1, 'Luxury': 2, 'VIP Zero-Gravity': 3.5}
    return base_price * multipliers.get(seat_class, 1)

# ---- AI Chatbot Using Langchain & Free Model ---- #
def get_ai_travel_tip():
    """Generate an AI travel tip without API costs."""
    try:
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        response = model([HumanMessage(content="Provide a futuristic space travel tip.")])
        return response.content.strip()
    except Exception:
        return "AI Assistant is currently unavailable. Try again later."

# ---- Streamlit UI ---- #
st.set_page_config(page_title="Dubai Space Travel", layout="wide")
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
        st.success(f"Your trip to {destination} on {departure_date.strftime('%Y-%m-%d')} is booked!")
        st.balloons()

elif menu == "My Dashboard":
    st.header("🧑‍🚀 Your Space Travel Dashboard")
    bookings = pd.DataFrame({
        "Destination": ["Lunar Hotel", "Orbital Space Yacht"],
        "Departure Date": [(datetime.today() + timedelta(days=random.randint(10, 90))).strftime('%Y-%m-%d') for _ in range(2)],
        "Seat Class": ["Luxury", "VIP Zero-Gravity"],
        "Status": ["Confirmed", "Upcoming"]
    })
    st.table(bookings)
    next_trip_date = datetime.strptime(bookings.iloc[0]['Departure Date'], '%Y-%m-%d')
    countdown = (next_trip_date - datetime.today()).days
    st.write(f"**{countdown} days** until your next space adventure! 🚀")

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
