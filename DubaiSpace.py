import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
import requests

# ---- Helper Functions ---- #
def generate_price(seat_class):
    base_price = 500000  # Base price in USD
    multipliers = {'Economy': 1, 'Luxury': 2, 'VIP Zero-Gravity': 3.5}
    return base_price * multipliers[seat_class]

# ---- AI Travel Assistant with Dynamic Options ---- #
def get_ai_travel_tip(option):
    tips = {
        "UAE's Space Vision": (
            "The United Arab Emirates has positioned itself as a leader in space exploration under the visionary guidance "
            "of His Highness Sheikh Mohammed bin Rashid Al Maktoum and His Highness Sheikh Mohamed bin Zayed Al Nahyan. "
            "Through the Mohammed bin Rashid Space Centre (MBRSC), the UAE has launched groundbreaking projects, including "
            "the Emirates Mars Mission and the Mars 2117 initiative, paving the way for interplanetary human settlement. "
            "With continued investment in research and technology, the UAE remains at the forefront of space innovation."
        ),
        "Space Travel Experience": (
            "Embarking on a space journey from Dubai is a once-in-a-lifetime experience. Our commercial flights offer "
            "unparalleled luxury and safety, featuring panoramic observation decks, artificial gravity sections, and "
            "Michelin-starred cuisine tailored for zero gravity. Whether you're heading to the International Space Station, "
            "the Lunar Hotel, or Mars, expect an extraordinary voyage with trained astronauts guiding you every step of the way."
        ),
        "Safety & Training": (
            "Dubai Space Travel prioritizes safety with cutting-edge spacecraft and rigorous training programs. "
            "Before your journey, you will undergo zero-gravity training and a full health assessment to ensure your "
            "readiness. Our state-of-the-art life-support systems and emergency protocols make space travel safer than ever, "
            "allowing you to explore the cosmos with confidence."
        ),
        "Upcoming Missions": (
            "The UAE is set to launch several ambitious missions, including lunar surface exploration and deep-space probes. "
            "The Emirates Lunar Mission will see the deployment of robotic rovers, while Mars 2117 aims to establish a "
            "permanent human colony on the Red Planet. These projects not only enhance scientific knowledge but also "
            "prepare humanity for the next frontier of space habitation."
        )
    }
    return tips.get(option, "Please select a topic to learn more about Dubai's space travel initiatives.")

# ---- Streamlit UI ---- #
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
    
    option = st.selectbox("Choose a topic:", ["Select", "UAE's Space Vision", "Space Travel Experience", "Safety & Training", "Upcoming Missions"])
    
    if st.button("Get AI Travel Tip 🛰️") and option != "Select":
        ai_tip = get_ai_travel_tip(option)
        st.info(ai_tip)
    elif option == "Select":
        st.warning("Please select a topic to get more information.")
    
elif menu == "About":
    st.header("🌌 About the Platform")
    st.write(
        "This futuristic booking platform allows users to schedule and manage their space journeys from Dubai. "
        "Designed for the Dubai Space Travel Hackathon, it aligns with the UAE's vision of pioneering space travel. "
        "The United Arab Emirates has made remarkable advancements in space exploration, thanks to the leadership "
        "of His Highness Sheikh Mohammed bin Rashid Al Maktoum and His Highness Sheikh Mohamed bin Zayed Al Nahyan. "
        "Their efforts in creating the Mohammed bin Rashid Space Centre (MBRSC) and initiatives like the Emirates "
        "Mars Mission showcase their commitment to positioning Dubai as a global hub for space tourism.\n\n"
        "Dubai’s commercial space travel industry is dedicated to providing travelers with cutting-edge spacecraft, "
        "exceptional in-flight experiences, and highly trained crews to ensure maximum safety and enjoyment. "
        "Flights departing from Dubai include immersive space journeys ranging from short orbital getaways to full-fledged "
        "interplanetary adventures. Whether you're heading to the International Space Station for scientific research, "
        "a lunar vacation, or preparing for Mars colonization, Dubai Space Travel ensures an unparalleled experience. "
        "With the UAE’s continued investments in space technology and infrastructure, the future of commercial space travel "
        "has never looked brighter."
    )
