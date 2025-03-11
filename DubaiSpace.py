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

# ---- AI Travel Tip Alternative ---- #
def get_ai_travel_tip():
    """Provide a text-based travel tip if AI is unavailable."""
    return (
        "The United Arab Emirates has established itself as a global leader in space exploration, "
        "thanks to the visionary leadership of its rulers. The Mohammed bin Rashid Space Centre (MBRSC) "
        "has spearheaded numerous ambitious projects, including the historic Emirates Mars Mission, which placed "
        "the Hope Probe into orbit around the Red Planet. This remarkable achievement has cemented Dubai’s position "
        "in the space industry, pushing the boundaries of innovation and technology.\n\n"
        "Under the guidance of His Highness Sheikh Mohammed bin Rashid Al Maktoum and His Highness Sheikh Mohamed bin Zayed Al Nahyan, "
        "Dubai continues to push forward in making commercial space travel accessible to all. With ongoing developments "
        "in space tourism, research, and interplanetary travel, the UAE is setting new standards in aerospace innovation.\n\n"
        "Our commercial space flights offer state-of-the-art spacecraft designed with passenger comfort and safety as the top priority. "
        "Flights to the International Space Station, the Lunar Hotel, and even Mars are equipped with advanced life-support systems, "
        "artificial gravity modules, and panoramic observation decks. Whether you choose our Economy, Luxury, or VIP Zero-Gravity "
        "packages, you will experience the finest space travel services.\n\n"
        "The UAE’s commitment to space exploration is further solidified by the ambitious Mars 2117 project, aiming to establish "
        "a sustainable human colony on Mars. The Emirates Lunar Mission is also progressing rapidly, with a focus on developing "
        "lunar habitats for long-term space habitation. These initiatives ensure that the UAE remains a dominant force in the "
        "future of space travel, bringing the dream of interplanetary tourism closer to reality."
    )

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
    
    if st.button("Get AI Travel Tip 🛰️"):
        ai_tip = get_ai_travel_tip()
        st.info(ai_tip)
    
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
