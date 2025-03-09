import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json
import base64
from PIL import Image
import io

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
def generate_price(destination, seat_class, departure_date):
    """Generate dynamic price based on destination, seat class, and date."""
    base_prices = {
        "International Space Station": 350000,
        "Lunar Hotel": 650000,
        "Mars Colony": 1250000,
        "Orbital Space Yacht": 500000,
        "Venus Observatory": 850000,
        "Jupiter Gateway": 1750000
    }
    
    seat_multipliers = {
        'Economy': 1.0, 
        'Business': 1.75, 
        'Luxury': 2.5, 
        'VIP Zero-Gravity': 4.0
    }
    
    # Calculate days from now to add demand-based pricing
    days_to_trip = (departure_date - datetime.today().date()).days
    if days_to_trip < 30:
        urgency_factor = 1.2
    elif days_to_trip < 90:
        urgency_factor = 1.1
    else:
        urgency_factor = 1.0
    
    # Add some randomness to simulate market fluctuations
    market_factor = random.uniform(0.95, 1.05)
    
    base_price = base_prices.get(destination, 500000)
    return round(base_price * seat_multipliers.get(seat_class, 1.0) * urgency_factor * market_factor)

def get_accommodation_options(destination):
    """Get accommodation options based on destination."""
    accommodations = {
        "International Space Station": [
            {"name": "ISS Premium Module", "price": 25000, "rating": 4.6},
            {"name": "Cupola Observation Suite", "price": 45000, "rating": 4.9}
        ],
        "Lunar Hotel": [
            {"name": "Tranquility Base Resort", "price": 85000, "rating": 4.7},
            {"name": "Armstrong Luxury Domes", "price": 120000, "rating": 4.8},
            {"name": "Mare Serenitatis Villas", "price": 160000, "rating": 4.9}
        ],
        "Mars Colony": [
            {"name": "Olympus Mons Habitat", "price": 95000, "rating": 4.5},
            {"name": "Valles Marineris Suites", "price": 140000, "rating": 4.7},
            {"name": "Elysium Executive Compound", "price": 200000, "rating": 4.8}
        ],
        "Orbital Space Yacht": [
            {"name": "Standard Cabin", "price": 35000, "rating": 4.5},
            {"name": "Executive Suite", "price": 75000, "rating": 4.8}
        ],
        "Venus Observatory": [
            {"name": "Cloud City Apartment", "price": 90000, "rating": 4.6},
            {"name": "Venusian Skybridge Suite", "price": 135000, "rating": 4.8}
        ],
        "Jupiter Gateway": [
            {"name": "Ganymede Station Lodging", "price": 110000, "rating": 4.7},
            {"name": "Europa Underwater Dome", "price": 195000, "rating": 4.9}
        ]
    }
    return accommodations.get(destination, [{"name": "Standard Cabin", "price": 50000, "rating": 4.5}])

def get_ai_travel_tip(query=None):
    """Generate an AI travel tip using a pre-defined responses system."""
    if not query:
        travel_tips = [
            "Remember to acclimate gradually to zero-gravity environments by using anti-nausea patches 24 hours before departure.",
            "Pack light! Most space destinations offer rental equipment designed for low-gravity environments.",
            "The cosmic radiation shield in your cabin can be adjusted to view the stars safely during your journey.",
            "Stay hydrated on your space journey. In zero-gravity, your body's fluid distribution changes, affecting your thirst response.",
            "Experience the 'overview effect' by scheduling meditation time near observation windows for a life-changing perspective.",
            "Save space in your luggage by packing vacuum-compressed clothing specialized for long-duration space journeys.",
            "Lunar dust is extremely fine and abrasive - keep your habitat sealed and follow decontamination procedures when returning from moonwalks.",
            "For first-time space travelers, we recommend starting with a short orbital journey before attempting interplanetary travel.",
            "The ideal time to visit Mars is during its summer season when temperatures are more moderate.",
            "Consider booking your return journey during Earth's closest approach to your destination to minimize travel time.",
            "Secure loose items in your cabin before sleep cycles - even small objects can cause injury when floating freely.",
            "Our Orbital Space Yacht features artificial gravity sections for guests who prefer traditional dining experiences.",
            "Many space travelers report vivid dreams during the journey - bring a journal to document your experiences.",
            "The Tranquility Base Resort on the Moon offers exclusive tours of the Apollo 11 landing site.",
            "Remember that communication delays increase with distance from Earth - video calls from Mars have a 20-minute delay each way."
        ]
        return random.choice(travel_tips)
    else:
        # Simple keyword-based chatbot responses
        query = query.lower()
        if any(word in query for word in ["hello", "hi", "hey", "greetings"]):
            return "Hello future space traveler! How can I assist with your cosmic journey today?"
        elif any(word in query for word in ["cost", "price", "expensive", "money", "afford"]):
            return "Space travel costs vary widely based on destination, class, and duration. Our Economy trips to the International Space Station start at $350,000, while Mars adventures begin at $1.25 million. Remember that our loyalty program offers significant discounts for return travelers!"
        elif any(word in query for word in ["food", "eat", "meal", "dining"]):
            return "Space cuisine has evolved dramatically! Our vessels offer gourmet meals designed by Michelin-starred chefs and optimized for zero-gravity consumption. Luxury and VIP packages include molecular gastronomy experiences that are only possible in microgravity conditions."
        elif any(word in query for word in ["safety", "safe", "danger", "risk", "emergency"]):
            return "Safety is our absolute priority. All vessels exceed international space safety standards with redundant life support systems, radiation shielding, and emergency protocols. Our crew-to-passenger ratio is the highest in the industry, and all staff are trained in advanced emergency response procedures."
        elif any(word in query for word in ["time", "travel time", "duration", "long", "hours"]):
            return "Travel times vary by destination. Orbital trips take hours, the Moon is a 2-3 day journey, while Mars takes 3-5 months depending on planetary alignment. Our vessels feature stasis options for longer journeys to minimize the subjective travel time for passengers."
        elif any(word in query for word in ["wifi", "internet", "connection", "online"]):
            return "All our vessels provide high-speed quantum-encrypted communications with Earth. Near-Earth destinations enjoy real-time connections, while deeper space journeys utilize our proprietary FTL relay network to minimize latency."
        elif any(word in query for word in ["bathroom", "toilet", "shower", "hygiene"]):
            return "Our spacecraft feature advanced hygiene systems. Economy class uses traditional space toilets with vacuum suction, while Luxury and VIP classes enjoy our patented gravity-simulation shower systems that provide an Earth-like bathing experience even in zero gravity."
        elif any(word in query for word in ["family", "children", "kids", "child"]):
            return "We welcome travelers of all ages! Our Family Suites include specialized activities for younger explorers and educational programs about space science. Note that some destinations have minimum age requirements due to medical considerations."
        else:
            return "That's an interesting question about space travel. While I'm still learning about specific details, our customer service team can provide more information when you call our Dubai booking center at +971-800-SPACE."

def book_trip(destination, departure_date, seat_class, accommodation=None):
    """Book a trip and add to session state."""
    booking_id = f"BK-{random.randint(100000, 999999)}"
    
    # Get accommodation details or default to None
    accommodation_details = None
    if accommodation:
        accommodation_details = {
            "name": accommodation.get("name", "Standard Package"),
            "price": accommodation.get("price", 0)
        }
    
    booking = {
        "id": booking_id,
        "destination": destination,
        "departure_date": departure_date.strftime('%Y-%m-%d'),
        "seat_class": seat_class,
        "accommodation": accommodation_details,
        "status": "Confirmed",
        "booking_date": datetime.today().strftime('%Y-%m-%d')
    }
    
    st.session_state.bookings.append(booking)
    
    # Add loyalty points
    trip_price = generate_price(destination, seat_class, departure_date)
    loyalty_points = int(trip_price / 10000)  # 1 point per $10,000 spent
    st.session_state.user_profile["loyalty_points"] += loyalty_points
    
    # Update membership level based on total points
    total_points = st.session_state.user_profile["loyalty_points"]
    if total_points > 10000:
        st.session_state.user_profile["membership_level"] = "Platinum Explorer"
    elif total_points > 5000:
        st.session_state.user_profile["membership_level"] = "Gold Voyager"
    elif total_points > 2000:
        st.session_state.user_profile["membership_level"] = "Silver Voyager"
    else:
        st.session_state.user_profile["membership_level"] = "Bronze Traveler"
    
    return booking_id

# Custom CSS styling
def local_css():
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
        }
        .stApp {
            background-image: linear-gradient(to bottom, #0a192f, #0d1b34, #0f1d3a, #111f40, #132146);
            color: #e6f1ff;
        }
        .stButton>button {
            background-color: #64ffda;
            color: #0a192f;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #00bfa5;
            color: white;
        }
        h1, h2, h3 {
            color: #ccd6f6 !important;
        }
        .card {
            background-color: rgba(10, 25, 47, 0.7);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #233554;
        }
        .highlight {
            color: #64ffda;
            font-weight: bold;
        }
        .stDateInput>div>div {
            background-color: rgba(10, 25, 47, 0.7);
            color: white;
        }
        .stSelectbox>div>div {
            background-color: rgba(10, 25, 47, 0.7);
            color: white;
        }
        .stRadio>div {
            padding: 10px;
            background-color: rgba(10, 25, 47, 0.7);
            border-radius: 5px;
        }
        .countdown {
            font-size: 2rem;
            font-weight: bold;
            color: #64ffda;
            text-align: center;
            margin: 1rem 0;
        }
        .subtext {
            color: #8892b0;
            font-style: italic;
        }
        .info-box {
            background-color: rgba(100, 255, 218, 0.1);
            border-left: 3px solid #64ffda;
            padding: 1rem;
            margin: 1rem 0;
        }
        .sidebar .sidebar-content {
            background-color: #172a45;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply custom styling
local_css()

# ---- Sidebar Content ---- #
with st.sidebar:
    st.image("https://i.ibb.co/xgYDZRx/logo-placeholder.png", width=200)
    
    st.markdown(f"""
    <div class='card'>
        <h3>Welcome, {st.session_state.user_profile["name"]}</h3>
        <p>Membership: <span class='highlight'>{st.session_state.user_profile["membership_level"]}</span></p>
        <p>Loyalty Points: <span class='highlight'>{st.session_state.user_profile["loyalty_points"]}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        ["Book a Trip", "My Dashboard", "AI Travel Assistant", "Destinations", "About"]
    )
    
    st.markdown("""
    <div class="subtext" style="margin-top: 2rem;">
        Dubai Space Authority<br>
        Future Travel Initiative<br>
        Est. 2024
    </div>
    """, unsafe_allow_html=True)

# ---- Main Content Areas ---- #
if menu == "Book a Trip":
    st.header("🛸 Schedule Your Space Trip")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        destinations = [
            "International Space Station", 
            "Lunar Hotel", 
            "Mars Colony", 
            "Orbital Space Yacht",
            "Venus Observatory",
            "Jupiter Gateway"
        ]
        
        destination = st.selectbox("Choose Your Destination", destinations)
        
        # Show destination image based on selection
        destination_images = {
            "International Space Station": "https://i.ibb.co/TYfQ3NM/iss.jpg",
            "Lunar Hotel": "https://i.ibb.co/nfTXYv8/lunar-hotel.jpg",
            "Mars Colony": "https://i.ibb.co/h9stKPX/mars-colony.jpg",
            "Orbital Space Yacht": "https://i.ibb.co/ZXWBpBN/space-yacht.jpg",
            "Venus Observatory": "https://i.ibb.co/sjX0VjP/venus-obs.jpg",
            "Jupiter Gateway": "https://i.ibb.co/6WvXr5y/jupiter-station.jpg"
        }
        
        # Display destination image with placeholder fallback
        image_url = destination_images.get(destination, "https://i.ibb.co/C2BbX1X/space-generic.jpg")
        st.image(image_url, use_column_width=True)
        
        # Destination information
        destination_info = {
            "International Space Station": "Experience life in Earth's orbit at an altitude of 408km. Tour historic modules, conduct microgravity experiments, and witness 16 sunrises daily.",
            "Lunar Hotel": "Enjoy luxury accommodations on the Moon with Earth-views, low-gravity recreation, and guided moonwalks across the Sea of Tranquility.",
            "Mars Colony": "Visit humanity's frontier on the Red Planet. Experience Martian gravity, venture into Valles Marineris, and witness terraforming in action.",
            "Orbital Space Yacht": "Cruise Earth's orbit in ultimate luxury. Our space yacht offers panoramic views, gourmet dining, and spacewalk opportunities.",
            "Venus Observatory": "Study our mysterious sister planet from our cloud-top research station. Witness sulfuric acid storms and participate in atmospheric experiments.",
            "Jupiter Gateway": "Our most distant destination allows visits to Jupiter's most spectacular moons while maintaining a safe distance from the gas giant's intense radiation."
        }
        
        st.markdown(f"""
        <div class="info-box">
            {destination_info.get(destination, "Select a destination to learn more.")}
        </div>
        """, unsafe_allow_html=True)
        
        col_date, col_class = st.columns(2)
        
        with col_date:
            min_date = datetime.today().date() + timedelta(days=30)  # Minimum 30 days advanced booking
            max_date = datetime.today().date() + timedelta(days=365 * 2)  # Maximum 2 years in advance
            departure_date = st.date_input(
                "Select Departure Date",
                min_value=min_date,
                max_value=max_date,
                value=min_date + timedelta(days=30)
            )
        
        with col_class:
            seat_class = st.radio(
                "Select Seat Class",
                ["Economy", "Business", "Luxury", "VIP Zero-Gravity"]
            )
        
        # Calculate trip price
        trip_price = generate_price(destination, seat_class, departure_date)
        
        # Get accommodation options
        accommodations = get_accommodation_options(destination)
        
        # Show accommodation selection
        st.subheader("🏨 Select Accommodation")
        
        accommodation_cols = st.columns(len(accommodations))
        selected_accommodation = None
        
        for i, (col, accommodation) in enumerate(zip(accommodation_cols, accommodations)):
            with col:
                st.markdown(f"""
                <div class='card'>
                    <h4>{accommodation['name']}</h4>
                    <p>Price per night: <span class='highlight'>${accommodation['price']:,}</span></p>
                    <p>Rating: {'⭐' * int(accommodation['rating'])} ({accommodation['rating']})</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Select {accommodation['name']}", key=f"accom_{i}"):
                    selected_accommodation = accommodation
        
        if selected_accommodation:
            st.success(f"Selected accommodation: {selected_accommodation['name']}")
            
            # Calculate total with accommodation
            duration_mapping = {
                "International Space Station": 7,
                "Lunar Hotel": 10,
                "Mars Colony": 60,
                "Orbital Space Yacht": 5,
                "Venus Observatory": 14,
                "Jupiter Gateway": 90
            }
            
            duration = duration_mapping.get(destination, 7)  # Default 7 days
            accommodation_total = selected_accommodation['price'] * duration
            grand_total = trip_price + accommodation_total
            
            st.markdown(f"""
            <div class='card'>
                <h3>Trip Summary</h3>
                <p>Destination: <span class='highlight'>{destination}</span></p>
                <p>Departure: <span class='highlight'>{departure_date.strftime('%B %d, %Y')}</span></p>
                <p>Duration: <span class='highlight'>{duration} days</span></p>
                <p>Seat Class: <span class='highlight'>{seat_class}</span></p>
                <p>Accommodation: <span class='highlight'>{selected_accommodation['name']} (${selected_accommodation['price']:,}/night)</span></p>
                <hr>
                <p>Travel Cost: <span class='highlight'>${trip_price:,}</span></p>
                <p>Accommodation Total: <span class='highlight'>${accommodation_total:,}</span></p>
                <h4>Grand Total: <span class='highlight'>${grand_total:,}</span></h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Loyalty program information
            membership = st.session_state.user_profile["membership_level"]
            discount_rates = {
                "Bronze Traveler": 0,
                "Silver Voyager": 5,
                "Gold Voyager": 10,
                "Platinum Explorer": 15
            }
            
            discount_rate = discount_rates.get(membership, 0)
            if discount_rate > 0:
                discount_amount = round(grand_total * (discount_rate / 100))
                discounted_total = grand_total - discount_amount
                
                st.markdown(f"""
                <div class='info-box'>
                    <h4>Loyalty Program Discount</h4>
                    <p>As a <span class='highlight'>{membership}</span> member, you receive a <span class='highlight'>{discount_rate}%</span> discount.</p>
                    <p>Discount Amount: <span class='highlight'>${discount_amount:,}</span></p>
                    <p>Final Price: <span class='highlight'>${discounted_total:,}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                grand_total = discounted_total
            
            if st.button("Confirm Booking 🚀", key="confirm_with_accom", use_container_width=True):
                booking_id = book_trip(destination, departure_date, seat_class, selected_accommodation)
                st.balloons()
                st.success(f"Your trip to {destination} has been booked! Booking reference: {booking_id}")
                
                st.markdown(f"""
                <div class='info-box'>
                    <h4>Next Steps:</h4>
                    <p>1. Check your email for confirmation details</p>
                    <p>2. Complete your medical assessment</p>
                    <p>3. Begin your pre-flight training program 30 days before departure</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>Why Travel with Dubai Space?</h3>
            <p>✓ Industry-leading safety record</p>
            <p>✓ Luxury accommodations in space</p>
            <p>✓ Expert crews and guides</p>
            <p>✓ Flexible booking and payment plans</p>
            <p>✓ Exclusive access to cosmic landmarks</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
            <h3>Special Offers</h3>
            <p>🔥 15% off Mars Colony trips departing in 2026</p>
            <p>🔥 Companion flies free on Lunar Hotel packages</p>
            <p>🔥 Zero-G training included with all VIP bookings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Trip advisor-style reviews
        st.markdown("""
        <div class='card'>
            <h3>Traveler Reviews</h3>
            <p>⭐⭐⭐⭐⭐ "The zero-gravity suite on the Orbital Yacht exceeded all expectations!" - Emma D.</p>
            <p>⭐⭐⭐⭐⭐ "Sunrise from the Lunar Hotel was life-changing. Worth every dirham." - Mohammed A.</p>
            <p>⭐⭐⭐⭐ "Mars Colony visit was incredible, though the 3-month journey is not for everyone." - John P.</p>
        </div>
        """, unsafe_allow_html=True)

elif menu == "My Dashboard":
    st.header("🧑‍🚀 Your Space Travel Dashboard")
    
    if not st.session_state.bookings:
        # Sample bookings for demo
        st.session_state.bookings = [
            {
                "id": "BK-182734",
                "destination": "Lunar Hotel",
                "departure_date": (datetime.today() + timedelta(days=45)).strftime('%Y-%m-%d'),
                "seat_class": "Luxury",
                "accommodation": {"name": "Tranquility Base Resort", "price": 85000},
                "status": "Confirmed",
                "booking_date": (datetime.today() - timedelta(days=15)).strftime('%Y-%m-%d')
            },
            {
                "id": "BK-193842",
                "destination": "Orbital Space Yacht",
                "departure_date": (datetime.today() + timedelta(days=120)).strftime('%Y-%m-%d'),
                "seat_class": "VIP Zero-Gravity",
                "accommodation": {"name": "Executive Suite", "price": 75000},
                "status": "Awaiting Medical Clearance",
                "booking_date": (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
            }
        ]
    
    # Upcoming trips section
    st.subheader("Upcoming Trips")
    
    if st.session_state.bookings:
        next_trip = st.session_state.bookings[0]
        next_trip_date = datetime.strptime(next_trip["departure_date"], '%Y-%m-%d')
        days_remaining = (next_trip_date - datetime.today()).days
        
        # Countdown display
        st.markdown(f"""
        <div class="countdown">
            {days_remaining} Days
            <div style="font-size: 1rem; color: #8892b0;">until your next space adventure</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Trip details
        st.markdown(f"""
        <div class="card">
            <h3>Next Destination: {next_trip["destination"]}</h3>
            <p>Departure: <span class="highlight">{next_trip_date.strftime('%B %d, %Y')}</span></p>
            <p>Class: <span class="highlight">{next_trip["seat_class"]}</span></p>
            <p>Status: <span class="highlight">{next_trip["status"]}</span></p>
            <p>Booking Reference: <span class="highlight">{next_trip["id"]}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress tracker
        st.subheader("Trip Preparation Progress")
        
        # Calculate progress based on days remaining
        if days_remaining <= 7:
            progress = 0.9
            status = "Final preparation phase"
        elif days_remaining <= 30:
            progress = 0.7
            status = "Training in progress"
        elif days_remaining <= 60:
            progress = 0.4
            status = "Medical evaluations"
        else:
            progress = 0.2
            status = "Initial preparation"
        
        st.progress(progress)
        st.write(f"Current stage: **{status}**")
        
        # Custom timeline
        st.subheader("Pre-Flight Timeline")
        
        timeline_data = [
            {"days": 90, "event": "Medical evaluation & clearance"},
            {"days": 60, "event": "Begin zero-G adaptation training"},
            {"days": 30, "event": "Spacecraft systems orientation"},
            {"days": 15, "event": "Final medical check & equipment fitting"},
            {"days": 7, "event": "Pre-flight quarantine begins"},
            {"days": 1, "event": "Departure briefing & final preparations"},
            {"days": 0, "event": "Launch day 🚀"}
        ]
        
        for item in timeline_data:
            days_to_event = days_remaining - item["days"]
            if days_to_event >= 0:
                st.markdown(f"""
                <div style="display: flex; margin-bottom: 0.5rem;">
                    <div style="width: 80px; color: {'#64ffda' if days_to_event == 0 else '#8892b0'};">
                        {f"Today" if days_to_event == 0 else f"In {days_to_event} days"}
                    </div>
                    <div style="flex-grow: 1; border-left: 2px solid {'#64ffda' if days_to_event == 0 else '#233554'}; padding-left: 1rem;">
                        <span style="color: {'#64ffda' if days_to_event == 0 else '#ccd6f6'};">{item["event"]}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # All bookings table
        st.subheader("All Bookings")
        bookings_df = pd.DataFrame(st.session_state.bookings)
        
        # Process DataFrame for display
        if 'accommodation' in bookings_df.columns:
            bookings_df['accommodation'] = bookings_df['accommodation'].apply(
                lambda x: x['name'] if isinstance(x, dict) and 'name' in x else 'Standard Package'
            )
        
        # Select and reorder columns
        display_columns = [
            'id', 'destination', 'departure_date', 'seat_class', 
            'accommodation', 'status'
        ]
        display_columns = [col for col in display_columns if col in bookings_df.columns]
        
        # Rename columns for display
        column_names = {
            'id': 'Booking ID',
            'destination': 'Destination',
            'departure_date': 'Departure',
            'seat_class': 'Class',
            'accommodation': 'Accommodation',
            'status': 'Status'
        }
        
        bookings_table = bookings_df[display_columns].rename(columns=column_names)
        st.dataframe(bookings_table, use_container_width=True)
    
    else:
        st.info("You don't have any upcoming trips. Book your first space adventure today!")
    
    # Loyalty program status
    st.subheader("Loyalty Program Status")
    
    # Create a custom progress bar for loyalty level
    membership_levels = ["Bronze Traveler", "Silver Voyager", "Gold Voyager", "Platinum Explorer"]
    current_level = st.session_state.user_profile["membership_level"]
    current_points = st.session_state.user_profile["loyalty_points"]
    
    # Define thresholds
    thresholds = [0, 2000, 5000, 10000]
    
    # Find current level index
    current_level_idx = membership_levels.index(current_level)
    
    # Calculate progress to next level
    if current_level_idx < len(membership_levels) - 1:
        next_level = membership_levels[current_level_idx + 1]
        next_threshold = thresholds[current_level_idx + 1]
        prev_threshold = thresholds[current_level_idx]
        progress_to_next = (current_points - prev_threshold) / (next_threshold - prev_threshold)
        points_needed = next_threshold - current_points
        
        st.markdown(f"""
        <div class="card">
            <h4>Current Level: <span class="highlight">{current_level}</span></h4>
            <p>Points: <span class="highlight">{
