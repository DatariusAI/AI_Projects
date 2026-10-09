"""Dubai to the Stars: a hackathon prototype for booking commercial space trips."""
import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dubai to the Stars", page_icon="🚀", layout="wide")

# ---- Pricing model ---- #
DESTINATIONS = pd.DataFrame(
    [
        ("Orbital Space Yacht", 300_000, 3, "Low Earth orbit cruise with a panoramic deck."),
        ("International Space Station", 450_000, 10, "Research stay with the station crew."),
        ("Lunar Hotel", 1_200_000, 14, "Surface hotel with moonwalk excursions."),
        ("Mars Colony", 4_500_000, 210, "One-way transit to the Mars settlement (about 7 months)."),
    ],
    columns=["Destination", "Base price (USD)", "Duration (days)", "Description"],
).set_index("Destination")

SEAT_CLASSES = {"Economy": 1.0, "Luxury": 2.0, "VIP Zero-Gravity": 3.5}
EARLY_BIRD_DAYS = 180
EARLY_BIRD_DISCOUNT = 0.10
TRAINING_FEE = 25_000  # per passenger, mandatory


def quote(destination: str, seat_class: str, passengers: int, departure: date) -> dict:
    base = DESTINATIONS.loc[destination, "Base price (USD)"] * SEAT_CLASSES[seat_class]
    days_ahead = (departure - date.today()).days
    discount = EARLY_BIRD_DISCOUNT if days_ahead >= EARLY_BIRD_DAYS else 0.0
    seat_total = base * passengers
    return {
        "seat_price": base,
        "seat_total": seat_total,
        "discount": seat_total * discount,
        "training": TRAINING_FEE * passengers,
        "total": seat_total * (1 - discount) + TRAINING_FEE * passengers,
        "days_ahead": days_ahead,
    }


# ---- Travel assistant ---- #
TIPS = {
    "UAE's Space Vision": (
        "The UAE has made space a national priority. The Mohammed bin Rashid Space Centre (MBRSC) runs the "
        "Emirates Mars Mission (Hope probe) and the Mars 2117 initiative, which aims at a human settlement on Mars. "
        "Continued investment in research and technology keeps the UAE among the fastest-moving space nations."
    ),
    "Space Travel Experience": (
        "Flights from Dubai offer panoramic observation decks, artificial-gravity sections and cuisine designed for "
        "zero gravity. Whether you head to the ISS, the Lunar Hotel or Mars, trained astronauts guide you at every step."
    ),
    "Safety & Training": (
        "Every passenger completes zero-gravity training and a full health assessment before departure. "
        "Spacecraft carry redundant life-support systems and rehearsed emergency protocols."
    ),
    "Upcoming Missions": (
        "Planned missions include lunar surface exploration with robotic rovers and deep-space probes, "
        "building toward the long-term Mars 2117 goal of a permanent human settlement."
    ),
}


def get_setting(name: str) -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value or os.getenv(name, "")).strip()


def llm_provider():
    if get_setting("GROQ_API_KEY"):
        return "Groq"
    if get_setting("OPENAI_API_KEY"):
        return "OpenAI"
    return None


SYSTEM_PROMPT = (
    "You are a friendly travel assistant for a fictional Dubai space-tourism company in a hackathon demo. "
    "Destinations: Orbital Space Yacht, International Space Station, Lunar Hotel, Mars Colony. "
    "Answer in 120 words or fewer. Say clearly when something is fictional."
)


@st.cache_data(ttl=3600, show_spinner=False)
def ask_llm(question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    if get_setting("GROQ_API_KEY"):
        from groq import Groq

        client = Groq(api_key=get_setting("GROQ_API_KEY"))
        model = get_setting("GROQ_MODEL") or "llama-3.3-70b-versatile"
    else:
        from openai import OpenAI

        client = OpenAI(api_key=get_setting("OPENAI_API_KEY"))
        model = get_setting("OPENAI_MODEL") or "gpt-4o-mini"
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.4, max_tokens=300)
    return resp.choices[0].message.content.strip()


# ---- Session state ---- #
if "bookings" not in st.session_state:
    today = date.today()
    st.session_state.bookings = [
        {"Destination": "Lunar Hotel", "Departure": today + timedelta(days=45), "Seat class": "Luxury",
         "Passengers": 1, "Total (USD)": quote("Lunar Hotel", "Luxury", 1, today + timedelta(days=45))["total"], "Status": "Sample"},
        {"Destination": "Orbital Space Yacht", "Departure": today + timedelta(days=200), "Seat class": "VIP Zero-Gravity",
         "Passengers": 2, "Total (USD)": quote("Orbital Space Yacht", "VIP Zero-Gravity", 2, today + timedelta(days=200))["total"], "Status": "Sample"},
    ]

# ---- UI ---- #
st.title("🚀 Dubai to the Stars")
st.caption(
    "A hackathon prototype of a space-tourism booking platform departing from Dubai. "
    "Prices and missions are fictional. Nothing is charged."
)

menu = st.sidebar.radio("Navigation", ["Book a Trip", "My Dashboard", "Travel Assistant", "About"])

if menu == "Book a Trip":
    st.header("🛸 Plan your trip")
    left, right = st.columns([1, 1.2])
    with left:
        destination = st.selectbox("Destination", DESTINATIONS.index, index=2)
        st.caption(DESTINATIONS.loc[destination, "Description"])
        departure = st.date_input(
            "Departure date",
            value=date.today() + timedelta(days=90),
            min_value=date.today() + timedelta(days=7),
            max_value=date.today() + timedelta(days=730),
            help="Bookings open from 7 days to 2 years ahead.",
        )
        seat_class = st.radio("Seat class", list(SEAT_CLASSES), horizontal=True)
        passengers = st.number_input("Passengers", min_value=1, max_value=6, value=1, step=1)

    q = quote(destination, seat_class, int(passengers), departure)
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total price", f"${q['total']:,.0f}")
        m2.metric("Trip length", f"{DESTINATIONS.loc[destination, 'Duration (days)']} days")
        m3.metric("Days to launch", q["days_ahead"])
        breakdown = pd.DataFrame(
            {
                "Item": [f"Seats ({passengers} × ${q['seat_price']:,.0f})", "Early-bird discount", "Mandatory training"],
                "USD": [q["seat_total"], -q["discount"], q["training"]],
            }
        )
        st.dataframe(breakdown.style.format({"USD": "{:,.0f}"}), hide_index=True, width="stretch")
        if q["discount"] == 0:
            st.caption(f"Book at least {EARLY_BIRD_DAYS} days ahead to save {EARLY_BIRD_DISCOUNT:.0%} on seats.")

        compare = pd.DataFrame(
            [(d, c, DESTINATIONS.loc[d, "Base price (USD)"] * m) for d in DESTINATIONS.index for c, m in SEAT_CLASSES.items()],
            columns=["Destination", "Seat class", "Seat price (USD)"],
        )
        fig = px.bar(compare, x="Destination", y="Seat price (USD)", color="Seat class", barmode="group", log_y=True,
                     title="Seat price per passenger (log scale)")
        fig.update_layout(height=340, margin=dict(t=40, b=10), legend_title=None)
        st.plotly_chart(fig, width="stretch")

    if st.button("Book now 🚀", type="primary"):
        st.session_state.bookings.append(
            {"Destination": destination, "Departure": departure, "Seat class": seat_class,
             "Passengers": int(passengers), "Total (USD)": q["total"], "Status": "Confirmed"}
        )
        st.success(f"Booked: {destination} on {departure:%d %b %Y} for {passengers} passenger(s). See it in My Dashboard.")
        st.balloons()

elif menu == "My Dashboard":
    st.header("🧑‍🚀 Your space travel dashboard")
    bookings = pd.DataFrame(st.session_state.bookings).sort_values("Departure")
    upcoming = bookings[bookings["Departure"] >= date.today()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Trips booked", len(bookings))
    m2.metric("Total spend", f"${bookings['Total (USD)'].sum():,.0f}")
    if not upcoming.empty:
        nxt = upcoming.iloc[0]
        m3.metric("Next launch in", f"{(nxt['Departure'] - date.today()).days} days", help=nxt["Destination"])
    st.dataframe(bookings.style.format({"Total (USD)": "{:,.0f}"}), hide_index=True, width="stretch")
    st.caption("Rows marked Sample are pre-loaded so the dashboard is not empty. Your bookings last for this browser session.")
    fig = px.timeline(
        bookings.assign(End=[d + timedelta(days=int(DESTINATIONS.loc[x, "Duration (days)"])) for d, x in zip(bookings["Departure"], bookings["Destination"])]),
        x_start="Departure", x_end="End", y="Destination", color="Seat class", title="Trip timeline",
    )
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

elif menu == "Travel Assistant":
    st.header("🤖 Space travel assistant")
    topic = st.selectbox("Pick a topic", list(TIPS))
    st.info(TIPS[topic])

    st.subheader("Ask your own question")
    provider = llm_provider()
    if provider:
        st.caption(f"Answers come from an LLM via {provider}.")
    else:
        st.warning(
            "Free-form answers need an LLM key. Add GROQ_API_KEY or OPENAI_API_KEY to the app's secrets. "
            "The topic guides above work without one."
        )
    question = st.text_input("Your question", placeholder="What should I pack for the Lunar Hotel?", disabled=not provider)
    if st.button("Ask", disabled=not provider):
        if not question.strip():
            st.error("Please type a question first.")
        else:
            try:
                with st.spinner("Thinking…"):
                    st.success(ask_llm(question.strip()[:500]))
            except Exception as exc:
                st.error(f"The assistant could not answer right now ({type(exc).__name__}). Please try again later.")

elif menu == "About":
    st.header("🌌 About the platform")
    st.write(
        "This prototype was built for the Dubai Space Travel Hackathon. It imagines Dubai as a hub for commercial "
        "space tourism, in line with the UAE's space programme led by the Mohammed bin Rashid Space Centre "
        "(Emirates Mars Mission, Mars 2117)."
    )
    with st.expander("How it works", expanded=True):
        st.markdown(
            f"""
- **Pricing:** seat price = destination base price × class multiplier ({', '.join(f'{k} ×{v}' for k, v in SEAT_CLASSES.items())}).
  Bookings {EARLY_BIRD_DAYS}+ days ahead get {EARLY_BIRD_DISCOUNT:.0%} off seats, and each passenger pays a ${TRAINING_FEE:,} training fee.
- **Dashboard:** bookings are kept in Streamlit session state, so they reset when you close the tab.
- **Assistant:** curated topic guides work offline. Free-form questions use an LLM (Groq or OpenAI) only when a key is set in secrets.
"""
        )
    st.dataframe(DESTINATIONS, width="stretch")
