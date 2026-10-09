import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Melbourne House Price Estimator", page_icon="🏠", layout="wide")

FEATURES = ["Regionname_Southern Metropolitan", "Rooms", "Distance", "Type_u", "Landsize"]
PRETTY = {
    "Regionname_Southern Metropolitan": "Southern Metropolitan region",
    "Rooms": "Rooms", "Distance": "Distance to CBD", "Type_u": "Unit (vs house)", "Landsize": "Land size",
}


# --- Load data ---
@st.cache_data
def load_data(n=2000, seed=42):
    """Synthetic Melbourne-style sales with the same columns as the original Melbourne housing model.

    Prices follow simple, plausible rules (more rooms and land raise the price, distance from the
    CBD and units lower it, Southern Metropolitan carries a premium) plus noise.
    """
    rng = np.random.default_rng(seed)
    region = rng.choice([0, 1], n, p=[0.65, 0.35])
    rooms = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.06, 0.24, 0.38, 0.22, 0.08, 0.02])
    distance = rng.gamma(2.5, 4.5, n).clip(0.5, 48).round(1)
    type_u = (rng.random(n) < np.where(distance < 8, 0.45, 0.15)).astype(int)
    landsize = np.where(type_u == 1, rng.normal(150, 60, n), rng.normal(550, 220, n)).clip(0, 2000).round()
    price = (
        350_000
        + 210_000 * rooms
        - 22_000 * distance
        + 330_000 * region
        - 260_000 * type_u
        + 420 * np.minimum(landsize, 1200)
    ) * rng.lognormal(0, 0.12, n)
    df = pd.DataFrame({
        "Regionname_Southern Metropolitan": region, "Rooms": rooms, "Distance": distance,
        "Type_u": type_u, "Landsize": landsize, "Price": np.maximum(price, 180_000).round(-3),
    })
    return df


@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred)}
    return model, metrics


df = load_data()
model, metrics = train_model(df)

# --- Sidebar: User Input ---
st.sidebar.header("🧮 Property features")


def user_input_features():
    region = st.sidebar.radio("Region", ["Southern Metropolitan", "Other region"], horizontal=True)
    prop_type = st.sidebar.radio("Property type", ["House", "Unit"], horizontal=True)
    rooms = st.sidebar.slider("Rooms", 1, 10, 3)
    distance = st.sidebar.number_input("Distance to CBD (km)", 0.0, 50.0, 7.0, step=0.5)
    landsize = st.sidebar.number_input("Land size (m²)", 0, 2000, 500 if prop_type == "House" else 150, step=10)
    data = {
        "Regionname_Southern Metropolitan": int(region == "Southern Metropolitan"),
        "Rooms": rooms,
        "Distance": distance,
        "Type_u": int(prop_type == "Unit"),
        "Landsize": landsize,
    }
    return pd.DataFrame([data])[FEATURES]


input_df = user_input_features()

# --- Main Layout ---
st.title("🏠 Melbourne Housing Price Prediction")
st.markdown(
    "Estimates the sale price of a Melbourne property with a **Random Forest regressor**. "
    "Set the property in the sidebar and the estimate updates instantly."
)
st.info(
    "Demo data: the model is trained on 2,000 synthetic sales generated in the app with the same "
    "columns as the Melbourne housing dataset, so it runs anywhere without downloads. Treat prices "
    "as illustrative, not a valuation.",
    icon="ℹ️",
)

with st.expander("How it works"):
    st.markdown(
        f"""
- **Algorithm:** scikit-learn `RandomForestRegressor` (200 trees, min 3 samples per leaf).
- **Features:** region (Southern Metropolitan or not), rooms, distance to the CBD, house vs unit and land size.
- **Validation:** 80/20 train-test split. On the hold-out set: R² = {metrics['r2']:.2f},
  mean absolute error = ${metrics['mae']:,.0f}.
- **Range:** the low and high figures are the 10th and 90th percentiles of the individual trees'
  predictions, a quick view of how much the trees agree.
"""
    )

tree_preds = np.array([t.predict(input_df.to_numpy()) for t in model.estimators_]).ravel()
prediction = float(model.predict(input_df)[0])
low, high = np.percentile(tree_preds, [10, 90])

m1, m2, m3 = st.columns(3)
m1.metric("🎯 Estimated price (AUD)", f"${prediction:,.0f}")
m2.metric("Likely range", f"${low:,.0f} – ${high:,.0f}")
m3.metric("Model accuracy (hold-out R²)", f"{metrics['r2']:.2f}")

c1, c2 = st.columns(2)
with c1:
    imp = pd.DataFrame({"feature": [PRETTY[f] for f in FEATURES], "importance": model.feature_importances_})
    fig_imp = px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                     title="What the model relies on most")
    fig_imp.update_layout(yaxis_title="", height=320)
    st.plotly_chart(fig_imp, width="stretch")
with c2:
    rooms_range = np.arange(1, 11)
    what_if = pd.concat([input_df] * len(rooms_range), ignore_index=True)
    what_if["Rooms"] = rooms_range
    fig_w = px.line(x=rooms_range, y=model.predict(what_if), markers=True,
                    labels={"x": "Rooms", "y": "Estimated price (AUD)"},
                    title="Same property with a different number of rooms")
    fig_w.add_vline(x=int(input_df["Rooms"].iloc[0]), line_dash="dot")
    fig_w.update_layout(height=320)
    st.plotly_chart(fig_w, width="stretch")

st.subheader("🔎 Your input vs. the training data")
shown = df.sample(600, random_state=1).assign(Type=lambda d: d["Type_u"].map({0: "House", 1: "Unit"}))
fig_d = px.scatter(shown, x="Distance", y="Price", color="Type",
                   opacity=0.45, labels={"Distance": "Distance to CBD (km)"},
                   title="Sale prices by distance to the CBD")
fig_d.add_scatter(x=input_df["Distance"], y=[prediction], mode="markers", name="Your property",
                  marker=dict(size=16, symbol="star", color="black"))
st.plotly_chart(fig_d, width="stretch")

with st.expander("Preview the training data"):
    st.dataframe(df.head(50), width="stretch", hide_index=True)

st.markdown("---")
st.caption("Model: Random Forest · Built with Streamlit and scikit-learn by DatariusAI")
