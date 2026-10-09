import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer Segmentation", page_icon="🧩", layout="wide")

NUMERIC_COLS = ["age", "annual_income", "account_balance", "num_products", "credit_score", "tenure_years"]
CATEGORICAL_COLS = ["channel_preference", "region", "gender"]
# Levels seen in training. The alphabetically first level was dropped by one-hot encoding.
LEVELS = {
    "channel_preference": ["ATM", "Branch", "Mobile", "Online"],
    "region": ["East", "North", "South", "West"],
    "gender": ["Female", "Male"],
}

# Business personas written for each segment.
segment_descriptions = {
    0: "Segment 0 – Digitally Comfortable Veterans: High digital engagement, prefers online and mobile channels.",
    1: "Segment 1 – Wealthy Traditionalists: Traditional users, prefer branch interaction, possibly older demographic.",
    2: "Segment 2 – Product-Rich Hybrids: Medium-income, high product ownership, mixed channel usage.",
    3: "Segment 3 – Low-Value Starters: New or low-value customers, fewer products, lower engagement.",
}
segment_colors = {0: "#91C8E4", 1: "#FFCF81", 2: "#97DECE", 3: "#FEC7B4"}


@st.cache_resource
def load_artifacts():
    return joblib.load("segmentation_scaler.joblib"), joblib.load("kmeans_model.joblib")


@st.cache_data
def sample_customers(n=300, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customer_id": [f"C{1000 + i}" for i in range(n)],
        "age": rng.integers(18, 70, n),
        "annual_income": rng.normal(134000, 66000, n).clip(15000, 400000).round(-2),
        "account_balance": rng.uniform(0, 100000, n).round(2),
        "num_products": rng.integers(1, 5, n),
        "credit_score": rng.integers(300, 850, n),
        "tenure_years": rng.integers(0, 30, n),
        "channel_preference": rng.choice(LEVELS["channel_preference"], n),
        "region": rng.choice(LEVELS["region"], n),
        "gender": rng.choice(LEVELS["gender"], n),
    })


def encode(df, expected):
    enc = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])
    return enc.reindex(columns=expected, fill_value=0).astype(float)


def validate(df):
    missing = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c not in df.columns]
    if missing:
        return f"Missing required column(s): {', '.join(missing)}."
    bad = [c for c in NUMERIC_COLS if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        return f"These columns must be numeric: {', '.join(bad)}."
    if df[NUMERIC_COLS].isna().any().any():
        return "Some numeric values are empty. Fill or drop those rows and upload again."
    return None


scaler, model = load_artifacts()
expected_cols = list(scaler.feature_names_in_)


def assign(df):
    X_scaled = scaler.transform(encode(df, expected_cols))
    labels = model.predict(X_scaled)
    dist = model.transform(X_scaled)
    return labels, dist


# Centroids back in original units, to describe what each cluster really contains.
centroids = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=expected_cols)
region_cols = [c for c in expected_cols if c.startswith("region_")]


def dominant_region(row):
    shares = {c.replace("region_", ""): row[c] for c in region_cols}
    shares["East"] = 1 - sum(shares.values())
    return max(shares, key=shares.get)


profile = pd.DataFrame({
    "Segment": range(model.n_clusters),
    "Main region": [dominant_region(r) for _, r in centroids.iterrows()],
    "Avg age": centroids["age"].round(1),
    "Avg income": centroids["annual_income"].round(0),
    "Avg balance": centroids["account_balance"].round(0),
    "Avg products": centroids["num_products"].round(2),
    "Avg credit score": centroids["credit_score"].round(0),
})

st.title("🧩 Customer Segmentation")
st.markdown(
    "Assigns bank customers to one of four segments with a **K-Means clustering** model. "
    "Try one customer, or segment a whole file in the batch tab (a built-in sample loads automatically)."
)

with st.expander("How it works"):
    st.markdown(
        f"""
- **Algorithm:** scikit-learn `KMeans` with k = {model.n_clusters}, on features standardised by a
  `StandardScaler`.
- **Inputs:** {len(NUMERIC_COLS)} numeric fields plus channel preference, region and gender, one-hot
  encoded into {len(expected_cols)} features.
- **Output:** the nearest cluster centre. "Distance" shows how typical a customer is for the segment.
- **Model note:** after scaling, the one-hot region columns have the largest spread, so the fitted
  clusters line up with region (see the centroid heatmap). The persona names are the analyst's
  business labels. A next iteration would cluster on behavioural features only.
"""
    )

tab_single, tab_batch, tab_profiles = st.tabs(["Single customer", "Batch segmentation", "Segment profiles"])

with tab_single:
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 90, 42)
        income = st.number_input("Annual income", 0, 1_000_000, 120000, step=1000)
        balance = st.number_input("Account balance", 0, 1_000_000, 45000, step=1000)
    with c2:
        products = st.slider("Number of products", 1, 8, 3)
        credit = st.slider("Credit score", 300, 850, 640)
        tenure = st.slider("Tenure (years)", 0, 40, 10)
    with c3:
        channel = st.selectbox("Channel preference", LEVELS["channel_preference"], index=2)
        region = st.selectbox("Region", LEVELS["region"], index=1)
        gender = st.selectbox("Gender", LEVELS["gender"])

    one = pd.DataFrame([{
        "age": age, "annual_income": income, "account_balance": balance, "num_products": products,
        "credit_score": credit, "tenure_years": tenure, "channel_preference": channel,
        "region": region, "gender": gender,
    }])
    labels, dist = assign(one)
    seg = int(labels[0])
    st.markdown(
        f"""<div style="background-color:{segment_colors[seg]};padding:1rem;border-radius:0.5rem;
        color:black;font-weight:bold;">{segment_descriptions[seg]}</div>""",
        unsafe_allow_html=True,
    )
    d = pd.DataFrame({"Segment": [f"Segment {i}" for i in range(model.n_clusters)], "Distance": dist[0]})
    fig_d = px.bar(d, x="Segment", y="Distance", title="Distance to each segment centre (lower = closer)",
                   color="Segment", color_discrete_sequence=list(segment_colors.values()))
    fig_d.update_layout(showlegend=False, height=300, margin=dict(t=40, b=10))
    st.plotly_chart(fig_d, width="stretch")

with tab_batch:
    st.markdown(
        "Upload a CSV with columns: " + ", ".join(f"`{c}`" for c in NUMERIC_COLS + CATEGORICAL_COLS)
        + ". Extra columns such as an ID are kept. No file? The app segments 300 synthetic customers."
    )
    sample = sample_customers()
    st.download_button("Download CSV template (sample data)", sample.to_csv(index=False).encode("utf-8"),
                       file_name="segmentation_sample.csv", mime="text/csv")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    df, source = sample, "built-in sample"
    if uploaded_file is not None:
        try:
            df, source = pd.read_csv(uploaded_file), uploaded_file.name
        except Exception as exc:
            st.error(f"Could not read that file as CSV: {exc}")
            st.stop()

    error = validate(df)
    if error:
        st.error(error)
        st.stop()

    labels, dist = assign(df)
    df = df.copy()
    df["Segment"] = labels
    df["Distance_to_centre"] = dist[np.arange(len(df)), labels].round(3)

    st.caption(f"Segmenting: {source}")
    counts = df["Segment"].value_counts().sort_index()
    cols = st.columns(model.n_clusters)
    for i, col in enumerate(cols):
        col.metric(f"Segment {i}", f"{int(counts.get(i, 0)):,}", f"{counts.get(i, 0) / len(df):.0%} of customers",
                   delta_color="off")

    g1, g2 = st.columns(2)
    with g1:
        fig_c = px.bar(x=[f"Segment {i}" for i in counts.index], y=counts.values,
                       color=[f"Segment {i}" for i in counts.index],
                       color_discrete_sequence=[segment_colors[i] for i in counts.index],
                       labels={"x": "", "y": "Customers"}, title="Segment sizes")
        fig_c.update_layout(showlegend=False)
        st.plotly_chart(fig_c, width="stretch")
    with g2:
        fig_s = px.scatter(df.assign(Segment=df["Segment"].astype(str)), x="annual_income", y="account_balance",
                           color="Segment", hover_data=["age", "region", "channel_preference"],
                           color_discrete_map={str(k): v for k, v in segment_colors.items()},
                           title="Income vs balance by segment")
        st.plotly_chart(fig_s, width="stretch")

    st.subheader("Segmentation results")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download segmented data", df.to_csv(index=False).encode("utf-8"),
                       "segmented_customers.csv", "text/csv")

with tab_profiles:
    st.markdown("### Segment insights")
    for seg_id, description in segment_descriptions.items():
        st.markdown(
            f"""<div style="background-color:{segment_colors[seg_id]};padding:1rem;margin-bottom:0.5rem;
            border-radius:0.5rem;color:black;font-weight:bold;">{description}</div>""",
            unsafe_allow_html=True,
        )
    st.markdown("### What the model's centroids contain")
    st.dataframe(profile, width="stretch", hide_index=True)
    z = pd.DataFrame(model.cluster_centers_, columns=expected_cols,
                     index=[f"Segment {i}" for i in range(model.n_clusters)])
    fig_h = px.imshow(z, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
                      title="Centroids in standardised units (how far each segment sits from the average)")
    st.plotly_chart(fig_h, width="stretch")
