"""Ambiscions case study: retail analytics from description to prescription.

Data (sales, promotions, customers, products, stores) is loaded from Google
Drive and cached for 24 hours. Sections: descriptive, diagnostic (incl. an
optional LLM analyst), predictive (XGBoost) and prescriptive (what-if forecast).
"""
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Ambiscions Case Study", page_icon="🛍️", layout="wide")

DEPLOYED_URL = os.environ.get(
    "STREAMLIT_DEPLOY_URL", "https://kewlfunky2023-blank-app-ambiscions-case-study-teoy6m.streamlit.app/"
)

# ---------------------------------------------------------------- data
DRIVE = "https://drive.google.com/uc?id="
FILES = {"sales": "1Lj7Zke3LHCOAqPIRwFJOrZeUbkMhyEfB", "promotion": "1idK_ctZD72TDWXy10qymhQ308qniZCvH",
         "customer": "18n8qug_i4OvRzFo1E0-pPBU6L038PH6L", "product": "1NYaVT8pnypvqGRGweiwwP7TrqR4cMPNn",
         "store": "1LIuZxAsBiEgNT0XfWkhM_YkPohEO_8uy"}


@st.cache_data(ttl=24 * 3600, show_spinner="Loading case-study data…")
def load_tables():
    return {name: pd.read_csv(DRIVE + fid, encoding="utf8") for name, fid in FILES.items()}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def prepare():
    t = load_tables()
    sales, promo, cust, prod, store = (t[k].copy() for k in ("sales", "promotion", "customer", "product", "store"))
    sales["Transaction_Date"] = pd.to_datetime(sales["Transaction_Date"])
    sales = sales.sort_values("Transaction_Date").reset_index(drop=True)
    sales["Quantity_Sold"] = sales["Quantity_Sold"].clip(upper=500)
    sales["Total_Amount"] = sales["Quantity_Sold"] * sales["Price_Per_Unit"]
    cust["First_Purchase_Date"] = pd.to_datetime(cust["First_Purchase_Date"])
    promo["Start_Date"] = pd.to_datetime(promo["Start_Date"])
    promo["End_Date"] = pd.to_datetime(promo["End_Date"])

    full = (sales.merge(prod, on="Product_ID", how="left")
                 .merge(cust, on="Customer_ID", how="left")
                 .merge(store, on="Store_ID", how="left")
                 .merge(promo, on="Promotion_ID", how="left"))
    full["Discount"] = full["Discount"].fillna(0)
    full["On_Promotion"] = full["Discount"] > 0
    full["Transaction_Year"] = full["Transaction_Date"].dt.year
    full["Transaction_Month"] = full["Transaction_Date"].dt.month
    return sales, promo, cust, prod, store, full


try:
    sales_data, promotion_data, customer_data, product_data, store_data, full = prepare()
except Exception as exc:
    st.title("🛍️ Ambiscions Case Study")
    st.error(
        "Could not load the case-study data from Google Drive right now. "
        "This usually clears up after a minute, so please refresh the page."
    )
    st.caption(f"Details: {type(exc).__name__}: {str(exc)[:200]}")
    st.stop()


def section_guard(name):
    """Decorator: show a friendly warning instead of crashing one section."""
    def wrap(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:
                st.warning(f"The {name} section could not be computed for this data ({type(exc).__name__}: {exc}).")
        return inner
    return wrap


# ---------------------------------------------------------------- models (cached)
@st.cache_data(show_spinner="Clustering customers…")
def segment_customers(k: int):
    per_customer = sales_data.groupby("Customer_ID").agg(Quantity_Sold=("Quantity_Sold", "sum")).reset_index()
    df = customer_data.merge(per_customer, on="Customer_ID")
    pre = ColumnTransformer([
        ("num", StandardScaler(), ["Age", "Quantity_Sold"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Gender", "Income_Level"]),
    ])
    X = pre.fit_transform(df)
    df["Cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X).astype(str)
    return df


@st.cache_resource(show_spinner="Training XGBoost model…")
def train_predictive():
    data = pd.get_dummies(full, columns=["Category", "Gender", "Income_Level", "Location"], drop_first=True)
    data = data.select_dtypes(include=[np.number, "bool"]).astype(float)
    leak = [c for c in ("Quantity_Sold", "Total_Amount") if c in data]
    ids = [c for c in data.columns if c.endswith("_ID")]
    y = data["Quantity_Sold"]
    X = data.drop(columns=leak + ids)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.1, max_depth=5, n_estimators=200)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "baseline": float(np.sqrt(mean_squared_error(y_val, np.full(len(y_val), y_train.mean())))),
        "r2": float(r2_score(y_val, pred)),
    }
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return metrics, importance


PRESCRIPTIVE_BASE = ["Age", "Discount", "Sentiment", "Transaction_Year", "Transaction_Month"]


@st.cache_resource(show_spinner="Training what-if model…")
def train_prescriptive():
    df = full.copy()
    dummies = pd.get_dummies(df[["Gender", "Income_Level"]].astype(str))
    base = [c for c in PRESCRIPTIVE_BASE if c in df and pd.api.types.is_numeric_dtype(df[c])]
    X = pd.concat([df[base], dummies], axis=1).astype(float)
    y = df["Quantity_Sold"]
    model = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.1, max_depth=5, n_estimators=150)
    model.fit(X, y)
    return model, list(X.columns)


# ---------------------------------------------------------------- tiny sentiment scorer
POS = {"good", "great", "excellent", "love", "amazing", "happy", "perfect", "nice", "recommend", "fast", "best", "fresh", "friendly"}
NEG = {"bad", "poor", "terrible", "hate", "awful", "slow", "worst", "broken", "expensive", "rude", "late", "disappointed", "dirty"}
NEGATORS = {"not", "no", "never", "isn't", "wasn't", "don't", "didn't"}


def sentiment_score(text: str):
    words = re.findall(r"[a-z']+", text.lower())
    score = 0
    for i, w in enumerate(words):
        s = 1 if w in POS else -1 if w in NEG else 0
        if s and i > 0 and words[i - 1] in NEGATORS:
            s = -s
        score += s
    return int(np.sign(score)), score


# ---------------------------------------------------------------- optional LLM
def get_setting(name: str) -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value or os.getenv(name, "")).strip()


def llm_provider():
    return "Groq" if get_setting("GROQ_API_KEY") else "OpenAI" if get_setting("OPENAI_API_KEY") else None


@st.cache_data(ttl=3600, show_spinner=False)
def ask_llm(question: str, facts: str) -> str:
    messages = [
        {"role": "system", "content": "You are a retail data analyst. Answer only from the facts given. Be concise: 5 bullet points at most."},
        {"role": "user", "content": f"Facts about the dataset:\n{facts}\n\nQuestion: {question}"},
    ]
    if get_setting("GROQ_API_KEY"):
        from groq import Groq
        client, model = Groq(api_key=get_setting("GROQ_API_KEY")), get_setting("GROQ_MODEL") or "llama-3.3-70b-versatile"
    else:
        from openai import OpenAI
        client, model = OpenAI(api_key=get_setting("OPENAI_API_KEY")), get_setting("OPENAI_MODEL") or "gpt-4o-mini"
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=400)
    return resp.choices[0].message.content.strip()


def dataset_facts() -> list:
    facts = []
    monthly = full.set_index("Transaction_Date").resample("MS")["Quantity_Sold"].sum()
    if len(monthly) >= 6:
        first, last = monthly.iloc[:3].mean(), monthly.iloc[-3:].mean()
        facts.append(f"Items sold per month moved from {first:,.0f} (first 3 months) to {last:,.0f} (last 3 months), a {last / first - 1:+.0%} change.")
    promo = full.groupby("On_Promotion")["Quantity_Sold"].mean()
    if len(promo) == 2:
        facts.append(f"Average items per transaction: {promo[True]:.1f} on promotion vs {promo[False]:.1f} without ({promo[True] / promo[False] - 1:+.0%}).")
    for col in ("Location", "Income_Level", "Gender", "Category"):
        if col in full:
            g = full.groupby(col)["Total_Amount"].sum().sort_values(ascending=False)
            share = g / g.sum()
            facts.append(f"Revenue share by {col}: " + ", ".join(f"{k} {v:.0%}" for k, v in share.head(5).items()) + ".")
    new_cust = customer_data.set_index("First_Purchase_Date").resample("QS").size()
    if len(new_cust) >= 4:
        facts.append(f"New customers per quarter: first {new_cust.iloc[0]}, latest {new_cust.iloc[-1]}.")
    if "Sentiment" in full and pd.api.types.is_numeric_dtype(full["Sentiment"]):
        facts.append(f"Correlation between product sentiment and items sold: {full['Sentiment'].corr(full['Quantity_Sold']):+.2f}.")
    return facts


# ================================================================ UI
st.title("🛍️ Ambiscions Case Study: Test Pilot")
st.write(
    "A retail analytics walk-through on transaction, customer, product, store and promotion data: "
    "what happened (descriptive), why (diagnostic), what will happen (predictive) and what to do (prescriptive)."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Revenue", f"{full['Total_Amount'].sum():,.0f}")
k2.metric("Transactions", f"{len(full):,}")
k3.metric("Customers", f"{customer_data['Customer_ID'].nunique():,}")
k4.metric("Stores", f"{store_data['Store_ID'].nunique():,}")
k5.metric("Period", f"{full['Transaction_Date'].min():%b %Y} – {full['Transaction_Date'].max():%b %Y}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Descriptive", "2. Diagnostic", "3. Predictive", "4. Prescriptive", "About"])


# ---------------------------------------------------------------- 1. descriptive
@section_guard("descriptive")
def descriptive():
    c1, c2 = st.columns(2)
    by_loc = full.groupby("Location")["Total_Amount"].mean().reset_index()
    c1.plotly_chart(px.bar(by_loc, x="Location", y="Total_Amount", title="Average sale per store location",
                           labels={"Total_Amount": "Avg sale"}), width="stretch")
    monthly = full.set_index("Transaction_Date").resample("MS")["Quantity_Sold"].mean().rolling(2).mean().reset_index()
    c2.plotly_chart(px.line(monthly, x="Transaction_Date", y="Quantity_Sold", title="Average items per transaction (2-month moving average)",
                            labels={"Transaction_Date": "", "Quantity_Sold": "Items"}), width="stretch")
    c3, c4 = st.columns(2)
    by_age = full.groupby("Age")["Quantity_Sold"].mean().reset_index()
    c3.plotly_chart(px.line(by_age, x="Age", y="Quantity_Sold", title="Average basket size by age",
                            labels={"Quantity_Sold": "Items"}), width="stretch")
    young = (customer_data[customer_data["Age"] < 35].set_index("First_Purchase_Date").resample("MS").size().reset_index(name="New customers"))
    c4.plotly_chart(px.line(young, x="First_Purchase_Date", y="New customers", title="New customers under 35 per month",
                            labels={"First_Purchase_Date": ""}), width="stretch")
    if "Sentiment" in full:
        st.plotly_chart(px.violin(full, x=full["Sentiment"].astype(str), y="Quantity_Sold", box=True,
                                  title="Items sold per product sentiment", labels={"x": "Sentiment"}), width="stretch")


@section_guard("segmentation")
def segmentation():
    st.subheader("Customer segmentation (k-means)")
    k = st.slider("Number of clusters", 2, 8, 4)
    seg = segment_customers(k)
    options = ["Age", "Quantity_Sold", "Gender", "Income_Level", "First_Purchase_Date"]
    a, b = st.columns(2)
    x_var = a.selectbox("X axis", options, index=0)
    y_var = b.selectbox("Y axis", options, index=1)
    fig = px.scatter(seg, x=x_var, y=y_var, color="Cluster", opacity=0.7, category_orders={"Cluster": [str(i) for i in range(k)]})
    st.plotly_chart(fig, width="stretch")
    profile = seg.groupby("Cluster").agg(Customers=("Customer_ID", "count"), Avg_age=("Age", "mean"),
                                         Avg_items=("Quantity_Sold", "mean")).round(1)
    st.dataframe(profile, width="stretch")


def sentiment_demo():
    st.subheader("Review sentiment scorer")
    text = st.text_input("Type a customer review", "The staff were friendly but delivery was slow and late.")
    if text.strip():
        label, raw = sentiment_score(text)
        st.metric("Sentiment score", {1: "+1 positive", 0: "0 neutral", -1: "-1 negative"}[label], f"raw {raw:+d}")
        st.caption("A small keyword lexicon with negation handling, producing the same -1 / 0 / +1 scale as the product Sentiment column.")


with tab1:
    descriptive()
    segmentation()
    sentiment_demo()


# ---------------------------------------------------------------- 2. diagnostic
@section_guard("discounts")
def discounts():
    st.subheader("Items sold per month, with promotion periods")
    monthly = full.set_index("Transaction_Date").resample("MS")["Quantity_Sold"].sum().reset_index()
    fig = go.Figure(go.Scatter(x=monthly["Transaction_Date"], y=monthly["Quantity_Sold"], mode="lines+markers", name="Items sold"))
    for _, p in promotion_data.dropna(subset=["Start_Date", "End_Date"]).iterrows():
        fig.add_vrect(x0=p["Start_Date"], x1=p["End_Date"], fillcolor="red", opacity=0.15, line_width=0)
    fig.update_layout(height=380, yaxis_title="Items sold", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    promo = full.groupby("On_Promotion")["Quantity_Sold"].mean()
    if len(promo) == 2:
        a, b, c = st.columns(3)
        a.metric("Avg items, no promotion", f"{promo[False]:.1f}")
        b.metric("Avg items, on promotion", f"{promo[True]:.1f}", f"{promo[True] / promo[False] - 1:+.1%}")
        c.metric("Share of transactions on promotion", f"{full['On_Promotion'].mean():.0%}")


@section_guard("correlation")
def correlation():
    st.subheader("Correlation between variables")
    numeric = [c for c in full.select_dtypes(include=[np.number]).columns if not c.endswith("_ID")]
    default = [c for c in ["Quantity_Sold", "Price_Per_Unit", "Discount", "Age", "Sentiment", "Total_Amount"] if c in numeric]
    chosen = st.multiselect("Variables", numeric, default=default or numeric[:5])
    if len(chosen) < 2:
        st.info("Pick at least two numeric variables.")
        return
    corr = full[chosen].corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), width="stretch")


@section_guard("AI analyst")
def ai_analyst():
    st.subheader("AI analyst")
    facts = dataset_facts()
    question = st.text_area("Ask a question about the data", "What factors are contributing to the sales trend?")
    provider = llm_provider()
    if not provider:
        st.info("No LLM key configured (GROQ_API_KEY or OPENAI_API_KEY in secrets), so the facts below are computed directly from the data.")
    if provider and st.button("Analyze", type="primary"):
        if question.strip():
            try:
                with st.spinner(f"Asking {provider}…"):
                    st.markdown(ask_llm(question.strip()[:1000], "\n".join(facts)))
                st.caption(f"Answer generated by {provider} from the computed facts below.")
            except Exception as exc:
                st.error(f"The LLM could not be reached ({type(exc).__name__}). Showing the computed facts instead.")
    with st.expander("Computed facts", expanded=not provider):
        for f in facts:
            st.markdown(f"- {f}")


with tab2:
    discounts()
    correlation()
    ai_analyst()


# ---------------------------------------------------------------- 3. predictive
@section_guard("predictive")
def predictive():
    st.subheader("Predicting items per transaction (XGBoost)")
    metrics, importance = train_predictive()
    a, b, c = st.columns(3)
    a.metric("RMSE (validation)", f"{metrics['rmse']:.2f}")
    b.metric("Baseline RMSE (predict the mean)", f"{metrics['baseline']:.2f}",
             f"{metrics['rmse'] / metrics['baseline'] - 1:+.0%}", delta_color="inverse")
    c.metric("R²", f"{metrics['r2']:.2f}")
    top = importance.head(15).sort_values().reset_index()
    top.columns = ["Feature", "Importance"]
    st.plotly_chart(px.bar(top, x="Importance", y="Feature", orientation="h", title="Top 15 features (gain-based importance)"),
                    width="stretch")
    st.caption("80/20 train/validation split. ID columns and Total_Amount (which contains the target) are excluded.")


with tab3:
    predictive()


# ---------------------------------------------------------------- 4. prescriptive
@section_guard("prescriptive")
def prescriptive():
    st.subheader("What-if forecast for the next six months")
    model, columns = train_prescriptive()
    a, b, c = st.columns(3)
    age = a.slider("Average customer age", int(full["Age"].min()), int(full["Age"].max()), int(full["Age"].mean()))
    discount = b.slider("Average discount", float(full["Discount"].min()), float(max(full["Discount"].max(), 0.01)),
                        float(full["Discount"].mean()), 0.01)
    sentiment = c.slider("Average product sentiment", -1.0, 1.0, 0.0, 0.1, disabled="Sentiment" not in columns)
    d, e = st.columns(2)
    gender = d.selectbox("Customer gender", sorted(full["Gender"].dropna().astype(str).unique()))
    income = e.selectbox("Customer income bracket", sorted(full["Income_Level"].dropna().astype(str).unique()))

    dates = pd.date_range(start=full["Transaction_Date"].max().normalize(), periods=183, freq="D")[1:]
    future = pd.DataFrame({"Age": age, "Discount": discount, "Sentiment": sentiment,
                           "Transaction_Year": dates.year, "Transaction_Month": dates.month}, index=dates)
    future[f"Gender_{gender}"] = 1
    future[f"Income_Level_{income}"] = 1
    future = future.reindex(columns=columns, fill_value=0).astype(float)
    pred = model.predict(future)

    baseline_in = future.copy()
    if "Discount" in baseline_in:
        baseline_in["Discount"] = 0.0
    base = model.predict(baseline_in)

    m1, m2 = st.columns(2)
    m1.metric("Predicted avg items per transaction", f"{pred.mean():.2f}")
    m2.metric("Uplift vs. no discount", f"{pred.mean() - base.mean():+.2f}", f"{pred.mean() / base.mean() - 1:+.1%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=pred, mode="lines", name="With chosen settings"))
    fig.add_trace(go.Scatter(x=dates, y=base, mode="lines", name="Same, without discount", line=dict(dash="dot")))
    fig.update_layout(title="Predicted items per transaction, next six months", xaxis_title="Date", yaxis_title="Items", height=400)
    st.plotly_chart(fig, width="stretch")
    st.caption("Tree models cannot extrapolate trends beyond the training period, so the forecast changes mainly with month and the chosen settings.")


with tab4:
    prescriptive()


# ---------------------------------------------------------------- about
with tab5:
    left, right = st.columns([2, 1])
    with left:
        st.markdown(
            """
**How it works**

- **Data:** five CSV tables loaded from Google Drive and cached for 24 hours. Quantities are capped at 500 to limit outliers.
- **Descriptive:** averages by location, age and month, plus k-means segmentation on age, items bought, gender and income.
- **Diagnostic:** monthly sales against promotion windows, a correlation matrix, and an AI analyst. The analyst sends only
  aggregated facts (never raw rows) to an LLM when a key is configured, and otherwise shows the facts directly.
- **Predictive:** an XGBoost regressor for items per transaction, compared with a predict-the-mean baseline.
- **Prescriptive:** a second XGBoost model drives a what-if forecast from your chosen age, discount, sentiment and segment.
"""
        )
    with right:
        try:
            import qrcode

            img = qrcode.make(DEPLOYED_URL).resize((180, 180))
            st.image(img, caption="Open this app on your phone")
        except Exception:
            st.caption(DEPLOYED_URL)
