import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- SETUP ---
st.set_page_config(page_title="Equity Insights", layout="wide")

# --- HEADER ---
st.title("📈 AAPL vs MSFT Dashboard (Power BI Style)")
st.markdown("Interactive analysis for investment banking decision support")

# --- DATA ---
tickers = ['AAPL', 'MSFT']
df = yf.download(tickers, start='2023-01-01', end='2024-01-01', auto_adjust=True)['Close']
df.columns.name = None
df.dropna(inplace=True)

# --- FEATURE ENGINEERING ---
returns = df.pct_change().dropna()
returns['AAPL_lag1'] = returns['AAPL'].shift(1)
returns['MSFT_lag1'] = returns['MSFT'].shift(1)
returns['target'] = (returns['AAPL'].shift(-1) > 0).astype(int)
returns.dropna(inplace=True)

# --- LAYOUT ---
col1, col2 = st.columns(2)

# --- TIME SERIES VISUAL ---
with col1:
    st.subheader("Price Comparison")
    fig = px.line(df, title="Adjusted Close Prices", labels={"value": "USD"}, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- CORRELATION ---
with col2:
    st.subheader("Correlation Matrix")
    corr_fig = px.imshow(df.corr(), text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(corr_fig, use_container_width=True)

# --- FEATURE IMPORTANCE ---
st.markdown("---")
st.subheader("Random Forest Model Insights")

features = ['AAPL_lag1', 'MSFT_lag1']
X = returns[features]
y = returns['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier().fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=features).sort_values()
fig_feat = px.bar(importances, orientation="h", title="Feature Importance", template="plotly_white")
st.plotly_chart(fig_feat, use_container_width=True)

# --- INVESTMENT BANKING SUMMARY ---
st.markdown("---")
st.subheader("📊 Executive Summary Table")

fig_table = go.Figure(data=[go.Table(
    columnwidth=[60, 240, 240],
    header=dict(
        values=["Area", "Key Insight", "Suggested Action"],
        fill_color="#1f2c56",
        font=dict(color="white", size=14),
        align="left"
    ),
    cells=dict(
        values=[
            ["Price Trend", "Correlation", "ML Insights", "ARIMA Forecast", "Overall Strategy"],
            [
                "MSFT outperformed AAPL in 2023",
                "Strong price correlation, weak return prediction",
                "AAPL's own lag best explains movement",
                "Stable short-term forecast for AAPL",
                "Weak predictive signal in linear models"
            ],
            [
                "Overweight MSFT, consider pair trading",
                "Avoid return-based pair trading",
                "Develop intra-day or swing trading models",
                "Use covered calls or low-volatility strategies",
                "Explore non-linear, multi-factor ML approaches"
            ]
        ],
        fill_color=[['#f9f9f9', '#ffffff']*3],
        font=dict(color='black', size=13),
        align="left"
    )
)])
fig_table.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=400)
st.plotly_chart(fig_table, use_container_width=True)
