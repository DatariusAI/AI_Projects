import datetime as dt
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# --- SETUP ---
st.set_page_config(page_title="Equity Insights", page_icon="📈", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("Settings")
ticker_a = st.sidebar.text_input("First ticker", "AAPL").strip().upper()
ticker_b = st.sidebar.text_input("Second ticker", "MSFT").strip().upper()
start = st.sidebar.date_input("Start date", dt.date(2023, 1, 1))
end = st.sidebar.date_input("End date", dt.date(2024, 1, 1))
forecast_days = st.sidebar.slider("ARIMA forecast horizon (trading days)", 5, 30, 10)

# --- HEADER ---
st.title(f"📈 {ticker_a} vs {ticker_b} Dashboard (Power BI Style)")
st.markdown(
    "Interactive analysis for investment banking decision support: performance, risk, co-movement, "
    "a machine-learning test of next-day predictability and a short ARIMA price forecast. "
    "Prices come live from Yahoo Finance."
)

with st.expander("How it works"):
    st.markdown(
        """
- **Data:** daily adjusted close prices from Yahoo Finance (`yfinance`), cached for 6 hours.
- **Risk metrics:** annualised return and volatility use 252 trading days. Sharpe ratio assumes a 0% risk-free rate.
  Max drawdown is the worst peak-to-trough fall.
- **Random Forest:** predicts whether the first stock rises tomorrow from both stocks' last two daily returns.
  It trains on the first 70% of days and is tested on the last 30% (time-ordered, no shuffling, so no look-ahead).
  It is compared with always guessing the more common outcome.
- **ARIMA(1,1,1):** a classic `statsmodels` time-series model fitted on the first stock's price, with a 95% interval.
"""
    )

if not ticker_a or not ticker_b or ticker_a == ticker_b:
    st.error("Enter two different ticker symbols in the sidebar.")
    st.stop()
if start >= end:
    st.error("The start date must be before the end date.")
    st.stop()

# --- DATA ---
tickers = [ticker_a, ticker_b]


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices from Yahoo Finance…")
def load_prices(symbols, start, end):
    data = yf.download(list(symbols), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data.columns.name = None
    return data.dropna()


try:
    df = load_prices(tuple(tickers), start, end)
except Exception as exc:
    df = pd.DataFrame()
    st.caption(f"Download error: {exc}")
if df.empty or len(df) < 30 or not set(tickers).issubset(df.columns):
    st.error(
        "Yahoo Finance did not return enough price data just now. Check the tickers and dates, "
        "or refresh in a minute."
    )
    load_prices.clear()
    st.stop()
df = df[tickers]

# --- KPIs ---
returns = df.pct_change().dropna()


def kpis(price, ret):
    total = price.iloc[-1] / price.iloc[0] - 1
    vol = ret.std() * np.sqrt(252)
    ann = (1 + total) ** (252 / len(ret)) - 1
    drawdown = (price / price.cummax() - 1).min()
    return {"total": total, "vol": vol, "sharpe": ann / vol if vol else np.nan, "mdd": drawdown}


stats = {t: kpis(df[t], returns[t]) for t in tickers}
for t, col in zip(tickers, st.columns(2)):
    with col:
        st.markdown(f"#### {t}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total return", f"{stats[t]['total']:.1%}")
        k2.metric("Volatility (ann.)", f"{stats[t]['vol']:.1%}")
        k3.metric("Sharpe", f"{stats[t]['sharpe']:.2f}")
        k4.metric("Max drawdown", f"{stats[t]['mdd']:.1%}")

# --- LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price comparison")
    view = st.radio("Show", ["Rebased to 100", "Adjusted close (USD)"], horizontal=True, label_visibility="collapsed")
    plot_df = df / df.iloc[0] * 100 if view == "Rebased to 100" else df
    fig = px.line(plot_df, labels={"value": "Index" if view == "Rebased to 100" else "USD", "variable": "Ticker"},
                  template="plotly_white")
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("Correlation")
    price_corr = df.corr().iloc[0, 1]
    ret_corr = returns.corr().iloc[0, 1]
    c1, c2 = st.columns(2)
    c1.metric("Price correlation", f"{price_corr:.2f}")
    c2.metric("Daily return correlation", f"{ret_corr:.2f}")
    rolling = returns[ticker_a].rolling(30).corr(returns[ticker_b]).dropna()
    fig_rc = px.line(rolling, labels={"value": "Correlation", "index": "Date"}, template="plotly_white",
                     title="30-day rolling correlation of daily returns")
    fig_rc.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_rc, width="stretch")

# --- FEATURE ENGINEERING ---
feat = pd.DataFrame(index=returns.index)
for t in tickers:
    feat[f"{t}_lag1"] = returns[t]
    feat[f"{t}_lag2"] = returns[t].shift(1)
feat["target"] = (returns[ticker_a].shift(-1) > 0).astype(int)
feat = feat.iloc[:-1].dropna()

# --- RANDOM FOREST ---
st.markdown("---")
st.subheader("Random Forest model insights")

features = [c for c in feat.columns if c != "target"]
split = int(len(feat) * 0.7)
train, test = feat.iloc[:split], feat.iloc[split:]
model = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42)
model.fit(train[features], train["target"])
accuracy = (model.predict(test[features]) == test["target"]).mean()
baseline = max(test["target"].mean(), 1 - test["target"].mean())

m1, m2, m3 = st.columns(3)
m1.metric("Test accuracy", f"{accuracy:.1%}")
m2.metric("Naive baseline", f"{baseline:.1%}", help="Always predicting the more common outcome in the test period.")
m3.metric("Edge over baseline", f"{accuracy - baseline:+.1%}")

importances = pd.Series(model.feature_importances_, index=features).sort_values()
fig_feat = px.bar(importances, orientation="h", title=f"Feature importance for predicting {ticker_a}'s next-day direction",
                  labels={"value": "Importance", "index": ""}, template="plotly_white")
fig_feat.update_layout(showlegend=False)
st.plotly_chart(fig_feat, width="stretch")

# --- ARIMA FORECAST ---
st.markdown("---")
st.subheader(f"ARIMA forecast for {ticker_a}")
arima_text = "ARIMA could not be fitted on this period."
try:
    from statsmodels.tsa.arima.model import ARIMA

    series = df[ticker_a].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(series, order=(1, 1, 1)).fit()
    fc = fitted.get_forecast(forecast_days)
    mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
    future_idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    fig_fc = go.Figure()
    hist = df[ticker_a].iloc[-120:]
    fig_fc.add_scatter(x=hist.index, y=hist.values, name="History", line=dict(color="#1f2c56"))
    fig_fc.add_scatter(x=future_idx, y=ci.iloc[:, 1].values, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig_fc.add_scatter(x=future_idx, y=ci.iloc[:, 0].values, fill="tonexty", line=dict(width=0),
                       fillcolor="rgba(76,139,245,0.2)", name="95% interval")
    fig_fc.add_scatter(x=future_idx, y=mean.values, name="Forecast", line=dict(color="#4C8BF5", dash="dash"))
    fig_fc.update_layout(template="plotly_white", yaxis_title="USD", height=380)
    st.plotly_chart(fig_fc, width="stretch")
    change = mean.iloc[-1] / df[ticker_a].iloc[-1] - 1
    width = (ci.iloc[-1, 1] - ci.iloc[-1, 0]) / df[ticker_a].iloc[-1]
    arima_text = f"{forecast_days}-day forecast {change:+.1%}, 95% band about ±{width / 2:.0%}"
except Exception as exc:
    st.warning(f"ARIMA forecast unavailable: {exc}")

# --- INVESTMENT BANKING SUMMARY ---
st.markdown("---")
st.subheader("📊 Executive summary table")

leader, laggard = sorted(tickers, key=lambda t: stats[t]["total"], reverse=True)
calmer = min(tickers, key=lambda t: stats[t]["vol"])
top_feature = importances.index[-1]
edge = accuracy - baseline

rows = [
    ("Price trend",
     f"{leader} returned {stats[leader]['total']:.1%} vs {stats[laggard]['total']:.1%} for {laggard}",
     f"Overweight {leader} on momentum; review {laggard} for catch-up potential"),
    ("Risk",
     f"{calmer} was less volatile ({stats[calmer]['vol']:.1%} annualised)",
     f"Use {calmer} as the lower-risk core holding"),
    ("Correlation",
     f"Prices correlate at {price_corr:.2f}, daily returns at {ret_corr:.2f}",
     "Pair trading needs high return correlation" if ret_corr < 0.6 else "Returns move together; pair trading is viable"),
    ("ML insights",
     f"Random Forest {accuracy:.0%} vs {baseline:.0%} baseline; top feature {top_feature}",
     "Weak daily signal; explore intraday or multi-factor models" if edge < 0.03 else "Some edge; validate on more history before use"),
    ("ARIMA forecast", arima_text, "Consider covered calls or low-volatility strategies if the band is narrow"),
]

fig_table = go.Figure(data=[go.Table(
    columnwidth=[60, 240, 240],
    header=dict(values=["Area", "Key insight", "Suggested action"], fill_color="#1f2c56",
                font=dict(color="white", size=14), align="left"),
    cells=dict(values=[list(c) for c in zip(*rows)], fill_color=[["#f9f9f9", "#ffffff"] * 3],
               font=dict(color="black", size=13), align="left", height=30),
)])
fig_table.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=300)
st.plotly_chart(fig_table, width="stretch")
st.caption("Insights are generated from the data above. Educational analysis, not investment advice.")
