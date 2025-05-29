#python -m streamlit run pro.py

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from yahooquery import Ticker


st.title(" Backtesting Trading Strategies with Benchmark Comparison")

# User Inputs
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.date_input("End Date", value=datetime.today())

strategy = st.selectbox(
    "Choose Trading Strategy",
    ["Moving Average Crossover", "RSI Strategy", "Bollinger Bands", "MACD Crossover"]
)

# Fetch Data Function (cached)
@st.cache_data(show_spinner=False)
def get_data(symbol, start, end):
    ticker = Ticker(symbol)
    df = ticker.history(start=start, end=end)
    df = df.reset_index().set_index("date")
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    return df

# Fetch benchmark data
@st.cache_data(show_spinner=False)
def get_benchmark_data(symbol, start, end):
    ticker = Ticker(symbol)
    df = ticker.history(start=start, end=end)
    df = df.reset_index().set_index("date")
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    return df[["close"]].copy()

try:
    data = get_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if data.empty:
        st.error("No data available. Please check the symbol and date range.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

data = data[["close"]].copy()
data.dropna(inplace=True)

# Fetch benchmarks: Nifty 50 (^NSEI) and S&P 500 (^GSPC)
try:
    nifty_data = get_benchmark_data("^NSEI", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    sp500_data = get_benchmark_data("^GSPC", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
except Exception as e:
    st.error(f"Error fetching benchmark data: {e}")
    st.stop()

nifty_close = nifty_data["close"]
sp500_close = sp500_data["close"]

# Align data indexes
combined_df = pd.DataFrame({
    symbol: data["close"],
    "Nifty 50": nifty_close,
    "S&P 500": sp500_close
}).dropna()

# Strategy Implementation
if strategy == "Moving Average Crossover":
    st.subheader(" Moving Average Crossover")
    short = st.slider("Short MA", 5, 50, 10)
    long = st.slider("Long MA", 10, 200, 50)

    data["MA_Short"] = data["close"].rolling(short).mean()
    data["MA_Long"] = data["close"].rolling(long).mean()
    data["Signal"] = (data["MA_Short"] > data["MA_Long"]).astype(int)

elif strategy == "RSI Strategy":
    st.subheader(" RSI Strategy")
    period = st.slider("RSI Period", 5, 30, 14)
    rsi_buy = st.slider("Buy Threshold", 0, 50, 30)
    rsi_sell = st.slider("Sell Threshold", 50, 100, 70)

    delta = data["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=data.index).rolling(period).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(period).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data["Signal"] = ((data["RSI"] < rsi_buy)).astype(int)

elif strategy == "Bollinger Bands":
    st.subheader(" Bollinger Bands")
    period = st.slider("BB Period", 10, 30, 20)
    stddev = st.slider("Standard Deviation", 1, 3, 2)

    data["MA"] = data["close"].rolling(period).mean()
    data["Upper"] = data["MA"] + stddev * data["close"].rolling(period).std()
    data["Lower"] = data["MA"] - stddev * data["close"].rolling(period).std()
    data["Signal"] = (data["close"] < data["Lower"]).astype(int)

elif strategy == "MACD Crossover":
    st.subheader(" MACD Crossover")
    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()

    data["MACD"] = macd
    data["Signal_Line"] = signal_line
    data["Signal"] = (data["MACD"] > data["Signal_Line"]).astype(int)

# Backtest Logic
data["Signal"].fillna(0, inplace=True)
data["Daily Return"] = data["close"].pct_change()
data["Strategy Return"] = data["Signal"].shift(1) * data["Daily Return"]

data.dropna(inplace=True)

# Performance Metrics
cumulative_market = (1 + data["Daily Return"]).cumprod()
cumulative_strategy = (1 + data["Strategy Return"]).cumprod()

sharpe = (
    data["Strategy Return"].mean() / data["Strategy Return"].std()
    * np.sqrt(252) if data["Strategy Return"].std() != 0 else 0
)

st.metric("📈 Total Strategy Return", f"{(cumulative_strategy.iloc[-1] - 1)*100:.2f}%")
st.metric("📉 Total Market Return", f"{(cumulative_market.iloc[-1] - 1)*100:.2f}%")
st.metric("📊 Sharpe Ratio", f"{sharpe:.2f}")

# Buy/Sell Signals Plot
data["Position Change"] = data["Signal"].diff()
buy = data[data["Position Change"] == 1]
sell = data[data["Position Change"] == -1]

fig_signals = go.Figure()
fig_signals.add_trace(go.Scatter(x=data.index, y=data["close"], name="Close"))
fig_signals.add_trace(go.Scatter(x=buy.index, y=buy["close"], name="Buy", mode='markers',
                                 marker=dict(color='green', symbol='triangle-up', size=10)))
fig_signals.add_trace(go.Scatter(x=sell.index, y=sell["close"], name="Sell", mode='markers',
                                 marker=dict(color='red', symbol='triangle-down', size=10)))
fig_signals.update_layout(title="Buy/Sell Signals", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_signals)

# Growth Chart including Benchmarks
fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(x=combined_df.index, y=100 * (1 + combined_df[symbol].pct_change().fillna(0)).cumprod(), name=symbol))
fig_growth.add_trace(go.Scatter(x=combined_df.index, y=100 * (1 + combined_df["Nifty 50"].pct_change().fillna(0)).cumprod(), name="Nifty 50"))
fig_growth.add_trace(go.Scatter(x=combined_df.index, y=100 * (1 + combined_df["S&P 500"].pct_change().fillna(0)).cumprod(), name="S&P 500"))
fig_growth.add_trace(go.Scatter(x=data.index, y=100 * cumulative_strategy, name="Strategy"))
fig_growth.update_layout(title="Investment Growth ($100)", xaxis_title="Date", yaxis_title="Value")
st.plotly_chart(fig_growth)


# Value at Risk (VaR)
log_ret = np.log(data["close"] / data["close"].shift(1)).dropna()
mu, sigma = log_ret.mean(), log_ret.std()


fig_var = go.Figure()
fig_var.add_trace(go.Histogram(x=log_ret, nbinsx=100, histnorm='probability density', opacity=0.6, name='Log Returns'))
x_vals = np.linspace(log_ret.min(), log_ret.max(), 500)
fig_var.add_trace(go.Scatter(x=x_vals, y=norm.pdf(x_vals, mu, sigma), name='Normal Curve'))
fig_var.update_layout(title="Distribution of Log Returns", xaxis_title="Log Return", yaxis_title="Density")
st.plotly_chart(fig_var)

var_95 = norm.ppf(0.05, mu, sigma)
var_99 = norm.ppf(0.01, mu, sigma)

st.subheader(" Value at Risk (VaR)")
st.write(f"95% 1-day VaR: {var_95:.4f} ({var_95 * 100:.2f}%)")
st.write(f"99% 1-day VaR: {var_99:.4f} ({var_99 * 100:.2f}%)")


# Correlation Heatmap of Returns
st.subheader("Correlation Heatmap of Daily Returns")
returns_df = combined_df.pct_change().dropna()
corr = returns_df.corr()

fig_corr = px.imshow(corr,
                     text_auto=True,
                     color_continuous_scale='RdBu_r',
                     origin='lower',
                     title="Correlation Heatmap of Daily Returns")
st.plotly_chart(fig_corr)

# Volatility Chart (Rolling Std Dev of Returns)
st.subheader("Volatility Chart (Rolling Std Dev of Returns)")
window_vol = st.slider("Volatility Rolling Window (days)", 5, 60, 20)
volatility = returns_df.rolling(window=window_vol).std()

fig_vol = go.Figure()
for col in volatility.columns:
    fig_vol.add_trace(go.Scatter(x=volatility.index, y=volatility[col], mode='lines', name=f'Volatility: {col}'))

fig_vol.update_layout(title=f"Rolling {window_vol}-Day Volatility of Returns",
                      xaxis_title="Date", yaxis_title="Volatility (Std Dev)",
                      template="plotly_white")
st.plotly_chart(fig_vol)

# ---------------------------
# Descriptive Statistics
# ---------------------------
st.subheader(" Descriptive Analysis")

# Summary stats
st.write("### Statistical Summary of Closing Prices")
st.dataframe(data["close"].describe().rename("Closing Price Stats"))

# Histogram of close prices
st.write("### Histogram of Closing Prices")
fig_hist = px.histogram(data, x="close", nbins=50, title="Distribution of Closing Prices")
st.plotly_chart(fig_hist)

# Rolling metrics
st.write("### Rolling Mean and Standard Deviation")
window_size = st.slider("Select Rolling Window Size (days)", 5, 60, 20)

data["Rolling Mean"] = data["close"].rolling(window_size).mean()
data["Rolling Std"] = data["close"].rolling(window_size).std()

fig_rolling = go.Figure()
fig_rolling.add_trace(go.Scatter(x=data.index, y=data["close"], name="Close", line=dict(color='blue')))
fig_rolling.add_trace(go.Scatter(x=data.index, y=data["Rolling Mean"], name="Rolling Mean", line=dict(color='orange')))
fig_rolling.add_trace(go.Scatter(x=data.index, y=data["Rolling Std"], name="Rolling Std Dev", line=dict(color='green')))
fig_rolling.update_layout(title="Rolling Mean and Standard Deviation",
                          xaxis_title="Date", yaxis_title="Price / Std Dev")
st.plotly_chart(fig_rolling)

#new



