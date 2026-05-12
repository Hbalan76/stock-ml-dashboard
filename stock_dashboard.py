import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stock ML Dashboard", layout="wide")

st.title("📈 Stock ML Dashboard")
st.warning("⚠️ Demo model. Not for real trading.")

# =============================
# SIDEBAR
# =============================
# =============================
# NIFTY 50 STOCK LIST
# =============================
nifty50_stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","WIPRO.NS","HCLTECH.NS",
    "POWERGRID.NS","NTPC.NS","TATAMOTORS.NS","JSWSTEEL.NS","ONGC.NS",
    "COALINDIA.NS","BAJFINANCE.NS","BAJAJFINSV.NS","INDUSINDBK.NS","ADANIENT.NS",
    "ADANIPORTS.NS","GRASIM.NS","DRREDDY.NS","CIPLA.NS","EICHERMOT.NS",
    "HEROMOTOCO.NS","DIVISLAB.NS","BRITANNIA.NS","APOLLOHOSP.NS","UPL.NS",
    "SBILIFE.NS","HDFCLIFE.NS","BAJAJ-AUTO.NS","TATASTEEL.NS","TECHM.NS",
    "SHREECEM.NS","BPCL.NS","HINDALCO.NS","IOC.NS","M&M.NS"
]

# =============================
# SIDEBAR INPUT
# =============================

mode = st.sidebar.radio("Select Mode", ["NIFTY 50", "Custom"])

if mode == "NIFTY 50":
    ticker = st.sidebar.selectbox("Select NIFTY 50 Stock", nifty50_stocks)
else:
    ticker = st.sidebar.text_input("Enter Custom Ticker", "RELIANCE.NS")

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))

run_button = st.sidebar.button("🚀 Run Model")

threshold = 0.5

# =============================
# FUNCTIONS
# =============================

def prepare_data(df):
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], 10)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], 50)

    df['RSI'] = ta.momentum.rsi(df['Close'], 14)
    df['MACD'] = ta.trend.macd(df['Close'])

    df['Returns'] = df['Close'].pct_change()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)
    return df


def train_model(df):
    features = ['SMA_10','SMA_50','RSI','MACD','Returns']

    X = df[features]
    y = df['Target']

    split = int(len(df)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, df.iloc[split:], features


def backtest(df, proba, threshold):

    df = df.copy()

    # BUY when strong probability
    df['Signal'] = 0
    df.loc[proba > threshold, 'Signal'] = 1
    df.loc[proba < (1 - threshold), 'Signal'] = -1

    # Position (carry forward)
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
    df['Position'].fillna(0, inplace=True)

    # Returns
    df['Strategy_Return'] = df['Position'].shift(1) * df['Close'].pct_change()
    df['Market_Return'] = df['Close'].pct_change()

    df['Strategy_Cum'] = (1 + df['Strategy_Return']).cumprod()
    df['Market_Cum'] = (1 + df['Market_Return']).cumprod()

    # Signals for chart
    df['Buy'] = np.where((df['Position'] == 1) & (df['Position'].shift(1) != 1), df['Close'], np.nan)
    df['Sell'] = np.where((df['Position'] == -1) & (df['Position'].shift(1) != -1), df['Close'], np.nan)

    return df


# =============================
# RUN
# =============================
if run_button:

    df = yf.download(ticker, start="2015-01-01")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open','High','Low','Close','Volume']].copy()
    df.dropna(inplace=True)

    df = prepare_data(df)

    model, X_test, y_test, df_test, features = train_model(df)

    proba = model.predict_proba(X_test)[:,1]

    predictions = (proba > threshold).astype(int)

    acc = accuracy_score(y_test, predictions)

    result = backtest(df_test, proba, threshold)

    # =============================
    # METRICS
    # =============================
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Strategy", f"{result['Strategy_Cum'].iloc[-1]:.2f}x")
    col3.metric("Market", f"{result['Market_Cum'].iloc[-1]:.2f}x")

    # =============================
    # TABS
    # =============================
    tab1, tab2, tab3 = st.tabs(["Overview", "Model", "Backtest"])

    # =============================
    # OVERVIEW
    # =============================
    with tab1:

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=result.index,
            open=result['Open'],
            high=result['High'],
            low=result['Low'],
            close=result['Close']
        ))

        fig.add_trace(go.Scatter(
            x=result.index,
            y=result['Buy'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10),
            name='Buy'
        ))

        fig.add_trace(go.Scatter(
            x=result.index,
            y=result['Sell'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10),
            name='Sell'
        ))

        st.plotly_chart(fig, use_container_width=True)

    # =============================
    # MODEL
    # =============================
    with tab2:

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        st.bar_chart(imp_df.set_index('Feature'))

    # =============================
    # BACKTEST
    # =============================
    with tab3:

        fig2, ax = plt.subplots()

        ax.plot(result['Strategy_Cum'], label='Strategy')
        ax.plot(result['Market_Cum'], label='Market')
        ax.legend()

        st.pyplot(fig2)