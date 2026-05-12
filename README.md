# 📈 Stock ML Dashboard

A Streamlit-based dashboard for analyzing NIFTY 50 and custom stocks using machine learning models and technical indicators.

## 🚀 Features
- Interactive dashboard built with **Streamlit**
- Fetches stock data using **Yahoo Finance (yfinance)**
- Calculates technical indicators:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - MACD
- Trains a **Random Forest Classifier** to predict next-day price movement
- Backtesting strategy with cumulative returns vs market
- Visualizations:
  - Candlestick charts with Buy/Sell signals (Plotly)
  - Feature importance bar chart
  - Strategy vs Market performance (Matplotlib)

## 📦 Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
