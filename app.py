#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

st.set_page_config(page_title="Stock Price Forecasting App")
st.title("📈 Stock Price Forecasting")

# Input stock ticker
symbol = st.text_input("Enter Stock Ticker (e.g., TCS.NS):", "TCS.NS")

start_date = "2020-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
forecast_days = 30

# Download and cache data
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df[['Close']].dropna()
    return df

data = load_data(symbol)

# Button 1: Show line chart
if st.button("📊 Show Historical Line Chart"):
    st.line_chart(data['Close'])

# Helper to plot forecast
def plot_forecast(data, forecast_values, title):
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    forecast_df = pd.DataFrame({'Forecast': forecast_values}, index=forecast_index)
    full_df = pd.concat([data['Close'], forecast_df['Forecast']])

    st.line_chart(full_df)
    st.write("Forecast Table:")
    st.dataframe(forecast_df)

# Button 2: Linear Regression
if st.button("📉 Linear Regression Forecast"):
    try:
        model = joblib.load("linear_model.pkl")
        scaler = joblib.load("linear_scaler.pkl")

        df = data.copy()
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df.dropna(inplace=True)

        X = df[['Close', 'MA_7', 'MA_21']]
        X_scaled = scaler.transform(X)

        last_values = X_scaled[-1].reshape(1, -1)
        forecast_values = []
        for _ in range(forecast_days):
            pred = model.predict(last_values)[0]
            forecast_values.append(pred)
            last_values = np.roll(last_values, -1)
            last_values[0, -1] = pred  # shift in new forecast

        plot_forecast(df, forecast_values, "Linear Regression Forecast")
    except Exception as e:
        st.error(f"Linear Regression failed: {e}")

# Button 3: Holt's model
if st.button("📈 Holt's Forecast"):
    try:
        model = joblib.load("holt_model.pkl")
        forecast_values = model.forecast(forecast_days)
        plot_forecast(data, forecast_values, "Holt's Model Forecast")
    except Exception as e:
        st.error(f"Holt's Model failed: {e}")

# Button 4: ARIMA model
if st.button("🧮 ARIMA Forecast"):
    try:
        model = joblib.load("arima_model.pkl")
        forecast_values = model.predict(n_periods=forecast_days)
        plot_forecast(data, forecast_values, "ARIMA Forecast")
    except Exception as e:
        st.error(f"ARIMA failed: {e}")

# Button 5: LSTM model
if st.button("🤖 LSTM Forecast"):
    try:
        model = load_model("lstm_model.h5")
        scaler = joblib.load("lstm_scaler.pkl")

        seq_len = 60
        close_scaled = scaler.transform(data[['Close']])
        last_seq = close_scaled[-seq_len:].reshape(1, seq_len, 1)

        predictions = []
        for _ in range(forecast_days):
            pred = model.predict(last_seq)[0][0]
            predictions.append(pred)
            last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

        forecast_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        plot_forecast(data, forecast_values, "LSTM Forecast")
    except Exception as e:
        st.error(f"LSTM failed: {e}")

