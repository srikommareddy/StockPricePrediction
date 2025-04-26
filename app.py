#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

st.title("📈 Stock Forecasting App")

# 1. Input stock code
stock_code = st.text_input("Enter Stock Symbol (e.g., TCS.NS):", "TCS.NS")

start_date = "2020-01-01"
end_date = date.today() - timedelta(days=1)
forecast_days = 30

@st.cache_data(show_spinner=False)
def load_data(symbol):
    df = yf.download(symbol, start=start_date, end=end_date)
    return df[['Close']].dropna()

@st.cache_data(show_spinner=False)
def prepare_lstm_data(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    return np.array(X), np.array(y), scaler

@st.cache_resource(show_spinner=False)
def train_lstm_model(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

# Load and show data
if st.button("Show Line Chart"):
    data = load_data(stock_code)
    st.line_chart(data['Close'])

# Forecasting functions
def forecast_linear_regression(data):
    df = data.reset_index()
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']].values
    y = df['Close'].values
    model = LinearRegression().fit(X, y)
    future_days = np.arange(X[-1][0] + 1, X[-1][0] + forecast_days + 1).reshape(-1, 1)
    future_preds = model.predict(future_days)
    forecast_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')
    return forecast_dates, future_preds

def forecast_holt(data):
    model = ExponentialSmoothing(data['Close'], trend='add', seasonal=None).fit()
    forecast = model.forecast(forecast_days)
    return forecast.index, forecast.values

def forecast_arima(data):
    model = ARIMA(data['Close'], order=(5,1,0)).fit()
    forecast = model.forecast(forecast_days)
    dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')
    return dates, forecast.values

def forecast_lstm(data):
    X, y, scaler = prepare_lstm_data(data[['Close']])
    model = train_lstm_model(X, y)
    last_sequence = X[-1]
    future_preds = []
    for _ in range(forecast_days):
        pred = model.predict(last_sequence.reshape(1, 60, 1))[0, 0]
        future_preds.append(pred)
        last_sequence = np.append(last_sequence[1:], pred).reshape(60, 1)
    forecast = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')
    return dates, forecast

# Forecast buttons
data = load_data(stock_code)

for label, func in [
    ("Forecast with Linear Regression", forecast_linear_regression),
    ("Forecast with Holt's Model", forecast_holt),
    ("Forecast with ARIMA", forecast_arima),
    ("Forecast with LSTM", forecast_lstm)
]:
    if st.button(label):
        dates, forecast_values = func(data)
        forecast_df = pd.DataFrame({"Date": dates, "Forecasted Price": forecast_values})
        st.subheader(f"Forecast Results: {label}")
        st.dataframe(forecast_df.set_index('Date'))

        # Plot historical + forecast
        plt.figure(figsize=(10, 4))
        plt.plot(data.index, data['Close'], label='Historical')
        plt.plot(dates, forecast_values, label='Forecast', color='green')
        plt.title(f"{label}")
        plt.xlabel("Date")
        plt.ylabel("Price (₹)")
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

