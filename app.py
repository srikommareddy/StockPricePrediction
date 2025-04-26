#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# App Title
st.title("📈 Stock Price Forecasting App")

# User Input
stock_code = st.text_input("Enter Stock Code (e.g., TCS.NS):", value="TCS.NS")
start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Download and Display Line Chart
if st.button("Download & Show Line Chart"):
    data = yf.download(stock_code, start=start_date, end=end_date)
    if not data.empty:
        st.success("Data downloaded successfully!")

        # Line Chart
        st.subheader("Stock Price Line Chart")
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Close'], label='Close Price')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.set_title(f"{stock_code} Closing Price")
        ax.legend()
        st.pyplot(fig)

        

        # Save for reuse
        data.to_csv("downloaded_data.csv")
    else:
        st.error("Failed to download data. Check the stock code.")

# Forecasting Function

def load_data():
    return pd.read_csv("downloaded_data.csv", index_col=0, parse_dates=True)

def display_forecast_chart(dates, forecast, label):
    data = load_data()
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Historical')
    ax.plot(dates, forecast, label=label, linestyle='--')
    ax.set_title(f"{label} Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(pd.DataFrame({"Date": dates, "Forecasted Price": forecast}))

# Forecast Buttons
if st.button("🔮 Forecast with Linear Regression"):
    data = load_data()
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    data = data.dropna()

    features = data[['MA7', 'MA21']]
    target = data['Close']

    X_train, y_train = features[:-30], target[:-30]
    X_test, y_test = features[-30:], target[-30:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, f"{stock_code}_linear.pkl")

    future_features = features[-1:].values
    forecast_values = model.predict(np.tile(future_features, (30, 1)))
    forecast_values = forecast_values.flatten()
    forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30, freq='B')

    display_forecast_chart(forecast_dates, forecast_values, "Linear Regression")

if st.button("📈 Forecast with Holt’s Model"):
    data = load_data()
    model = ExponentialSmoothing(data['Close'], trend='add', seasonal=None)
    model_fit = model.fit()
    forecast = model_fit.forecast(30)
    forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30, freq='B')
    joblib.dump(model_fit, f"{stock_code}_holt.pkl")

    display_forecast_chart(forecast_dates, forecast.values, "Holt's Model")

if st.button("⚙ Forecast with ARIMA"):
    data = load_data()
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(30)
    forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30, freq='B')
    joblib.dump(model_fit, f"{stock_code}_arima.pkl")

    display_forecast_chart(forecast_dates, forecast.values, "ARIMA Model")

if st.button("🤖 Forecast with LSTM"):
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler

    data = load_data()
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    X_train, X_test = X[:-30], X[-30:]
    y_train, y_test = y[:-30], y[-30:]

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    model.save(f"{stock_code}_lstm.h5")
    joblib.dump(scaler, f"{stock_code}_scaler.pkl")

    # Forecasting
    forecast_input = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    forecast = []
    for _ in range(30):
        pred = model.predict(forecast_input, verbose=0)
        forecast.append(pred[0][0])
        forecast_input = np.append(forecast_input[:, 1:, :], [[pred]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30, freq='B')

    display_forecast_chart(forecast_dates, forecast, "LSTM Model")
