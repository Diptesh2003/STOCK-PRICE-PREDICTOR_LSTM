
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Price Forecaster", layout="wide")
model = load_model("stock_lstm_model.h5")
st.title("📈 Extended Stock Price Forecasting App")
st.write("Use an LSTM model to predict and visualize stock prices, including a 7-day forecast.")
model_type = st.sidebar.selectbox("Choose Model", ["LSTM (Default)", "Coming Soon..."])
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")

def load_data(ticker):
    df = yf.download(ticker, start="2015-01-01", end="2024-12-31")
    df = df[['Close']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return df, scaled_data, scaler

def forecast_next_days(model, scaled_data, scaler, days=7):
    last_60 = scaled_data[-60:]
    predictions = []
    for _ in range(days):
        input_seq = np.reshape(last_60, (1, 60, 1))
        pred_scaled = model.predict(input_seq, verbose=0)
        predictions.append(pred_scaled[0][0])
        last_60 = np.append(last_60, pred_scaled)[-60:]
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predicted_prices

if st.button("Run Forecast"):
    df, scaled_data, scaler = load_data(ticker)
    last_60_days = scaled_data[-60:]
    X_input = np.reshape(last_60_days, (1, 60, 1))
    next_day_scaled = model.predict(X_input)
    next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
    st.success(f"📌 Predicted Next Closing Price for {ticker}: **${next_day_price:.2f}**")
    forecast_prices = forecast_next_days(model, scaled_data, scaler, days=7)
    last_date = df.index[-1]
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
    st.subheader("📉 Last 100 Days and 7-Day Forecast")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[-100:], label='Actual Closing Price', linewidth=2)
    ax.plot(forecast_dates, forecast_prices, label='7-Day Forecast', marker='o', linestyle='--')
    ax.set_title(f"{ticker.upper()} - Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': forecast_prices})
    st.dataframe(forecast_df.set_index('Date').style.format({'Predicted Price': '${:.2f}'}))
