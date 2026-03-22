import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "AAPL")

# Set date range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# --- THE FIX FOR 'NO DATA FOUND' ---
@st.cache_data # This caches the data so you don't hit Yahoo Finance too often
def load_data(ticker):
    # We use 'multi_level_index=False' to keep the dataframe simple
    df = yf.download(ticker, start=start, end=end, multi_level_index=False)
    return df

try:
    data = load_data(stock)
    
    if data.empty:
        st.error(f"No data found for '{stock}'. Please verify the ticker symbol.")
        st.info("Tip: Use 'AAPL' for Apple, 'TSLA' for Tesla, or 'RELIANCE.NS' for NSE stocks.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# --- MODEL LOADING ---
try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"Could not load model file: {e}")
    st.stop()

st.subheader(f"Latest Data for {stock}")
st.write(data.tail(5))

# --- DATA PREPARATION ---
splitting_len = int(len(data) * 0.7)
x_test_data = pd.DataFrame(data['Close'][splitting_len:])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test_data)

# Prepare prediction arrays
x_data = []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])

if len(x_data) > 0:
    x_data = np.array(x_data)
    predictions = model.predict(x_data)
    
    # Inverse scaling for display
    # Note: We create a dummy scaler or use the existing one correctly
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(scaled_data[100:])

    # Plotting
    st.subheader('Original vs Predicted Price')
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(inv_y_test, 'b', label='Original Price')
    ax2.plot(inv_pre, 'r', label='Predicted Price')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Price')
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("Not enough historical data to make a prediction.")
