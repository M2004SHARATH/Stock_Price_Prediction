import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Title of the app
st.title("Stock Price Predictor App")

# Input field for stock ID
stock = st.text_input("Enter the Stock ID", "AAPL")

# Date range for stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# --- FIX 1: DOWNLOAD DATA CORRECTLY ---
# We add multi_level_index=False to avoid the KeyError 'Close'
steel_authority = yf.download(stock, start=start, end=end, multi_level_index=False)

# Load pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader("Stock Data")
st.write(steel_authority)

# --- FIX 2: DATA CHECK ---
if steel_authority.empty:
    st.error("No data found for this Stock ID. Please check the symbol (e.g., AAPL, MSFT, TSLA).")
    st.stop()

# Splitting the data
splitting_len = int(len(steel_authority) * 0.7)
# Use a clear reference to 'Close'
x_test = pd.DataFrame(steel_authority.Close[splitting_len:])

# Define plot function (Original Logic)
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset= None):
    fig = plt.figure(figsize=figsize)
    plt.plot(full_data.Close, 'b', label='Original Price')
    plt.plot(values, 'orange', label='Moving Average')
    if extra_data:
        plt.plot(extra_dataset, 'g')
    return fig

# Moving Averages
steel_authority['MA_for_250_days'] = steel_authority.Close.rolling(250).mean()
st.subheader('Original Close Price and MA for 250 days')
st.pyplot(plot_graph((15, 6), steel_authority['MA_for_250_days'], steel_authority, 0))

steel_authority['MA_for_200_days'] = steel_authority.Close.rolling(200).mean()
st.subheader('Original Close Price and MA for 200 days')
st.pyplot(plot_graph((15, 6), steel_authority['MA_for_200_days'], steel_authority, 0))

steel_authority['MA_for_100_days'] = steel_authority.Close.rolling(100).mean()
st.subheader('Original Close Price and MA for 100 days')
st.pyplot(plot_graph((15, 6), steel_authority['MA_for_100_days'], steel_authority, 1, steel_authority['MA_for_250_days']))

# --- FIX 3: SCALING LOGIC ---
scaler = MinMaxScaler(feature_range=(0, 1))
# We use the 'Close' column directly to avoid the 'stock' variable KeyError
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Making predictions
predictions = model.predict(x_data)

# Inversing the scaling for plotting
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Plotting the results
plotting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index = steel_authority.index[splitting_len+100:]
)

st.subheader("Original vs Predicted Price")
st.write(plotting_data.tail(10))

fig2 = plt.figure(figsize=(15,6))
plt.plot(plotting_data['original_test_data'], 'b', label='Original Price')
plt.plot(plotting_data['predictions'], 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
