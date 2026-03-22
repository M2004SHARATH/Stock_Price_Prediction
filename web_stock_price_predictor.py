import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("Stock Price Predictor App")

# Input stock ticker
stock = st.text_input("Enter the Stock ID", "HDB")

# Date range
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download data
data = yf.download(stock, start=start, end=end)

# Check if data is empty
if data.empty:
    st.error("No data found. Please enter a valid stock ticker.")
    st.stop()

# Load model
model = load_model("Latest_stock_price_model.keras")

# Show data
st.subheader("Stock Data")
st.write(data.tail())

# Split data
splitting_len = int(len(data) * 0.7)
train_data = data[:splitting_len]
test_data = data[splitting_len:]

# Use ONLY Close column
x_test = pd.DataFrame(test_data['Close'])

# Plot function
def plot_graph(figsize, values, full_data, extra_data=False, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label='Moving Average')
    plt.plot(full_data.Close, 'blue', label='Close Price')
    if extra_data and extra_dataset is not None:
        plt.plot(extra_dataset, 'green', label='Extra MA')
    plt.legend()
    return fig

# Moving averages
data['MA250'] = data['Close'].rolling(250).mean()
st.subheader('Close Price & 250-Day MA')
st.pyplot(plot_graph((15, 6), data['MA250'], data))

data['MA200'] = data['Close'].rolling(200).mean()
st.subheader('Close Price & 200-Day MA')
st.pyplot(plot_graph((15, 6), data['MA200'], data))

data['MA100'] = data['Close'].rolling(100).mean()
st.subheader('Close Price & 100-Day MA')
st.pyplot(plot_graph((15, 6), data['MA100'], data, True, data['MA250']))

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit ONLY on training data (important)
scaler.fit(train_data[['Close']])

scaled_test = scaler.transform(x_test[['Close']])

# Create sequences
x_data, y_data = [], []

for i in range(100, len(scaled_test)):
    x_data.append(scaled_test[i-100:i])
    y_data.append(scaled_test[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Check if enough data
if len(x_data) == 0:
    st.error("Not enough data for prediction.")
    st.stop()

# Predictions
predictions = model.predict(x_data)

# Inverse transform
inv_predictions = scaler.inverse_transform(predictions)
inv_y = scaler.inverse_transform(y_data)

# Create dataframe
plotting_data = pd.DataFrame({
    'Original': inv_y.reshape(-1),
    'Predicted': inv_predictions.reshape(-1)
}, index=data.index[splitting_len + 100:])

# Show table
st.subheader("Original vs Predicted")
st.write(plotting_data.tail())

# Plot results
st.subheader("Prediction Graph")
fig = plt.figure(figsize=(15, 6))

plt.plot(data['Close'][:splitting_len + 100], label="Training Data")
plt.plot(plotting_data['Original'], label="Actual Price")
plt.plot(plotting_data['Predicted'], label="Predicted Price")

plt.legend()
st.pyplot(fig)
