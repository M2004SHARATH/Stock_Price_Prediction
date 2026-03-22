import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# -------------------- TITLE --------------------
st.title("📈 Stock Price Predictor App")

# -------------------- INPUT --------------------
stock_input = st.text_input("Enter Stock Ticker", "RELIANCE")

# -------------------- DATE RANGE --------------------
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# -------------------- DOWNLOAD DATA (SMART HANDLING) --------------------
def load_data(ticker):
    data = yf.download(ticker, start=start, end=end)
    return data

data = load_data(stock_input)

# Try Indian suffix if no data
if data.empty and "." not in stock_input:
    stock_input_ns = stock_input + ".NS"
    data = load_data(stock_input_ns)
    if not data.empty:
        stock_input = stock_input_ns

# Final check
if data.empty:
    st.error("❌ No data found. Try valid tickers like:")
    st.write([
        "AAPL", "TSLA", "GOOGL",
        "RELIANCE.NS", "TCS.NS", "INFY.NS"
    ])
    st.stop()

# -------------------- SHOW DATA --------------------
st.subheader(f"Stock Data for {stock_input}")
st.write(data.tail())

# -------------------- LOAD MODEL --------------------
model = load_model("Latest_stock_price_model.keras")

# -------------------- SPLIT DATA --------------------
splitting_len = int(len(data) * 0.7)
train_data = data[:splitting_len]
test_data = data[splitting_len:]

x_test = pd.DataFrame(test_data['Close'])

# -------------------- PLOT FUNCTION --------------------
def plot_graph(figsize, ma_values, full_data, extra=False, extra_data=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(full_data['Close'], label='Close Price', color='blue')
    plt.plot(ma_values, label='Moving Average', color='orange')
    
    if extra and extra_data is not None:
        plt.plot(extra_data, label='Extra MA', color='green')

    plt.legend()
    return fig

# -------------------- MOVING AVERAGES --------------------
data['MA250'] = data['Close'].rolling(250).mean()
st.subheader("Close Price & 250-Day MA")
st.pyplot(plot_graph((15, 6), data['MA250'], data))

data['MA200'] = data['Close'].rolling(200).mean()
st.subheader("Close Price & 200-Day MA")
st.pyplot(plot_graph((15, 6), data['MA200'], data))

data['MA100'] = data['Close'].rolling(100).mean()
st.subheader("Close Price & 100-Day MA")
st.pyplot(plot_graph((15, 6), data['MA100'], data, True, data['MA250']))

# -------------------- SCALING --------------------
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit ONLY on training data
scaler.fit(train_data[['Close']])

scaled_test = scaler.transform(x_test[['Close']])

# -------------------- CREATE SEQUENCES --------------------
x_data, y_data = [], []

for i in range(100, len(scaled_test)):
    x_data.append(scaled_test[i-100:i])
    y_data.append(scaled_test[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Safety check
if len(x_data) == 0:
    st.error("❌ Not enough data to make predictions.")
    st.stop()

# -------------------- PREDICTION --------------------
predictions = model.predict(x_data)

# -------------------- INVERSE SCALING --------------------
inv_predictions = scaler.inverse_transform(predictions)
inv_y = scaler.inverse_transform(y_data)

# -------------------- RESULTS DATAFRAME --------------------
plotting_data = pd.DataFrame({
    'Actual Price': inv_y.reshape(-1),
    'Predicted Price': inv_predictions.reshape(-1)
}, index=data.index[splitting_len + 100:])

# -------------------- DISPLAY RESULTS --------------------
st.subheader("📊 Actual vs Predicted Prices")
st.write(plotting_data.tail())

# -------------------- FINAL GRAPH --------------------
st.subheader("📉 Prediction Graph")

fig = plt.figure(figsize=(15, 6))

plt.plot(data['Close'][:splitting_len + 100], label="Training Data")
plt.plot(plotting_data['Actual Price'], label="Actual Price")
plt.plot(plotting_data['Predicted Price'], label="Predicted Price")

plt.legend()
st.pyplot(fig)
