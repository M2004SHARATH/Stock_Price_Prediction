import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import time

# -------------------- TITLE --------------------
st.title("📈 Stock Price Predictor App")

# -------------------- INPUT --------------------
stock = st.text_input("Enter Stock Ticker", "AAPL")

# -------------------- FETCH DATA (RETRY LOGIC) --------------------
@st.cache_data
def load_data(ticker):
    for i in range(3):  # retry 3 times
        try:
            data = yf.Ticker(ticker).history(period="20y")

            if not data.empty:
                return data

        except:
            time.sleep(2)

    return pd.DataFrame()

data = load_data(stock)

# Try Indian ticker automatically
if data.empty and "." not in stock:
    data = load_data(stock + ".NS")

# Final check
if data.empty:
    st.error("❌ Unable to fetch stock data.")
    st.info("👉 Try: AAPL, TSLA, MSFT, RELIANCE.NS, TCS.NS")
    st.stop()

# -------------------- SHOW DATA --------------------
st.subheader(f"Stock Data for {stock}")
st.write(data.tail())

# -------------------- LOAD MODEL --------------------
model = load_model("Latest_stock_price_model.keras")

# -------------------- SPLIT --------------------
split = int(len(data) * 0.7)
train = data[:split]
test = data[split:]

x_test = test[['Close']]

# -------------------- MOVING AVERAGES --------------------
data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()

st.subheader("📊 Moving Averages")
fig1 = plt.figure(figsize=(15,6))
plt.plot(data['Close'], label="Close")
plt.plot(data['MA100'], label="MA100")
plt.plot(data['MA200'], label="MA200")
plt.legend()
st.pyplot(fig1)

# -------------------- SCALING --------------------
scaler = MinMaxScaler()
scaler.fit(train[['Close']])

scaled_test = scaler.transform(x_test)

# -------------------- SEQUENCE --------------------
x_data, y_data = [], []

for i in range(100, len(scaled_test)):
    x_data.append(scaled_test[i-100:i])
    y_data.append(scaled_test[i])

x_data, y_data = np.array(x_data), np.array(y_data)

if len(x_data) == 0:
    st.error("❌ Not enough data for prediction.")
    st.stop()

# -------------------- PREDICT --------------------
pred = model.predict(x_data)

# -------------------- INVERSE --------------------
pred_inv = scaler.inverse_transform(pred)
y_inv = scaler.inverse_transform(y_data)

# -------------------- RESULT --------------------
result = pd.DataFrame({
    "Actual": y_inv.flatten(),
    "Predicted": pred_inv.flatten()
}, index=data.index[split+100:])

st.subheader("📊 Results")
st.write(result.tail())

# -------------------- GRAPH --------------------
st.subheader("📉 Prediction Graph")

fig2 = plt.figure(figsize=(15,6))
plt.plot(data['Close'][:split+100], label="Train")
plt.plot(result['Actual'], label="Actual")
plt.plot(result['Predicted'], label="Predicted")
plt.legend()

st.pyplot(fig2)
