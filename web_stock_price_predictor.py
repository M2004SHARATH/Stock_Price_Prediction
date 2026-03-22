import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests

# -------------------- API KEY --------------------
API_KEY = "V7WDUTTBBL4KH406"

# -------------------- TITLE --------------------
st.title("📈 Stock Price Predictor App")

# -------------------- INPUT --------------------
stock = st.text_input("Enter Stock Symbol (e.g. AAPL, MSFT, TSLA)", "AAPL")

# -------------------- FETCH DATA --------------------
@st.cache_data
def get_stock_data(symbol):
    url = "https://www.alphavantage.co/query"
    
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Handle API limit
    if "Note" in data:
        st.warning("⚠️ API limit reached (5 requests/min). Please wait 1 minute.")
        return pd.DataFrame()

    # Handle invalid symbol
    if "Time Series (Daily)" not in data:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")

    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adj Close",
        "6. volume": "Volume"
    })

    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df

# Load data
data = get_stock_data(stock)

# -------------------- ERROR HANDLING --------------------
if data.empty:
    st.error("❌ Failed to fetch data.")
    st.info("👉 Try valid US stocks like: AAPL, MSFT, TSLA")
    st.stop()

# -------------------- SHOW DATA --------------------
st.subheader(f"Stock Data for {stock}")
st.write(data.tail())

# -------------------- LOAD MODEL --------------------
model = load_model("Latest_stock_price_model.keras")

# -------------------- SPLIT DATA --------------------
split = int(len(data) * 0.7)
train = data[:split]
test = data[split:]

x_test = test[['Close']]

# -------------------- MOVING AVERAGES --------------------
data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()

st.subheader("📊 Moving Averages")
fig1 = plt.figure(figsize=(15, 6))
plt.plot(data['Close'], label="Close Price")
plt.plot(data['MA100'], label="MA100")
plt.plot(data['MA200'], label="MA200")
plt.legend()
st.pyplot(fig1)

# -------------------- SCALING --------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train[['Close']])

scaled_test = scaler.transform(x_test)

# -------------------- CREATE SEQUENCES --------------------
x_data, y_data = [], []

for i in range(100, len(scaled_test)):
    x_data.append(scaled_test[i-100:i])
    y_data.append(scaled_test[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Safety check
if len(x_data) == 0:
    st.error("❌ Not enough data for prediction.")
    st.stop()

# -------------------- PREDICTION --------------------
predictions = model.predict(x_data)

# -------------------- INVERSE SCALING --------------------
pred_inv = scaler.inverse_transform(predictions)
y_inv = scaler.inverse_transform(y_data)

# -------------------- RESULTS --------------------
result = pd.DataFrame({
    "Actual Price": y_inv.flatten(),
    "Predicted Price": pred_inv.flatten()
}, index=data.index[split + 100:])

st.subheader("📊 Actual vs Predicted Prices")
st.write(result.tail())

# -------------------- FINAL GRAPH --------------------
st.subheader("📉 Prediction Graph")

fig2 = plt.figure(figsize=(15, 6))
plt.plot(data['Close'][:split + 100], label="Training Data")
plt.plot(result['Actual Price'], label="Actual Price")
plt.plot(result['Predicted Price'], label="Predicted Price")
plt.legend()

st.pyplot(fig2)
