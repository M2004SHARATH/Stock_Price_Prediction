import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import requests

# -------------------- CONFIG --------------------
API_KEY = "V7WDUTTBBL4KH406"

st.title("📈 Stock Price Predictor App")

stock = st.text_input("Enter Stock Symbol (AAPL, MSFT, TSLA)", "AAPL")

# -------------------- FETCH DATA --------------------
@st.cache_data
def get_stock_data(symbol):
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": API_KEY
        }

        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        # API limit hit
        if "Note" in data:
            return None, "limit"

        # Invalid symbol
        if "Time Series (Daily)" not in data:
            return None, "invalid"

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

        return df, "ok"

    except Exception:
        return None, "error"

# Try API
data, status = get_stock_data(stock)

# -------------------- FALLBACK (NEVER FAILS) --------------------
if status != "ok":
    st.warning("⚠️ API failed. Using demo dataset instead (app will still work).")

    # Fallback dataset (guaranteed working)
    data = pd.read_csv(
        "https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv"
    )

    data = data[['Date', 'SP500']].dropna()
    data.columns = ['Date', 'Close']
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

# -------------------- SHOW DATA --------------------
st.subheader("Stock Data")
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
    st.error("❌ Not enough data.")
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
