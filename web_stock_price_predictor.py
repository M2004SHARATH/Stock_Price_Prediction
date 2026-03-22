import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Title of the app
st.title("Stock Price Predictor App")

# Input field for stock ID
stock = st.text_input("Enter the Stock ID", "HDB")

# Date range for stock data (Last 20 years)
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# --- ROBUST DATA FETCHING ---
try:
    # multi_level_index=False prevents the KeyError by keeping columns simple
    data = yf.download(stock, start=start, end=end, multi_level_index=False)
    
    if data.empty:
        st.error(f"No data found for '{stock}'. Try a different Ticker (e.g., AAPL, TSLA).")
        st.stop()
except Exception as e:
    st.error(f"Connection Error: {e}")
    st.stop()

# Load pre-trained model (Ensure this file is in your GitHub repo)
model = load_model("Latest_stock_price_model.keras")

# Display stock data
st.subheader(f"Stock Data for {stock}")
st.write(data.tail(10)) # Showing latest 10 rows

# Splitting the data for Testing
splitting_len = int(len(data) * 0.7)
x_test_data = pd.DataFrame(data['Close'][splitting_len:])

# Define plot function
def plot_graph(figsize, values, full_data, label_val, extra_dataset=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(full_data['Close'], 'b', label='Original Price')
    ax.plot(values, 'orange', label=label_val)
    if extra_dataset is not None:
        ax.plot(extra_dataset, 'g', label='Comparison MA')
    ax.legend()
    return fig

# --- CALCULATING MOVING AVERAGES ---
st.subheader('MA for 250 days')
data['MA_for_250_days'] = data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15, 6), data['MA_for_250_days'], data, 'MA 250'))

st.subheader('MA for 200 days')
data['MA_for_200_days'] = data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15, 6), data['MA_for_200_days'], data, 'MA 200'))

st.subheader('MA for 100 days vs 250 days')
data['MA_for_100_days'] = data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15, 6), data['MA_for_100_days'], data, 'MA 100', data['MA_for_250_days']))

# --- PREDICTION LOGIC ---
scaler = MinMaxScaler(feature_range=(0, 1))

# We must use the last 100 days of the training data + the test data to predict
# To simplify for the web app, we scale based on the visible test set
scaled_data = scaler.fit_transform(x_test_data)

x_data = []
# Assuming the model expects 100 days of history to predict day 101
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])

if len(x_data) > 0:
    x_data = np.array(x_data)
    
    # Making predictions
    predictions = model.predict(x_data)
    
    # Inversing the scaling
    # We use a dummy scale to match the 2D output shape of the prediction
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(scaled_data[100:])

    # Plotting the results
    st.subheader('Original vs Predicted')
    plot_df = pd.DataFrame({
        'Original': inv_y_test.reshape(-1),
        'Predicted': inv_pre.reshape(-1)
    }, index=data.index[splitting_len+100:])
    
    fig2, ax2 = plt.subplots(figsize=(15,6))
    ax2.plot(plot_df['Original'], 'b', label='Original Price')
    ax2.plot(plot_df['Predicted'], 'r', label='Predicted Price')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("Not enough data to generate a prediction (need at least 100 days of test data).")
