# app.py
import streamlit as st
import matplotlib.pyplot as plt
from crypto_data import get_crypto_data
from arima_forcast import train_arima

st.set_page_config(page_title="Crypto Price Predictor", layout="wide")
st.title("📈 Cryptocurrency Price Forecast using ARIMA")

# Set symbol and forecast days
st.sidebar.title("⚙️ Settings")
symbol = st.sidebar.text_input("Enter Crypto Pair (e.g. BTC/USDT):", "BTC/USDT")
forecast_days = 10  # Fixed prediction for next 10 days

# Fetch data
with st.spinner("📥 Fetching data..."):
    df = get_crypto_data(symbol)

    if df is None or df.empty:
        st.error("❌ Failed to fetch data. Please check the symbol or try again.")
        st.stop()

    st.subheader("📊 Historical Closing Prices")
    st.line_chart(df['close'])

    with st.expander("📋 View Raw Data"):
        st.dataframe(df.tail(30))

# Train and forecast
with st.spinner("🤖 Training ARIMA model..."):
    forecast_df, rmse, actual, predicted = train_arima(df, forecast_days)
    forecast_df.set_index('date', inplace=True)

# Plot: Forecast vs Historical
st.subheader("🔮 Forecast Chart (Next 10 Days)")
fig1, ax1 = plt.subplots(figsize=(10, 4))
df['close'].plot(ax=ax1, label='Historical', color='blue')
forecast_df['forecast'].plot(ax=ax1, label='Forecast', linestyle='--', color='orange')
ax1.set_title(f"{symbol} Forecast for Next {forecast_days} Days")
ax1.set_ylabel("Price")
ax1.set_xlabel("Date")
ax1.legend()
st.pyplot(fig1)

# RMSE
st.subheader("📉 RMSE Evaluation")
st.markdown(f"**Root Mean Squared Error (RMSE):** `{rmse:.2f}`")

# Plot: Actual vs Predicted (Test Data)
st.subheader("📈 Actual vs Predicted on Test Data")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(actual, label='Actual', marker='o')
ax2.plot(predicted, label='Predicted', linestyle='--', marker='x')
ax2.set_title("Test Performance on Last 10 Days")
ax2.set_xlabel("Day")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# Forecast Table
st.subheader("📅 Forecasted Prices for Next 10 Days")
st.dataframe(forecast_df.reset_index())
