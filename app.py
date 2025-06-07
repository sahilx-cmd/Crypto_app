import streamlit as st
import matplotlib.pyplot as plt
from crypto_data import get_crypto_data
from sarima_model import train_sarima
st.set_page_config(page_title="Crypto Price Predictor", layout="wide")
st.title("📈 Cryptocurrency Price Forecast using SARIMA")

symbol = st.sidebar.text_input("Enter Crypto Pair (e.g. BTC/USDT):", "BTC/USDT")
forecast_days = 10

df = get_crypto_data(symbol)
if df is None or df.empty:
    st.error(f"❌ Could not load data for {symbol}")
    st.stop()

st.subheader("📊 Historical Closing Prices")
st.line_chart(df['close'])

with st.expander("📋 View Raw Data"):
    st.dataframe(df.tail(30))

forecast_df, rmse, actual, predicted = train_sarima(df, forecast_days)

st.subheader("🔮 Forecast Chart (Next 10 Days)")
fig, ax = plt.subplots(figsize=(10, 4))
df['close'].plot(ax=ax, label='Historical', color='blue')
forecast_df.set_index('date')['forecast'].plot(ax=ax, label='Forecast', linestyle='--', color='orange')
ax.set_title(f"{symbol} Forecast for Next {forecast_days} Days")
ax.set_ylabel("Price")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig)

st.subheader("📉 RMSE Evaluation")
st.markdown(f"**Root Mean Squared Error (RMSE):** `{rmse:.2f}`")

st.subheader("📈 Actual vs Predicted on Test Data")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(actual, label='Actual', marker='o')
ax2.plot(predicted, label='Predicted', linestyle='--', marker='x')
ax2.set_title("Test Performance on Last 10 Days")
ax2.set_xlabel("Day")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

st.subheader("📅 Forecasted Prices for Next 10 Days")
st.dataframe(forecast_df)
