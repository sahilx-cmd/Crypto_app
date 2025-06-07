# arima_forecast.py
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def train_arima(df, forecast_days=10):
    series = df['close'].dropna()

    train = series[:-forecast_days]
    test = series[-forecast_days:]

    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_days)
    # Calculate accuracy as 1 - (MAE / mean of actual values)
    mae = np.mean(np.abs(test - forecast))
    accuracy = 1 - (mae / np.mean(np.abs(test)))

    # Forecast into the future using full dataset
    full_model = ARIMA(series, order=(5,1,0))
    full_fit = full_model.fit()
    future_forecast = full_fit.forecast(steps=forecast_days)

    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq='D')[1:]
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': future_forecast})

    return forecast_df, accuracy, test.reset_index(drop=True), forecast.reset_index(drop=True)