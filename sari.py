from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np

def train_sarima(df, forecast_days=10):
    series = df['close'].dropna()

    train = series[:-forecast_days]
    test = series[-forecast_days:]

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)  # weekly seasonality

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=forecast_days)

    # Calculate accuracy as 1 - (MAE / mean of actual values)
    mae = np.mean(np.abs(test - forecast))
    accuracy = 1 - (mae / np.mean(np.abs(test)))

    full_model = SARIMAX(series, order=order, seasonal_order=seasonal_order,enforce_stationarity=False, enforce_invertibility=False)
    full_fit = full_model.fit(disp=False)
    future_forecast = full_fit.forecast(steps=forecast_days)

    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': future_forecast})

    return forecast_df, accuracy, test.reset_index(drop=True), forecast.reset_index(drop=True)
