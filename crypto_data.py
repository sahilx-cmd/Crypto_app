import ccxt
import pandas as pd
from datetime import datetime, timedelta

def get_crypto_data(symbol='BTC/USDT', exchange_name='binance', timeframe='1d', limit=365):
    try:
        exchange = getattr(ccxt, exchange_name)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data from {exchange_name}: {e}")
        # Fallback: generate dummy data for last 365 days
        dates = pd.date_range(end=datetime.today(), periods=limit)
        prices = pd.Series(1000 + (pd.np.random.randn(limit).cumsum()), index=dates).abs()
        df_fallback = pd.DataFrame({'close': prices})
        df_fallback.index.name = 'timestamp'
        return df_fallback
