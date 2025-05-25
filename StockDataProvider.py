import yfinance as yf # pip install yfinance
import pandas as pd

# --- StockDataProvider Class ---
class StockDataProvider:
    def get_stock_prices(self, ticker_symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
        """Get historical stock data for a single ticker symbol."""
        try:
            ticker_data = yf.Ticker(ticker_symbol)
            # Fetching daily data, empyrical expects daily returns for annualization=252
            hist = ticker_data.history(period=period, interval="1d") 
            if not hist.empty:
                print(f"Successfully fetched historical data for {ticker_symbol}. Shape: {hist.shape}", flush=True)
                return hist
            else:
                print(f"No historical data found for {ticker_symbol} for the period {period} and interval 1d.", flush=True)
                return pd.DataFrame()
        except Exception as e:
            print(f"Could not retrieve historical data for {ticker_symbol}: {e}", flush=True)
            return pd.DataFrame()