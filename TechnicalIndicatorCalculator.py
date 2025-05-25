# --- TechnicalIndicatorCalculator Class ---
import ta
import ta.momentum
import ta.volume


from StockDataProvider import StockDataProvider


class TechnicalIndicatorCalculator:
    INDICATOR_RSI = "RSI"
    INDICATOR_STOCHASTIC_OSCILLATOR = "StochasticOscillator"
    INDICATOR_ROC = "ROC"
    INDICATOR_MFI = "MFI"
    SUPPORTED_INDICATORS = [INDICATOR_RSI, INDICATOR_STOCHASTIC_OSCILLATOR, INDICATOR_ROC, INDICATOR_MFI]

    def __init__(self, stock_provider: StockDataProvider):
        self.stock_provider = stock_provider

    def calculate_technical_indicator(self, ticker_symbol: str, indicator_type: str, data_period: str = "6mo", data_interval: str = "1d", window: int = 14, smooth_window: int = 3) -> dict:
        if indicator_type not in self.SUPPORTED_INDICATORS:
            return {"error": "Unsupported indicator type"}
        df = self.stock_provider.get_stock_prices(ticker_symbol, period=data_period, interval=data_interval)
        if df.empty:
            return {"error": "Could not retrieve historical data"}

        required_columns_for_mfi = ['High', 'Low', 'Close', 'Volume']
        if indicator_type == self.INDICATOR_MFI and not all(col in df.columns for col in required_columns_for_mfi):
            return {"error": "Dataframe missing required HLCV columns for MFI"}
        required_columns_for_stoch = ['High', 'Low', 'Close']
        if indicator_type == self.INDICATOR_STOCHASTIC_OSCILLATOR and not all(col in df.columns for col in required_columns_for_stoch):
            return {"error": "Dataframe missing required HLC columns for Stochastic Oscillator"}
        if indicator_type in [self.INDICATOR_RSI, self.INDICATOR_ROC] and 'Close' not in df.columns:
            return {"error": "Dataframe missing required Close column"}
        df.dropna(inplace=True)
        if df.empty: # Check if df is empty after dropna
            return {"error": f"Not enough data after cleaning for {indicator_type} calculation"}


        result = {}
        try:
            if indicator_type == self.INDICATOR_RSI:
                indicator = ta.momentum.RSIIndicator(close=df['Close'], window=window)
                rsi_series = indicator.rsi()
                result = {"rsi": rsi_series.iloc[-1] if rsi_series is not None and not rsi_series.empty else "Not enough data"}
            elif indicator_type == self.INDICATOR_STOCHASTIC_OSCILLATOR:
                indicator = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=window, smooth_window=smooth_window)
                stoch_k_series = indicator.stoch()
                stoch_d_series = indicator.stoch_signal()
                result = {
                    "stoch_k": stoch_k_series.iloc[-1] if stoch_k_series is not None and not stoch_k_series.empty else "Not enough data",
                    "stoch_d": stoch_d_series.iloc[-1] if stoch_d_series is not None and not stoch_d_series.empty else "Not enough data"
                }
            elif indicator_type == self.INDICATOR_ROC:
                indicator = ta.momentum.ROCIndicator(close=df['Close'], window=window)
                roc_series = indicator.roc()
                result = {"roc": roc_series.iloc[-1] if roc_series is not None and not roc_series.empty else "Not enough data"}
            elif indicator_type == self.INDICATOR_MFI:
                indicator = ta.volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=window)
                mfi_series = indicator.money_flow_index()
                result = {"mfi": mfi_series.iloc[-1] if mfi_series is not None and not mfi_series.empty else "Not enough data"}
            else:
                return {"error": "Unknown indicator calculation error"}
        except IndexError: # Handles cases where iloc[-1] fails due to insufficient data for window
             return {"error": f"Not enough data to calculate {indicator_type} for the given window {window}"}
        except Exception as e:
            print(f"Error calculating {indicator_type} for {ticker_symbol}: {e}", flush=True)
            return {"error": f"An error occurred during {indicator_type} calculation: {str(e)}"}
        return result
