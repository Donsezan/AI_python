
# --- FinancialRiskCalculator Class ---
import empyrical #pip install empyrical
import pandas as pd
import numpy as np # Added for VaR calculation
from StockDataProvider import StockDataProvider


class FinancialRiskCalculator:
    def __init__(self, stock_provider: StockDataProvider, risk_free_rate: float = 0.01):
        self.stock_provider = stock_provider
        self.risk_free_rate = risk_free_rate # Annual risk-free rate

    def _get_daily_returns(self, ticker_symbol: str, period: str) -> pd.Series:
        df = self.stock_provider.get_stock_prices(ticker_symbol, period=period, interval="1d") # Ensure daily interval
        if df.empty or 'Close' not in df.columns:
            return pd.Series(dtype=float) 
        return df['Close'].pct_change().dropna()

    def get_standard_deviation(self, ticker_symbol: str, period: str = "1y") -> dict:
        returns = self._get_daily_returns(ticker_symbol, period)
        if returns.empty or len(returns) < 2: 
            return {"error": "Not enough data to calculate standard deviation"}
        try:
            annual_volatility = empyrical.annual_volatility(returns, annualization=252)
            return {"annual_standard_deviation": annual_volatility}
        except Exception as e:
            return {"error": f"Error calculating standard deviation: {str(e)}"}

    def get_max_drawdown(self, ticker_symbol: str, period: str = "1y") -> dict:
        returns = self._get_daily_returns(ticker_symbol, period)
        if returns.empty or len(returns) < 2:
            return {"error": "Not enough data to calculate max drawdown"}
        try:
            max_dd = empyrical.max_drawdown(returns)
            return {"max_drawdown": max_dd} 
        except Exception as e:
            return {"error": f"Error calculating max drawdown: {str(e)}"}

    def get_sharpe_ratio(self, ticker_symbol: str, period: str = "1y") -> dict:
        # Note: empyrical.sharpe_ratio takes daily risk-free rate if annualization is used.
        # Here, self.risk_free_rate is annual. Empyrical handles this if period is 'daily'.
        # For daily returns, risk_free should be daily. (annual_rf / 252)
        returns = self._get_daily_returns(ticker_symbol, period)
        if returns.empty or len(returns) < 2:
            return {"error": "Not enough data to calculate Sharpe ratio"}
        try:
            daily_risk_free = self.risk_free_rate / 252
            sharpe = empyrical.sharpe_ratio(returns, risk_free=int(daily_risk_free), annualization=252)
            return {"sharpe_ratio": sharpe}
        except Exception as e:
            return {"error": f"Error calculating Sharpe ratio: {str(e)}"}

    def get_alpha_beta(self, ticker_symbol: str, period: str = "1y", benchmark_ticker: str = 'SPY') -> dict:
        security_returns = self._get_daily_returns(ticker_symbol, period)
        benchmark_returns = self._get_daily_returns(benchmark_ticker, period)

        if security_returns.empty or benchmark_returns.empty:
            return {"error": "Not enough data for security or benchmark to calculate alpha and beta"}
        
        aligned_returns, aligned_benchmark_returns = security_returns.align(benchmark_returns, join='inner')
        
        if aligned_returns.empty or aligned_benchmark_returns.empty or len(aligned_returns) < 2:
             return {"error": "Not enough aligned data points to calculate alpha and beta"}
        try:
            # risk_free for alpha_beta should be daily if returns are daily and annualization is used.
            daily_risk_free_for_alpha_beta = self.risk_free_rate / 252
            alpha, beta = empyrical.alpha_beta(
                returns=aligned_returns,
                factor_returns=aligned_benchmark_returns,
                risk_free=daily_risk_free_for_alpha_beta, # Using daily risk-free rate
                annualization=252 
            )
            return {"alpha": alpha, "beta": beta}
        except Exception as e:
            return {"error": f"Error calculating alpha and beta: {str(e)}"}

    def get_value_at_risk(self, ticker_symbol: str, period: str = "1y", confidence_level: float = 0.05) -> dict:
        returns = self._get_daily_returns(ticker_symbol, period)
        if returns.empty or len(returns) < 2: # np.percentile needs at least 1 data point
            return {"error": "Not enough data to calculate Value at Risk"}
        try:
            var_value = -np.percentile(returns.dropna(), 100 * confidence_level)
            return {"value_at_risk_percentage": var_value, "confidence_level": confidence_level}
        except Exception as e:
            return {"error": f"Error calculating Value at Risk: {str(e)}"}
