import json
from urllib.parse import urlparse # Keep if needed
from datetime import datetime, timedelta # Keep if needed
import pandas as pd
from ChatApplication import ChatApplication
from FinancialRiskCalculator import FinancialRiskCalculator
from StockDataProvider import StockDataProvider
import numpy as np # Added for VaR calculation
from unittest.mock import patch, MagicMock, call # Keep for tests, added call
from openai import OpenAI
import empyrical
import yfinance as yf # pip install yfinance

# --- Test functions ---
def test_get_stock_prices_json_format():
    # ... (implementation as before, truncated for brevity) ...
    print("Running test_get_stock_prices_json_format...", flush=True)
    temp_stock_provider = StockDataProvider()
    @patch('main.yf.Ticker') 
    def test_successful_fetch(mock_yf_ticker):
        print("  Running test_successful_fetch...", flush=True)
        data = {'Open': [150.0, 151.0], 'High': [152.0, 151.5], 'Low': [149.0, 150.0], 'Close': [151.5, 150.5], 'Volume': [100000, 120000]}
        index = pd.to_datetime(['2023-01-01T00:00:00', '2023-01-02T00:00:00'], utc=True)
        mock_df = pd.DataFrame(data, index=index); mock_df.index.name = 'Date'
        mock_ticker_instance = MagicMock(); mock_ticker_instance.history.return_value = mock_df
        mock_yf_ticker.return_value = mock_ticker_instance
        df_result = temp_stock_provider.get_stock_prices("TEST_SUCCESS_TICKER", period="2d", interval="1d")
        assert not df_result.empty
        json_output_string = df_result.reset_index().to_json(orient='records', date_format='iso')
        parsed_json_list = json.loads(json_output_string)
        assert isinstance(parsed_json_list, list) and len(parsed_json_list) == 2
        first_record_dict = parsed_json_list[0]
        assert isinstance(first_record_dict, dict)
        expected_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] 
        assert all(key in first_record_dict for key in expected_keys) and len(first_record_dict.keys()) == len(expected_keys)
        assert first_record_dict['Date'] == '2023-01-01T00:00:00.000Z'
        assert first_record_dict['Open'] == 150.0
        print("  Test case 1 (successful fetch) PASSED", flush=True)
    test_successful_fetch()

    @patch('main.yf.Ticker') 
    def test_empty_fetch(mock_yf_ticker):
        print("  Running test_empty_fetch...", flush=True)
        mock_ticker_instance = MagicMock(); mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_yf_ticker.return_value = mock_ticker_instance
        df_result = temp_stock_provider.get_stock_prices("TEST_EMPTY_TICKER", period="1d", interval="1d")
        assert df_result.empty
        json_output_string = df_result.to_json(orient='records', date_format='iso')
        assert json_output_string == '[]'
        print("  Test case 2 (empty fetch) PASSED", flush=True)
    test_empty_fetch()
    print("test_get_stock_prices_json_format completed successfully.", flush=True)

# --- New Test Suite for FinancialRiskCalculator ---
MOCK_STOCK_TICKER = "MOCKSTOCK"
MOCK_BENCH_TICKER = "MOCKBENCH"
MOCK_RISK_FREE_RATE = 0.01 # Annual

# Generate more realistic mock price data (60 days)
np.random.seed(42) # for reproducibility
mock_dates = pd.date_range(start='2023-01-01', periods=60, freq='B') # Business days

mock_stock_prices = pd.DataFrame({
    'Close': 100 + np.random.randn(60).cumsum() * 0.5 # Random walk based prices
}, index=mock_dates)
mock_stock_prices.loc[mock_stock_prices['Close'] <= 0, 'Close'] = 0.01 # Ensure prices are positive

mock_bench_prices = pd.DataFrame({
    'Close': 100 + np.random.randn(60).cumsum() * 0.3
}, index=mock_dates)
mock_bench_prices.loc[mock_bench_prices['Close'] <= 0, 'Close'] = 0.01


# Helper to convert price DF to returns Series
def prices_to_returns(price_df: pd.DataFrame) -> pd.Series:
    if price_df.empty or 'Close' not in price_df.columns:
        return pd.Series(dtype=float)
    return price_df['Close'].pct_change().dropna()

mock_stock_returns = prices_to_returns(mock_stock_prices)
mock_bench_returns = prices_to_returns(mock_bench_prices)

# Mock for StockDataProvider.get_stock_prices
def mock_get_prices_side_effect(ticker_symbol, period, interval="1d"):
    print(f"Mocked get_stock_prices called for: {ticker_symbol}, period: {period}, interval: {interval}", flush=True)
    if ticker_symbol == MOCK_STOCK_TICKER:
        # Simulate period slicing if needed, though empyrical uses the whole series
        return mock_stock_prices.copy() 
    elif ticker_symbol == MOCK_BENCH_TICKER:
        return mock_bench_prices.copy()
    elif ticker_symbol == "EMPTY":
        return pd.DataFrame()
    elif ticker_symbol == "INSUFFICIENT": # For testing cases needing >1 data point for returns
        return mock_stock_prices.iloc[:1].copy() # Only one price point, so zero returns
    elif ticker_symbol == "BARELY_SUFFICIENT": # For testing cases needing >1 data point for returns
        return mock_stock_prices.iloc[:2].copy() # Two price points, so one return
    return pd.DataFrame()

@patch('main.StockDataProvider.get_stock_prices', side_effect=mock_get_prices_side_effect)
def _test_calc_std_dev(mock_get_prices_method):
    print("  Running _test_calc_std_dev...", flush=True)
    # Use a fresh StockDataProvider instance for each test or rely on the global mock if appropriate
    # Here, FinancialRiskCalculator will instantiate its own StockDataProvider, which we are patching the method of.
    # So, we don't need to pass a StockDataProvider mock to FinancialRiskCalculator.
    risk_calc = FinancialRiskCalculator(StockDataProvider(), risk_free_rate=MOCK_RISK_FREE_RATE) # StockDataProvider() is a dummy here due to patching its method

    # Success case
    result = risk_calc.get_standard_deviation(MOCK_STOCK_TICKER, period="60d")
    expected_std_dev = empyrical.annual_volatility(mock_stock_returns, annualization=252)
    assert 'annual_standard_deviation' in result, f"StdDev Success: Expected key missing. Got: {result}"
    assert np.isclose(result['annual_standard_deviation'], expected_std_dev), f"StdDev Success: Expected {expected_std_dev}, Got {result['annual_standard_deviation']}"
    print("    StdDev Success PASSED", flush=True)

    # Empty data case
    result_empty = risk_calc.get_standard_deviation("EMPTY", period="60d")
    assert "error" in result_empty and "Not enough data" in result_empty["error"], f"StdDev Empty: Unexpected error. Got: {result_empty}"
    print("    StdDev Empty Data PASSED", flush=True)
    
    # Insufficient data case (1 price point -> 0 returns)
    result_insufficient = risk_calc.get_standard_deviation("INSUFFICIENT", period="1d")
    assert "error" in result_insufficient and "Not enough data" in result_insufficient["error"], f"StdDev Insufficient: Unexpected error. Got: {result_insufficient}"
    print("    StdDev Insufficient Data (0 returns) PASSED", flush=True)

    # Barely sufficient data case (2 price points -> 1 return, empyrical needs >1 return for some calcs)
    result_barely = risk_calc.get_standard_deviation("BARELY_SUFFICIENT", period="2d")
    assert "error" in result_barely and "Not enough data" in result_barely["error"], f"StdDev Barely Sufficient: Expected error for 1 return. Got: {result_barely}"
    print("    StdDev Barely Sufficient Data (1 return) PASSED", flush=True)

@patch('main.StockDataProvider.get_stock_prices', side_effect=mock_get_prices_side_effect)
def _test_calc_max_drawdown(mock_get_prices_method):
    print("  Running _test_calc_max_drawdown...", flush=True)
    risk_calc = FinancialRiskCalculator(StockDataProvider(), risk_free_rate=MOCK_RISK_FREE_RATE)
    result = risk_calc.get_max_drawdown(MOCK_STOCK_TICKER, period="60d")
    expected_max_dd = empyrical.max_drawdown(mock_stock_returns)
    assert 'max_drawdown' in result, f"MaxDD Success: Key missing. Got: {result}"
    assert np.isclose(result['max_drawdown'], expected_max_dd), f"MaxDD Success: Expected {expected_max_dd}, Got {result['max_drawdown']}"
    print("    MaxDD Success PASSED", flush=True)
    result_empty = risk_calc.get_max_drawdown("EMPTY", period="60d")
    assert "error" in result_empty and "Not enough data" in result_empty["error"]
    print("    MaxDD Empty Data PASSED", flush=True)

@patch('main.StockDataProvider.get_stock_prices', side_effect=mock_get_prices_side_effect)
def _test_calc_sharpe_ratio(mock_get_prices_method):
    print("  Running _test_calc_sharpe_ratio...", flush=True)
    risk_calc = FinancialRiskCalculator(StockDataProvider(), risk_free_rate=MOCK_RISK_FREE_RATE)
    result = risk_calc.get_sharpe_ratio(MOCK_STOCK_TICKER, period="60d")
    expected_sharpe = empyrical.sharpe_ratio(mock_stock_returns, risk_free=int(MOCK_RISK_FREE_RATE/252), annualization=252)
    assert 'sharpe_ratio' in result, f"Sharpe Success: Key missing. Got: {result}"
    assert np.isclose(result['sharpe_ratio'], expected_sharpe), f"Sharpe Success: Expected {expected_sharpe}, Got {result['sharpe_ratio']}"
    print("    Sharpe Success PASSED", flush=True)
    result_empty = risk_calc.get_sharpe_ratio("EMPTY", period="60d")
    assert "error" in result_empty and "Not enough data" in result_empty["error"]
    print("    Sharpe Empty Data PASSED", flush=True)

@patch('main.StockDataProvider.get_stock_prices', side_effect=mock_get_prices_side_effect)
def _test_calc_alpha_beta(mock_get_prices_method):
    print("  Running _test_calc_alpha_beta...", flush=True)
    risk_calc = FinancialRiskCalculator(StockDataProvider(), risk_free_rate=MOCK_RISK_FREE_RATE)
    result = risk_calc.get_alpha_beta(MOCK_STOCK_TICKER, period="60d", benchmark_ticker=MOCK_BENCH_TICKER)
    
    aligned_stock, aligned_bench = mock_stock_returns.align(mock_bench_returns, join='inner')
    expected_alpha, expected_beta = empyrical.alpha_beta(aligned_stock, aligned_bench, risk_free=MOCK_RISK_FREE_RATE/252, annualization=252)
    
    assert 'alpha' in result and 'beta' in result, f"AlphaBeta Success: Keys missing. Got: {result}"
    assert np.isclose(result['alpha'], expected_alpha), f"Alpha Success: Expected {expected_alpha}, Got {result['alpha']}"
    assert np.isclose(result['beta'], expected_beta), f"Beta Success: Expected {expected_beta}, Got {result['beta']}"
    print("    AlphaBeta Success PASSED", flush=True)
    
    result_empty_stock = risk_calc.get_alpha_beta("EMPTY", period="60d", benchmark_ticker=MOCK_BENCH_TICKER)
    assert "error" in result_empty_stock and "Not enough data" in result_empty_stock["error"]
    print("    AlphaBeta Empty Stock Data PASSED", flush=True)
    
    result_empty_bench = risk_calc.get_alpha_beta(MOCK_STOCK_TICKER, period="60d", benchmark_ticker="EMPTY")
    assert "error" in result_empty_bench and "Not enough data" in result_empty_bench["error"]
    print("    AlphaBeta Empty Bench Data PASSED", flush=True)

@patch('main.StockDataProvider.get_stock_prices', side_effect=mock_get_prices_side_effect)
def _test_calc_value_at_risk(mock_get_prices_method):
    print("  Running _test_calc_value_at_risk...", flush=True)
    risk_calc = FinancialRiskCalculator(StockDataProvider(), risk_free_rate=MOCK_RISK_FREE_RATE)
    
    # Test with 0.05 confidence
    result_05 = risk_calc.get_value_at_risk(MOCK_STOCK_TICKER, period="60d", confidence_level=0.05)
    expected_var_05 = -np.percentile(mock_stock_returns.dropna(), 5)
    assert 'value_at_risk_percentage' in result_05, f"VaR 0.05 Success: Key missing. Got: {result_05}"
    assert np.isclose(result_05['value_at_risk_percentage'], expected_var_05), f"VaR 0.05 Success: Expected {expected_var_05}, Got {result_05['value_at_risk_percentage']}"
    assert result_05['confidence_level'] == 0.05
    print("    VaR 0.05 Success PASSED", flush=True)

    # Test with 0.01 confidence
    result_01 = risk_calc.get_value_at_risk(MOCK_STOCK_TICKER, period="60d", confidence_level=0.01)
    expected_var_01 = -np.percentile(mock_stock_returns.dropna(), 1)
    assert 'value_at_risk_percentage' in result_01, f"VaR 0.01 Success: Key missing. Got: {result_01}"
    assert np.isclose(result_01['value_at_risk_percentage'], expected_var_01), f"VaR 0.01 Success: Expected {expected_var_01}, Got {result_01['value_at_risk_percentage']}"
    assert result_01['confidence_level'] == 0.01
    print("    VaR 0.01 Success PASSED", flush=True)

    result_empty = risk_calc.get_value_at_risk("EMPTY", period="60d")
    assert "error" in result_empty and "Not enough data" in result_empty["error"]
    print("    VaR Empty Data PASSED", flush=True)


def test_financial_risk_calculations():
    """Main test function to orchestrate risk calculation sub-tests."""
    print("\nRunning test_financial_risk_calculations...", flush=True)
    # It's important that each sub-test patches independently if they need different mock behaviors
    # or to ensure clean state. The @patch decorator on each sub-test handles this.
    _test_calc_std_dev()
    _test_calc_max_drawdown()
    _test_calc_sharpe_ratio()
    _test_calc_alpha_beta()
    _test_calc_value_at_risk()
    print("test_financial_risk_calculations completed successfully.", flush=True)


# --- Main execution block ---
if __name__ == "__main__":
    # Run unit tests first
    test_financial_risk_calculations() # New test suite
    test_get_stock_prices_json_format() 

    app = ChatApplication()
    app.start_chat()
