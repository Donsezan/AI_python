import json
import random # Keep if needed by any function, otherwise can remove
from urllib.parse import urlparse # Keep if needed
import webbrowser # Keep if needed
from datetime import datetime, timedelta # Keep if needed
import os # Keep if needed

import yfinance as yf
import pandas as pd
import ta
import empyrical # Added for financial metrics
import numpy as np # Added for VaR calculation
from unittest.mock import patch, MagicMock, call # Keep for tests, added call
from openai import OpenAI


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

# --- TechnicalIndicatorCalculator Class ---
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

# --- FinancialRiskCalculator Class ---
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
            sharpe = empyrical.sharpe_ratio(returns, risk_free=daily_risk_free, annualization=252)
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

# --- ToolProcessor Class (remains largely the same, ensure it has risk_calculator) ---
class ToolProcessor:
    def __init__(self, stock_provider: StockDataProvider, 
                 indicator_calculator: TechnicalIndicatorCalculator, 
                 risk_calculator: FinancialRiskCalculator, 
                 client: OpenAI, model: str):
        self.stock_provider = stock_provider
        self.indicator_calculator = indicator_calculator
        self.risk_calculator = risk_calculator 
        self.client = client
        self.model = model
        
        self.function_mapping = {
            "get_stock_prices": lambda args: self.stock_provider.get_stock_prices(
                args["ticker_symbol"], args.get("period", "1mo"), args.get("interval", "1d")),
            "calculate_technical_indicator": lambda args: self.indicator_calculator.calculate_technical_indicator(
                args["ticker_symbol"], args["indicator_type"], args.get("data_period", "6mo"),
                args.get("data_interval", "1d"), args.get("window", 14), args.get("smooth_window", 3)),
            "get_standard_deviation": lambda args: self.risk_calculator.get_standard_deviation(
                args["ticker_symbol"], args.get("period", "1y")),
            "get_max_drawdown": lambda args: self.risk_calculator.get_max_drawdown(
                args["ticker_symbol"], args.get("period", "1y")),
            "get_sharpe_ratio": lambda args: self.risk_calculator.get_sharpe_ratio( # benchmark_ticker removed from direct call
                args["ticker_symbol"], args.get("period", "1y")),
            "get_alpha_beta": lambda args: self.risk_calculator.get_alpha_beta(
                args["ticker_symbol"], args.get("period", "1y"), args.get("benchmark_ticker", "SPY")),
            "get_value_at_risk": lambda args: self.risk_calculator.get_value_at_risk(
                args["ticker_symbol"], args.get("period", "1y"), args.get("confidence_level", 0.05)),
        }

    def process_tool_calls(self, response_with_tool_calls, current_call_context_messages):
        # ... (implementation as before) ...
        tool_calls = response_with_tool_calls.choices[0].message.tool_calls
        messages_for_tool_sequence = list(current_call_context_messages)
        assistant_tool_call_request_message = response_with_tool_calls.choices[0].message
        messages_for_tool_sequence.append(assistant_tool_call_request_message)

        for tool_call in tool_calls:
            arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments.strip() else {}
            result_content = "" 
            if tool_call.function.name in self.function_mapping:
                try:
                    if tool_call.function.name == "get_stock_prices":
                        df_result = self.function_mapping[tool_call.function.name](arguments)
                        if not df_result.empty:
                            result_content = df_result.reset_index().to_json(orient='records', date_format='iso')
                        else:
                            result_content = json.dumps({"message": "No data found or error fetching data."}) 
                    else: 
                         result_content_dict = self.function_mapping[tool_call.function.name](arguments)
                         result_content = json.dumps(result_content_dict)
                except Exception as e:
                    print(f"Error during tool call {tool_call.function.name}: {e}", flush=True)
                    result_content = json.dumps({"error": f"Error executing tool {tool_call.function.name}: {str(e)}"})
            else:
                print(f"Unknown function name: {tool_call.function.name}", flush=True)
                result_content = json.dumps({"error": f"Unknown function {tool_call.function.name}"})

            tool_result_message = {
                "role": "tool",
                "content": result_content,
                "tool_call_id": tool_call.id,
            }
            messages_for_tool_sequence.append(tool_result_message)

        final_response_after_tools = self.client.chat.completions.create(
            model=self.model,
            messages=messages_for_tool_sequence,
        )
        final_content = final_response_after_tools.choices[0].message.content
        messages_for_tool_sequence.append({"role": "assistant", "content": final_content})
        return final_content, messages_for_tool_sequence


# --- ChatApplication Class (TOOLS_DEFINITION and __init__ are key changes) ---
class ChatApplication:
    TOOLS_DEFINITION = [
        # ... (existing get_stock_prices and calculate_technical_indicator definitions) ...
        {
            "type": "function",
            "function": {
                "name": "get_stock_prices",
                "description": "Get historical stock data for a given ticker symbol, period, and interval.",
                "parameters": { "type": "object", "properties": {
                        "ticker_symbol": {"type": "string", "description": "The stock ticker symbol (e.g., 'AAPL')."},
                        "period": {"type": "string", "description": "The period for which to fetch data (e.g., '1mo', '1y'). Default is '1mo'."},
                        "interval": {"type": "string", "description": "The interval of data points (e.g., '1d', '1wk'). Default is '1d'."}
                    }, "required": ["ticker_symbol"] }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_technical_indicator",
                "description": "Calculates RSI, Stochastic Oscillator, ROC, or MFI for a stock.",
                "parameters": { "type": "object", "properties": {
                        "ticker_symbol": {"type": "string", "description": "The stock ticker symbol."},
                        "indicator_type": {"type": "string", "description": "Supported: 'RSI', 'StochasticOscillator', 'ROC', 'MFI'."},
                        "data_period": {"type": "string", "description": "Data period (e.g., '6mo', '1y'). Default: '6mo'."},
                        "data_interval": {"type": "string", "description": "Data interval (e.g., '1d', '1wk'). Default: '1d'."},
                        "window": {"type": "integer", "description": "Calculation window. Default: 14."},
                        "smooth_window": {"type": "integer", "description": "Smoothing window (for Stochastic Oscillator). Default: 3."}
                    }, "required": ["ticker_symbol", "indicator_type"] }
            }
        },
        {
            "type": "function", "function": {
                "name": "get_standard_deviation", "description": "Calculates the annualized standard deviation (volatility) of a stock's returns.",
                "parameters": { "type": "object", "properties": {
                        "ticker_symbol": {"type": "string", "description": "The stock ticker symbol."},
                        "period": {"type": "string", "description": "Data period (e.g., '1y', '60d')."}
                    }, "required": ["ticker_symbol", "period"] }
            }
        },
        {
            "type": "function", "function": {
                "name": "get_max_drawdown", "description": "Calculates the maximum drawdown of a stock's returns.",
                "parameters": { "type": "object", "properties": {
                        "ticker_symbol": {"type": "string", "description": "The stock ticker symbol."},
                        "period": {"type": "string", "description": "Data period (e.g., '1y', '60d')."}
                    }, "required": ["ticker_symbol", "period"] }
            }
        },
        {
            "type": "function", "function": { # Removed benchmark_ticker from user-facing params
                "name": "get_sharpe_ratio", "description": "Calculates the Sharpe ratio of a stock's returns.",
                "parameters": { "type": "object", "properties": {
                        "ticker_symbol": {"type": "string", "description": "The stock ticker symbol."},
                        "period": {"type": "string", "description": "Data period (e.g., '1y', '60d')."}
                    }, "required": ["ticker_symbol", "period"] }
            }
        },
        {
            "type": "function", "function": {
                "name": "get_alpha_beta", "description": "Calculates Alpha and Beta of a stock relative to a benchmark (default SPY).",
                "parameters": { "type": "object", "properties": {
                        "ticker_symbol": {"type": "string", "description": "The stock ticker symbol."},
                        "period": {"type": "string", "description": "Data period (e.g., '1y', '60d')."},
                        "benchmark_ticker": {"type": "string", "description": "Benchmark ticker symbol. Default: 'SPY'."}
                    }, "required": ["ticker_symbol", "period"] }
            }
        },
        {
            "type": "function", "function": {
                "name": "get_value_at_risk", "description": "Calculates the Value at Risk (VaR) of a stock at a given confidence level.",
                "parameters": { "type": "object", "properties": {
                        "ticker_symbol": {"type": "string", "description": "The stock ticker symbol."},
                        "period": {"type": "string", "description": "Data period (e.g., '1y', '60d')."},
                        "confidence_level": {"type": "number", "description": "Confidence level for VaR (e.g., 0.05 for 5%). Default: 0.05."}
                    }, "required": ["ticker_symbol", "period"] }
            }
        }
    ]
    # ... (rest of ChatApplication as before, __init__ will change) ...
    DEFAULT_OPENAI_API_KEY = "lm-studio"
    DEFAULT_OPENAI_BASE_URL = "http://localhost:1234/v1"
    DEFAULT_MODEL_NAME = "gemma-3-12b-it-qat"

    def __init__(self, 
                 openai_api_key: str = os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY), 
                 openai_base_url: str = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL), 
                 model_name: str = DEFAULT_MODEL_NAME):
        
        self.client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)
        self.model = model_name
        
        stock_provider = StockDataProvider()
        indicator_calculator = TechnicalIndicatorCalculator(stock_provider)
        risk_calculator = FinancialRiskCalculator(stock_provider, risk_free_rate=0.01) # Using default 0.01 annual RF rate
        
        self.tool_processor = ToolProcessor(
            stock_provider, 
            indicator_calculator, 
            risk_calculator, 
            self.client, 
            self.model
        )
    
    def start_chat(self):
        # ... (start_chat implementation as before, truncated for brevity)
        overall_chat_history = [
            {
                "role": "system",
                "content": "You are an expert financial consultant with deep knowledge of the stock market, financial analysis, and risk assessment. Your goal is to provide insightful advice to users regarding their financial decisions, particularly in relation to stocks. When a user asks for information or advice, you should:\n- Leverage your understanding of market trends, financial metrics, technical indicators, and risk assessment.\n- If you use tools to fetch data (like stock prices or calculate metrics), incorporate this data into your analysis.\n- Clearly explain the reasoning behind your advice, including any potential risks or alternative viewpoints.\n- If the user asks about specific orders or non-financial topics, you can politely state that your expertise is in financial consultancy and stock market analysis. Use the available tools to provide financial data and analysis when requested."
            }
        ]
        print("Assistant: Hello! I am your expert financial consultant. I can help you with stock market analysis, technical indicators, and financial risk assessment. How can I assist you today?")
        print("(Type 'quit' to exit)")
        # ... (rest of loop) ...
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "quit":
                print("Assistant: Goodbye!")
                break
            overall_chat_history.append({"role": "user", "content": user_input})
            current_turn_messages = list(overall_chat_history)
            reasoning_summary_log = []
            MAX_ITERATIONS = 100 
            current_answer = "" 
            critique_text = "" # Ensure critique_text is defined before potential use in loop

            for iteration in range(MAX_ITERATIONS):
                reasoning_summary_log.append(f"Iteration {iteration + 1}")
                messages_for_llm_call = []
                system_message = next((m for m in overall_chat_history if m['role'] == 'system'), None)
                if system_message:
                    messages_for_llm_call.append(system_message)
                messages_for_llm_call.append({"role": "user", "content": user_input})

                if iteration > 0:
                    messages_for_llm_call.append({"role": "assistant", "content": current_answer})
                    messages_for_llm_call.append({"role": "user", "content": f"I have reviewed my previous answer. Critique: {critique_text}. Please provide a new, refined answer..."}) 

                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages_for_llm_call, 
                        tools=self.TOOLS_DEFINITION, 
                    )
                    
                    if response.choices[0].message.tool_calls:
                        reasoning_summary_log.append("Assistant requests tool call(s).")
                        processed_answer_content, messages_during_tool_processing = self.tool_processor.process_tool_calls(response, list(messages_for_llm_call))
                        current_answer = processed_answer_content
                        reasoning_summary_log.append(f"Tool(s) used. Answer after tool use: {current_answer}")
                        current_turn_messages = messages_during_tool_processing
                    else:
                        current_answer = response.choices[0].message.content
                        reasoning_summary_log.append(f"Attempted Answer (no tools): {current_answer}")
                except Exception as e:
                    print(f"\nError during LLM call in reasoning loop: {str(e)}")
                    reasoning_summary_log.append(f"Error: {str(e)}")
                    break 

                reasoning_summary_log.append("Critique Step: [Placeholder - critique logic to be implemented]")
                reasoning_summary_log.append("Satisfaction Check: [Placeholder - satisfaction logic to be implemented]")

                if current_answer and current_answer.strip().endswith("?"):
                    reasoning_summary_log.append(f"Assistant generated a question for the user: {current_answer}")
                    print(f"\nAssistant (clarification): {current_answer}")
                    user_clarification_input = input("\nYou (clarification): ").strip()
                    overall_chat_history.append({"role": "assistant", "content": current_answer})
                    overall_chat_history.append({"role": "user", "content": user_clarification_input})
                    current_turn_messages.append({"role": "assistant", "content": current_answer})
                    current_turn_messages.append({"role": "user", "content": user_clarification_input})
                    reasoning_summary_log.append(f"User provided clarification: {user_clarification_input}")
                elif current_answer:
                     current_turn_messages.append({"role": "assistant", "content": current_answer})

                critique_prompt_messages = [
                    {"role": "system", "content": "You are an AI assistant evaluating your own previous answer..."},
                    {"role": "user", "content": f"The original user query was: {user_input}"},
                    {"role": "assistant", "content": f"The answer I generated is: {current_answer}"},
                    {"role": "user", "content": "Now, critically evaluate this answer..."}
                ]
                try:
                    critique_response = self.client.chat.completions.create(model=self.model, messages=critique_prompt_messages)
                    critique_text = critique_response.choices[0].message.content
                    reasoning_summary_log.append(f"Critique: {critique_text}")
                except Exception as e:
                    critique_text = f"Error during critique: {e}"
                    reasoning_summary_log.append(critique_text)
                
                internal_critique_message_for_context = {"role": "user", "content": f"Internal Critique of my last answer: {critique_text}"}
                current_turn_messages.append(internal_critique_message_for_context)

                if critique_text and critique_text.strip().endswith("?"):
                    reasoning_summary_log.append(f"Critique seems to be a question for the user: {critique_text}")
                    # ... 

                satisfaction_check_messages = [
                    {"role": "user", "content": f"My self-critique of this answer was: {critique_text}"},
                    {"role": "user", "content": "Considering the critique, is the current answer now sufficiently accurate... Respond with first 'YES' or 'NO'..."}
                ]
                try:
                    satisfaction_response = self.client.chat.completions.create(model=self.model, messages=satisfaction_check_messages)
                    satisfaction_decision_text = satisfaction_response.choices[0].message.content.strip().upper()
                    reasoning_summary_log.append(f"Satisfaction Check Response: {satisfaction_decision_text}")
                except Exception as e:
                    satisfaction_decision_text = f"ERROR DURING SATISFACTION CHECK: {e}"
                    reasoning_summary_log.append(satisfaction_decision_text)
                    satisfaction_decision_text = "NO"

                if satisfaction_decision_text.startswith("YES"):
                    reasoning_summary_log.append("Assistant is satisfied with the answer. Terminating loop.")
                    break
                else:
                    reasoning_summary_log.append("Assistant is not satisfied. Continuing refinement if iterations allow.")
                    current_turn_messages.append({"role": "user", "content": f"Satisfaction Assessment: {satisfaction_decision_text}. Further refinement needed."})
            
            if iteration == MAX_ITERATIONS - 1:
                reasoning_summary_log.append(f"NOTE: Reasoning process completed all {MAX_ITERATIONS} iterations.")
            final_response_content = f"Final Answer:\n{current_answer}\n\nReasoning Summary:\n" + "\n".join(reasoning_summary_log)
            print(f"\nAssistant: {final_response_content}")
            overall_chat_history.append({"role": "assistant", "content": final_response_content})


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
    expected_sharpe = empyrical.sharpe_ratio(mock_stock_returns, risk_free=MOCK_RISK_FREE_RATE/252, annualization=252)
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
