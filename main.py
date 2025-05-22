import json
import random
from urllib.parse import urlparse
import webbrowser
from datetime import datetime, timedelta
import os
from openai import OpenAI
import yfinance as yf # Make sure to install yfinance: pip install yfinance
import pandas as pd
import ta # Make sure to install pandas and ta: pip install pandas ta
from unittest.mock import patch, MagicMock

# Indicator Constants
INDICATOR_RSI = "RSI"
INDICATOR_STOCHASTIC_OSCILLATOR = "StochasticOscillator"
INDICATOR_ROC = "ROC"
INDICATOR_MFI = "MFI"
SUPPORTED_INDICATORS = [INDICATOR_RSI, INDICATOR_STOCHASTIC_OSCILLATOR, INDICATOR_ROC, INDICATOR_MFI]

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = "qwen3-4b"

def get_stock_prices(ticker_symbol: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    """Get historical stock data for a single ticker symbol."""
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        hist = ticker_data.history(period=period, interval=interval)
        if not hist.empty:
            print(f"Successfully fetched historical data for {ticker_symbol}. Shape: {hist.shape}", flush=True)
            return hist
        else:
            print(f"No historical data found for {ticker_symbol} for the period {period} and interval {interval}.", flush=True)
            return pd.DataFrame()
    except Exception as e:
        print(f"Could not retrieve historical data for {ticker_symbol}: {e}", flush=True)
        return pd.DataFrame()

def calculate_technical_indicator(ticker_symbol: str, indicator_type: str, data_period: str = "6mo", data_interval: str = "1d", window: int = 14, smooth_window: int = 3) -> dict:
    """Calculates a specified technical indicator for a given stock ticker."""
    if indicator_type not in SUPPORTED_INDICATORS:
        return {"error": "Unsupported indicator type"}

    df = get_stock_prices(ticker_symbol, period=data_period, interval=data_interval)

    if df.empty:
        return {"error": "Could not retrieve historical data"}

    # Basic data cleaning
    required_columns_for_mfi = ['High', 'Low', 'Close', 'Volume']
    if indicator_type == INDICATOR_MFI and not all(col in df.columns for col in required_columns_for_mfi):
        return {"error": "Dataframe missing required HLCV columns for MFI"}
    
    required_columns_for_stoch = ['High', 'Low', 'Close']
    if indicator_type == INDICATOR_STOCHASTIC_OSCILLATOR and not all(col in df.columns for col in required_columns_for_stoch):
        return {"error": "Dataframe missing required HLC columns for Stochastic Oscillator"}

    # Ensure 'Close' column exists for RSI and ROC
    if indicator_type in [INDICATOR_RSI, INDICATOR_ROC] and 'Close' not in df.columns:
        return {"error": "Dataframe missing required Close column"}
        
    df.dropna(inplace=True) # Drop rows with NaN values that can affect calculations

    result = {}
    try:
        if indicator_type == INDICATOR_RSI:
            indicator = ta.momentum.RSIIndicator(close=df['Close'], window=window)
            rsi_series = indicator.rsi()
            result = {"rsi": rsi_series.iloc[-1] if rsi_series is not None and not rsi_series.empty and len(rsi_series) > 0 else "Not enough data"}
        elif indicator_type == INDICATOR_STOCHASTIC_OSCILLATOR:
            indicator = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=window, smooth_window=smooth_window)
            stoch_k_series = indicator.stoch()
            stoch_d_series = indicator.stoch_signal()
            result = {
                "stoch_k": stoch_k_series.iloc[-1] if stoch_k_series is not None and not stoch_k_series.empty and len(stoch_k_series) > 0 else "Not enough data",
                "stoch_d": stoch_d_series.iloc[-1] if stoch_d_series is not None and not stoch_d_series.empty and len(stoch_d_series) > 0 else "Not enough data"
            }
        elif indicator_type == INDICATOR_ROC:
            indicator = ta.momentum.ROCIndicator(close=df['Close'], window=window)
            roc_series = indicator.roc()
            result = {"roc": roc_series.iloc[-1] if roc_series is not None and not roc_series.empty and len(roc_series) > 0 else "Not enough data"}
        elif indicator_type == INDICATOR_MFI:
            indicator = ta.volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=window)
            mfi_series = indicator.money_flow_index()
            result = {"mfi": mfi_series.iloc[-1] if mfi_series is not None and not mfi_series.empty and len(mfi_series) > 0 else "Not enough data"}
        else:
            return {"error": "Unknown indicator calculation error"} # Should be caught by initial validation
    except IndexError: # Handles cases where iloc[-1] fails due to insufficient data for window
        return {"error": f"Not enough data to calculate {indicator_type} for the given window {window}"}
    except Exception as e:
        print(f"Error calculating {indicator_type} for {ticker_symbol}: {e}", flush=True)
        return {"error": f"An error occurred during {indicator_type} calculation: {str(e)}"}
        
    return result

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_prices",
            "description": "Get historical stock data for a given ticker symbol, period, and interval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL')."
                    },
                    "period": {
                        "type": "string",
                        "description": "The period for which to fetch data (e.g., '1mo', '3mo', '1y'). Default is '1mo'."
                    },
                    "interval": {
                        "type": "string",
                        "description": "The interval of data points (e.g., '1d', '1wk', '1mo'). Default is '1d'."
                    }
                },
                "required": ["ticker_symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_technical_indicator",
            "description": "Calculates a specified technical indicator (RSI, StochasticOscillator, ROC, MFI) for a stock ticker using historical data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., 'AAPL')."
                    },
                    "indicator_type": {
                        "type": "string",
                        "description": "The type of technical indicator to calculate. Supported values: 'RSI', 'StochasticOscillator', 'ROC', 'MFI'."
                    },
                    "data_period": {
                        "type": "string",
                        "description": "The period for fetching historical data (e.g., '6mo', '1y'). Default: '6mo'."
                    },
                    "data_interval": {
                        "type": "string",
                        "description": "The interval for historical data points (e.g., '1d', '1wk'). Default: '1d'."
                    },
                    "window": {
                        "type": "integer",
                        "description": "The calculation window for the indicator (e.g., 14 for RSI). Default: 14."
                    },
                    "smooth_window": {
                        "type": "integer",
                        "description": "The smoothing window, primarily for Stochastic Oscillator's %D line. Default: 3."
                    }
                },
                "required": ["ticker_symbol", "indicator_type"]
            }
        }
    }
]


def process_tool_calls(response, messages):
    """Process multiple tool calls and return the final response and updated messages"""
    # Get all tool calls from the response
    tool_calls = response.choices[0].message.tool_calls

    # Create the assistant message with tool calls
    assistant_tool_call_message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": tool_call.function,
            }
            for tool_call in tool_calls
        ],
    }

    # Add the assistant's tool call message to the history
    messages.append(assistant_tool_call_message)

    # Process each tool call and collect results
    tool_results = []
    for tool_call in tool_calls:
        # For functions with no arguments, use empty dict
        arguments = (
            json.loads(tool_call.function.arguments)
            if tool_call.function.arguments.strip()
            else {}
        )


# Define a mapping of function names to their corresponding functions
    function_mapping = {
        "get_stock_prices": lambda args: get_stock_prices(
            args["ticker_symbol"],
            args.get("period", "1mo"),  # Use .get for optional args with defaults
            args.get("interval", "1d")
        ),
        "calculate_technical_indicator": lambda args: calculate_technical_indicator(
            args["ticker_symbol"],
            args["indicator_type"],
            args.get("data_period", "6mo"),
            args.get("data_interval", "1d"),
            args.get("window", 14),
            args.get("smooth_window", 3)
        )
    }

    # Determine which function to call based on the tool call name
    if tool_call.function.name in function_mapping:
        result = function_mapping[tool_call.function.name](arguments)
    else:
        # Skip processing this tool call if the function name is not in the mapping
        print(f"Unknown function name: {tool_call.function.name}", flush=True)
       

        #######

        # Add the result message
        tool_result_message = {
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id,
        }
        tool_results.append(tool_result_message)
        messages.append(tool_result_message)

    # Get the final response
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    return final_response


def chat():
    messages = [
        {
            "role": "system",
            "content": "You are an expert financial consultant with deep knowledge of the stock market. Your goal is to provide insightful advice to users regarding their financial decisions, particularly in relation to stocks. When a user asks for information or advice, you should:\n- Leverage your understanding of market trends, financial metrics, and risk assessment.\n- If you use tools to fetch data (like stock prices), incorporate this data into your analysis.\n- Clearly explain the reasoning behind your advice, including any potential risks or alternative viewpoints.\n- If the user asks about specific orders or non-financial topics, you can politely state that your expertise is in financial consultancy and stock market analysis. Use the available tools to provide financial data when requested."
        }
    ]

    print(
        "Assistant: Hello! I am your expert financial consultant. I can help you with stock market analysis and financial advice. How can I assist you today?"
    )
    print("(Type 'quit' to exit)")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for quit command
        if user_input.lower() == "quit":
            print("Assistant: Goodbye!")
            break

        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})

        try:
            # Get initial response
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )

            # Check if the response includes tool calls
            if response.choices[0].message.tool_calls:
                # Process all tool calls and get final response
                final_response = process_tool_calls(response, messages)
                print("\nAssistant:", final_response.choices[0].message.content)

                # Add assistant's final response to messages
                messages.append(
                    {
                        "role": "assistant",
                        "content": final_response.choices[0].message.content,
                    }
                )
            else:
                # If no tool call, just print the response
                print("\nAssistant:", response.choices[0].message.content)

                # Add assistant's response to messages
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            exit(1)


if __name__ == "__main__":
    # Simple test for the modified get_stock_prices
    # test_df_aapl = get_stock_prices("AAPL", period="1wk")
    # print("\nTest - AAPL weekly data for 1 week:")
    # print(test_df_aapl)

    # test_df_invalid = get_stock_prices("INVALIDTICKER")
    # print("\nTest - Invalid ticker data:")
    # print(test_df_invalid)
    # test_get_stock_prices() # Commented out previous test for get_stock_prices
    test_calculate_technical_indicators() # New test call
    chat()

