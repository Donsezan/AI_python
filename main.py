import json
import random
from urllib.parse import urlparse
import webbrowser
from datetime import datetime, timedelta
import os
from openai import OpenAI
import yfinance as yf # Make sure to install yfinance: pip install yfinance
from unittest.mock import patch, MagicMock

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = "qwen3-4b"

orders = ["1045", "1046", "1047", "1048", "1049", "1050", "1051", "1052", "1053", "1054", "1055"]

def is_valid_order(order_id: int) -> bool:
    """Check if the order is valid"""
    return {"is_valid": order_id in orders}

def get_delivery_date(order_id: str) -> datetime:
    # Generate a random delivery date between today and 14 days from now
    # in a real-world scenario, this function would query a database or API
    today = datetime.now()
    random_days = random.randint(1, 14)
    delivery_date = today + timedelta(days=random_days)
    print(
        f"\nget_delivery_date function returns delivery date:\n\n{delivery_date}",
        flush=True,
    )
    return {"delivery_date": delivery_date.isoformat()}

def get_order_status(order_id: str) -> str:
    # Simulate getting order status from a database or API
    # In a real-world scenario, this function would query a database or API
    statuses = ["Processing", "Shipped", "Delivered", "Cancelled"]
    status = random.choice(statuses)
    print(f"\nget_order_status function returns order status:\n\n{status}", flush=True)
    return {"status": status}

def get_stock_prices(tickers: list[str]) -> dict:
    """Get the current stock prices for a list of tickers."""
    stock_prices = {}
    for ticker_symbol in tickers:
        try:
            ticker_data = yf.Ticker(ticker_symbol)
            # Fetching historical data for the most recent trading day
            hist = ticker_data.history(period="1d")
            if not hist.empty and 'Close' in hist:
                # Using the closing price of the most recent day
                stock_prices[ticker_symbol] = hist['Close'].iloc[-1]
            elif 'currentPrice' in ticker_data.info:
                stock_prices[ticker_symbol] = ticker_data.info['currentPrice']
            elif 'regularMarketPrice' in ticker_data.info:
                stock_prices[ticker_symbol] = ticker_data.info['regularMarketPrice']
            elif 'previousClose' in ticker_data.info: # Fallback to previous close
                stock_prices[ticker_symbol] = ticker_data.info['previousClose']
            else:
                stock_prices[ticker_symbol] = "Price not found"
        except Exception as e:
            print(f"Error fetching price for {ticker_symbol}: {e}", flush=True)
            stock_prices[ticker_symbol] = "Price not found"
    print(f"\nget_stock_prices function returns:\n\n{stock_prices}", flush=True)
    return stock_prices

tools = [
    {
        "type": "function",
        "function": {
            "name": "is_valid_order",
            "description": "Check if the order ID is valid",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID to check",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_delivery_date",
            "description": "Get the estimated delivery date for an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID to check",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status of an order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID to check",
                    },
                },
                "required": ["order_id"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_prices",
            "description": "Get the current stock price for a list of tickers using Yahoo Finance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of stock ticker symbols (e.g., ['AAPL', 'MSFT'])."
                    }
                },
                "required": ["tickers"]
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
        "is_valid_order": lambda args: is_valid_order(args["order_id"]),
        "get_delivery_date": lambda args: get_delivery_date(args["order_id"]),
        "get_order_status": lambda args: get_order_status(args["order_id"]),
        "get_stock_prices": lambda args: get_stock_prices(args["tickers"])
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
    chat()

