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
model = "gemma-3-12b-it-qat"

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


def process_tool_calls(response_with_tool_calls, current_call_context_messages):
    """
    Processes tool calls requested by the LLM, executes them, and gets a final summary from the LLM.
    Returns the final summary content and the sequence of messages generated during this process.
    """
    tool_calls = response_with_tool_calls.choices[0].message.tool_calls
    
    # Make a copy of the context to append new messages related to tool processing
    messages_for_tool_sequence = list(current_call_context_messages)

    # Append the assistant's message that requested the tool calls
    # This is crucial for the LLM to understand why it's seeing tool results later
    assistant_tool_call_request_message = response_with_tool_calls.choices[0].message
    messages_for_tool_sequence.append(assistant_tool_call_request_message)

    function_mapping = {
        "get_stock_prices": lambda args: get_stock_prices(
            args["ticker_symbol"],
            args.get("period", "1mo"),
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

    for tool_call in tool_calls:
        arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments.strip() else {}
        
        result_content = ""
        if tool_call.function.name in function_mapping:
            try:
                # For get_stock_prices, the result is a DataFrame. We need to convert it.
                if tool_call.function.name == "get_stock_prices":
                    df_result = function_mapping[tool_call.function.name](arguments)
                    if not df_result.empty:
                        result_content = df_result.reset_index().to_json(orient='records', date_format='iso')
                    else:
                        result_content = "No data found or error fetching data."
                else: # For other functions returning dict
                     result_content_dict = function_mapping[tool_call.function.name](arguments)
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

    # Get the final response from the LLM after tools have been called and results appended
    final_response_after_tools = client.chat.completions.create(
        model=model,
        messages=messages_for_tool_sequence,
        # No tools parameter here, as we expect a textual summary now
    )
    final_content = final_response_after_tools.choices[0].message.content
    
    # Append the LLM's summary message to the sequence
    messages_for_tool_sequence.append({"role": "assistant", "content": final_content})
    
    return final_content, messages_for_tool_sequence

def test_get_stock_prices_json_format():
    """Tests the JSON output format of get_stock_prices, simulating process_tool_calls."""
    print("Running test_get_stock_prices_json_format...", flush=True)

    # Test case 1: Successful data fetch and JSON conversion
    @patch('main.yf.Ticker') # Patching yf.Ticker within the main module
    def test_successful_fetch(mock_yf_ticker):
        print("  Running test_successful_fetch...", flush=True)
        # Prepare mock DataFrame
        data = {'Open': [150.0, 151.0], 'High': [152.0, 151.5], 'Low': [149.0, 150.0], 'Close': [151.5, 150.5], 'Volume': [100000, 120000]}
        # Ensure UTC timezone for direct ISO conversion with timezone offset
        index = pd.to_datetime(['2023-01-01T00:00:00', '2023-01-02T00:00:00'], utc=True)
        mock_df = pd.DataFrame(data, index=index)
        mock_df.index.name = 'Date' # Naming the index is important for it to be included in 'records' orientation

        # Configure the mock
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_yf_ticker.return_value = mock_ticker_instance

        # Call the function being tested
        df_result = get_stock_prices("TEST_SUCCESS_TICKER", period="2d", interval="1d")

        assert not df_result.empty, "DataFrame result should not be empty for successful fetch"

        # Simulate the JSON conversion, ensuring index is included as per test spec
        json_output_string = df_result.reset_index().to_json(orient='records', date_format='iso')
        
        # Parse and assert
        parsed_json_list = json.loads(json_output_string)
        
        assert isinstance(parsed_json_list, list), "Parsed JSON should be a list"
        assert len(parsed_json_list) == 2, "Parsed JSON list should have 2 records"
        
        first_record_dict = parsed_json_list[0]
        assert isinstance(first_record_dict, dict), "Each item in parsed JSON list should be a dict"
        
        # Expected keys: DataFrame columns + named index
        expected_keys = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] 
        assert all(key in first_record_dict for key in expected_keys), f"Record keys do not match. Expected: {expected_keys}, Got: {list(first_record_dict.keys())}"
        assert len(first_record_dict.keys()) == len(expected_keys), "Number of keys in record is incorrect."

        # Check date format (ISO 8601)
        # pd.to_datetime with utc=True creates timezone-aware datetimes.
        # to_json with date_format='iso' and UTC index produces 'Z' for Zulu time.
        assert first_record_dict['Date'] == '2023-01-01T00:00:00.000Z', f"Date format incorrect. Expected ISO format with Z (UTC). Got: {first_record_dict['Date']}"
        assert first_record_dict['Open'] == 150.0

        print("  Test case 1 (successful fetch) PASSED", flush=True)

    test_successful_fetch()

    # Test case 2: Empty DataFrame response
    @patch('main.yf.Ticker') # Patching yf.Ticker within the main module
    def test_empty_fetch(mock_yf_ticker):
        print("  Running test_empty_fetch...", flush=True)
        # Configure the mock to return an empty DataFrame
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame() # Empty DataFrame
        mock_yf_ticker.return_value = mock_ticker_instance

        # Call the function
        df_result = get_stock_prices("TEST_EMPTY_TICKER", period="1d", interval="1d")

        assert df_result.empty, "DataFrame result should be empty for this test case"

        # Simulate JSON conversion for an empty DataFrame
        json_output_string = df_result.to_json(orient='records', date_format='iso')

        # Assert
        assert json_output_string == '[]', f"Expected '[]' for empty DataFrame, got {json_output_string}"
        print("  Test case 2 (empty fetch) PASSED", flush=True)

    test_empty_fetch()
    print("test_get_stock_prices_json_format completed successfully.", flush=True)


def chat():
    overall_chat_history = [
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
        overall_chat_history.append({"role": "user", "content": user_input})

        current_turn_messages = list(overall_chat_history)
        reasoning_summary_log = []
        MAX_ITERATIONS = 100 # Max iterations for the reasoning loop for a single user query
        current_answer = "" # To store the evolving answer from the LLM

        for iteration in range(MAX_ITERATIONS):
            reasoning_summary_log.append(f"Iteration {iteration + 1}")

            # --- Answer Refinement Mechanism: Prepare messages for LLM ---
            messages_for_llm_call = []
            system_message = next((m for m in overall_chat_history if m['role'] == 'system'), None)
            if system_message:
                messages_for_llm_call.append(system_message)
            
            messages_for_llm_call.append({"role": "user", "content": user_input}) # Original user query

            if iteration > 0:
                # This is a refinement iteration
                messages_for_llm_call.append({"role": "assistant", "content": current_answer}) # Previous answer
                # critique_text is from the critique step of the current iteration, critiquing the 'current_answer' from previous iteration
                messages_for_llm_call.append({"role": "user", "content": f"I have reviewed my previous answer. Critique: {critique_text}. Please provide a new, refined answer based on this critique to better address the original query: {user_input}"})
            # For iteration == 0, messages_for_llm_call will just be [system_prompt, user_input]
            # --- End Answer Refinement Mechanism ---

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages_for_llm_call, 
                    tools=tools, 
                )
                
                if response.choices[0].message.tool_calls:
                    reasoning_summary_log.append("Assistant requests tool call(s).")
                    # messages_for_llm_call already contains the history that led to this tool call request.
                    # The 'response' object contains the tool call request itself.
                    
                    processed_answer_content, messages_during_tool_processing = process_tool_calls(response, list(messages_for_llm_call))
                    
                    current_answer = processed_answer_content
                    reasoning_summary_log.append(f"Tool(s) used. Answer after tool use: {current_answer}")
                    
                    current_turn_messages = messages_during_tool_processing
                    
                else: # No tool calls
                    current_answer = response.choices[0].message.content
                    reasoning_summary_log.append(f"Attempted Answer (no tools): {current_answer}")
                    # Append the assistant's direct answer to current_turn_messages
                    # This is done *after* the clarification check below for current_answer
                    # current_turn_messages.append({"role": "assistant", "content": current_answer}) # Moved below

            except Exception as e:
                print(f"\nError during LLM call in reasoning loop: {str(e)}")
                reasoning_summary_log.append(f"Error: {str(e)}")
                break # Exit reasoning loop on error

            # Critique Step (Placeholder)
            reasoning_summary_log.append("Critique Step: [Placeholder - critique logic to be implemented]")
            # TODO: Call LLM for critique. Update current_turn_messages with critique.
            # Example: current_turn_messages.append({"role": "user", "content": "Critique: The answer is too vague."})

            # Satisfaction Check (Placeholder)
            reasoning_summary_log.append("Satisfaction Check: [Placeholder - satisfaction logic to be implemented]")
            # TODO: Call LLM for satisfaction. If satisfied, break.
            # Example: if is_satisfied(current_answer): break

            # --- Handling questions generated as current_answer ---
            if current_answer and current_answer.strip().endswith("?"):
                reasoning_summary_log.append(f"Assistant generated a question for the user: {current_answer}")
                print(f"\nAssistant (clarification): {current_answer}")
                
                user_clarification_input = input("\nYou (clarification): ").strip()
                overall_chat_history.append({"role": "assistant", "content": current_answer}) # Log assistant's question
                overall_chat_history.append({"role": "user", "content": user_clarification_input}) # Log user's answer
                
                # Update current_turn_messages for the ongoing reasoning process
                current_turn_messages.append({"role": "assistant", "content": current_answer}) # Assistant's question
                current_turn_messages.append({"role": "user", "content": user_clarification_input}) # User's answer to clarification
                
                reasoning_summary_log.append(f"User provided clarification: {user_clarification_input}")
                # The loop will proceed to critique. Critique will see the question and user's answer.
            elif current_answer: # Only append if it's not a question (questions are appended above with user response)
                # This handles the case where current_answer was from a tool call summary or direct LLM response (not a question)
                 current_turn_messages.append({"role": "assistant", "content": current_answer})


            # --- Internal Critique Mechanism ---
            critique_prompt_messages = [
                {"role": "system", "content": "You are an AI assistant evaluating your own previous answer. Be critical and thorough."},
                {"role": "user", "content": f"The original user query was: {user_input}"},
                {"role": "assistant", "content": f"The answer I generated is: {current_answer}"},
                {"role": "user", "content": "Now, critically evaluate this answer based on the following questions: 1. Is the answer correct and relevant to the original query? 2. How confident am I in this answer's accuracy and completeness? 3. What are the potential flaws, biases, or missing information in this answer? 4. How can this answer be improved or made more certain? 5. Does it directly address all parts of the user's query?"}
            ]
            try:
                critique_response = client.chat.completions.create(
                    model=model,
                    messages=critique_prompt_messages
                )
                critique_text = critique_response.choices[0].message.content
                reasoning_summary_log.append(f"Critique: {critique_text}")
            except Exception as e:
                critique_text = f"Error during critique: {e}"
                reasoning_summary_log.append(critique_text)
                # Continue and log error, as per instructions
            
            # --- Handling questions generated as critique_text & appending critique to current_turn_messages ---
            # Store the original critique message for context in refinement
            internal_critique_message_for_context = {"role": "user", "content": f"Internal Critique of my last answer: {critique_text}"}
            current_turn_messages.append(internal_critique_message_for_context) # Add the critique itself

            if critique_text and critique_text.strip().endswith("?"):
                reasoning_summary_log.append(f"Critique seems to be a question for the user: {critique_text}")
                print(f"\nAssistant (clarification from critique): {critique_text}")
                
                user_clarification_for_critique = input("\nYou (clarification for critique-question): ").strip()
                # Log to overall history
                overall_chat_history.append({"role": "assistant", "content": critique_text}) 
                overall_chat_history.append({"role": "user", "content": user_clarification_for_critique})
                
                # Add user's clarification to current_turn_messages *after* the critique itself
                current_turn_messages.append({"role": "user", "content": user_clarification_for_critique})
                reasoning_summary_log.append(f"User provided clarification to critique-question: {user_clarification_for_critique}")
            # --- End Internal Critique Mechanism & handling critique-questions---

            # --- Loop Termination (Satisfaction Check) ---
            satisfaction_check_messages = [
                #{"role": "system", "content": "You are an AI assistant evaluating if your last answer is now satisfactory after self-critique, or if further refinement is strictly necessary."},
                #{"role": "user", "content": f"The original user query was: {user_input}"},
                #{"role": "assistant", "content": f"My current answer is: {current_answer}"},
                {"role": "user", "content": f"My self-critique of this answer was: {critique_text}"}, # critique_text is available from the previous step
                {"role": "user", "content": "Considering the critique, is the current answer now sufficiently accurate, complete, and directly addresses the user's query? Respond with first 'YES' if no further refinement is essential, or 'NO' if significant improvements are still needed. If 'NO', briefly state what key improvement is still required."}
            ]
            try:
                satisfaction_response = client.chat.completions.create(
                    model=model,
                    messages=satisfaction_check_messages
                )
                satisfaction_decision_text = satisfaction_response.choices[0].message.content.strip().upper()
                reasoning_summary_log.append(f"Satisfaction Check Response: {satisfaction_decision_text}")
            except Exception as e:
                satisfaction_decision_text = f"ERROR DURING SATISFACTION CHECK: {e}"
                reasoning_summary_log.append(satisfaction_decision_text)
                satisfaction_decision_text = "NO" # Default to not satisfied on error

            if satisfaction_decision_text.startswith("YES"):
                reasoning_summary_log.append("Assistant is satisfied with the answer. Terminating loop.")
                break # Exit the reasoning loop
            else: # "NO" or other, including error during check
                reasoning_summary_log.append("Assistant is not satisfied. Continuing refinement if iterations allow.")
                # Add the satisfaction assessment to current_turn_messages to guide the next refinement
                current_turn_messages.append({"role": "user", "content": f"Satisfaction Assessment: {satisfaction_decision_text}. Further refinement needed."})
            # --- End Loop Termination (Satisfaction Check) ---
        
        # After Reasoning Loop
        # Check if the loop completed all its iterations
        if iteration == MAX_ITERATIONS - 1:
            reasoning_summary_log.append(f"NOTE: Reasoning process completed all {MAX_ITERATIONS} iterations.")
            
        final_response_content = f"Final Answer:\n{current_answer}\n\nReasoning Summary:\n" + "\n".join(reasoning_summary_log)
        
        print(f"\nAssistant: {final_response_content}")
        overall_chat_history.append({"role": "assistant", "content": final_response_content})


if __name__ == "__main__":
    # Simple test for the modified get_stock_prices
    # test_df_aapl = get_stock_prices("AAPL", period="1wk")
    # print("\nTest - AAPL weekly data for 1 week:")
    # print(test_df_aapl)

    # test_df_invalid = get_stock_prices("INVALIDTICKER")
    # print("\nTest - Invalid ticker data:")
    # print(test_df_invalid)
    # test_get_stock_prices() # Commented out previous test for get_stock_prices
    #test_calculate_technical_indicators() # New test call
    test_get_stock_prices_json_format()
    chat()

