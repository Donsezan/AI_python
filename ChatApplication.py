# --- ChatApplication Class (TOOLS_DEFINITION and __init__ are key changes) ---
import os
from openai import OpenAI

from ToolProcessor import ToolProcessor
from FinancialRiskCalculator import FinancialRiskCalculator
from StockDataProvider import StockDataProvider
from TechnicalIndicatorCalculator import TechnicalIndicatorCalculator


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

