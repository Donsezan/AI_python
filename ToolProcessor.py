# --- ToolProcessor Class (remains largely the same, ensure it has risk_calculator) ---
import json

from openai import OpenAI
from FinancialRiskCalculator import FinancialRiskCalculator
from StockDataProvider import StockDataProvider
from TechnicalIndicatorCalculator import TechnicalIndicatorCalculator

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

