import requests
from ai.base_ai_service import BaseAIService
import ai.ai_prompts as ai_prompts
import response_parser

_API_URL = "http://localhost:1234/v1/chat/completions"
_MODEL = "microsoft/phi-4-reasoning-plus"

# Provider-specific sampling — kept here so other providers stay independent.
_TEMPERATURE_EVALUATE = 0.2
_TEMPERATURE_SUMMARIZE = 0.7


def _wrap_article(article_text):
    return f"<article>\n{article_text}\n</article>"


def _json_schema_response_format(schema, name="article_evaluation"):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema},
    }


class OpenAIService(BaseAIService):
    def __init__(self):
        self.headers = {"Authorization": "Bearer lm-studio", "Content-Type": "application/json"}

    def _chat(self, messages, *, temperature, response_format=None, model=_MODEL):
        body = {
            "model": model,
            "messages": messages,
            "response_format": response_format or {"type": "text"},
            "temperature": temperature,
        }
        resp = requests.post(_API_URL, json=body, headers=self.headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def summarize_with_emojis(self, article_text, target_language='en', source_language='es'):
        messages = [
            {"role": "system", "content": ai_prompts.get_summarize_with_emojis_prompt(target_language, source_language)},
            {"role": "user", "content": _wrap_article(article_text)},
        ]
        return response_parser.parse_summary_with_emojis(
            self._chat(messages, temperature=_TEMPERATURE_SUMMARIZE)
        )

    def evaluate_article(self, article_text, source_language='es'):
        messages = [
            {"role": "system", "content": ai_prompts.get_evaluate_article_prompt(source_language)},
            {"role": "user", "content": _wrap_article(article_text)},
        ]
        text = self._chat(
            messages,
            temperature=_TEMPERATURE_EVALUATE,
            response_format=_json_schema_response_format(ai_prompts.EVALUATION_SCHEMA),
        )
        return response_parser.parse_evaluate_article(text)
