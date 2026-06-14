import requests
from ai.base_ai_service import BaseAIService
import ai.ai_prompts as ai_prompts
import response_parser

_API_URL = "http://localhost:1234/v1/chat/completions"
_MODEL = "microsoft/phi-4-reasoning-plus"

# Provider-specific sampling — kept here so other providers stay independent.
_TEMPERATURE = 0.4


def _wrap_article(article_text, title=None):
    head = f"<title>\n{title}\n</title>\n" if title else ""
    return f"{head}<article>\n{article_text}\n</article>"


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

    def evaluate_and_summarize(self, article_text, title, source_language='es', target_language='en'):
        messages = [
            {"role": "system", "content": ai_prompts.get_evaluate_and_summarize_prompt(source_language, target_language)},
            {"role": "user", "content": _wrap_article(article_text, title)},
        ]
        text = self._chat(
            messages,
            temperature=_TEMPERATURE,
            response_format=_json_schema_response_format(ai_prompts.EVALUATION_SCHEMA),
        )
        return response_parser.parse_evaluate_and_summarize(text)
