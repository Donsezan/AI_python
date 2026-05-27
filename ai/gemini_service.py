import os
import logging
import requests
from ai.base_ai_service import BaseAIService, RateLimitError
import ai.ai_prompts as ai_prompts
import response_parser

logger = logging.getLogger(__name__)

_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:generateContent"

# Provider-specific sampling — kept here so other providers stay independent.
_TEMPERATURE_EVALUATE = 0.2
_TEMPERATURE_SUMMARIZE = 0.7
_MAX_OUTPUT_TOKENS_EVALUATE = 256
_MAX_OUTPUT_TOKENS_SUMMARIZE = 512

# Gemini's responseSchema is a subset of OpenAPI 3.0 — it rejects JSON-Schema-only
# keywords (additionalProperties, minimum, maximum) and expects Type values in
# uppercase form (OBJECT, INTEGER, ...).
_GEMINI_SCHEMA_ALLOWED = {
    "type", "format", "description", "nullable", "enum",
    "maxItems", "minItems", "properties", "required",
    "propertyOrdering", "items",
}


def _sanitize_schema(schema):
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k not in _GEMINI_SCHEMA_ALLOWED:
                continue
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            elif k == "properties" and isinstance(v, dict):
                # Keys here are user-defined property names, not schema keywords —
                # keep them verbatim and only sanitize the nested schema values.
                out[k] = {pname: _sanitize_schema(pschema) for pname, pschema in v.items()}
            else:
                out[k] = _sanitize_schema(v)
        return out
    if isinstance(schema, list):
        return [_sanitize_schema(v) for v in schema]
    return schema


def _wrap_article(article_text):
    return f"<article>\n{article_text}\n</article>"


def _parse_retry_delay(payload):
    """Extract retryDelay seconds from a Gemini error payload's RetryInfo detail."""
    details = ((payload or {}).get("error") or {}).get("details") or []
    for detail in details:
        if detail.get("@type", "").endswith("/google.rpc.RetryInfo"):
            delay = detail.get("retryDelay", "")
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    return None
    return None


class GeminiService(BaseAIService):
    def __init__(self, api_key):
        self.api_key = api_key

    def _generate(self, system_prompt, user_content, *, temperature, max_output_tokens, response_schema=None):
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            # Gemini 2.5 models enable "thinking" by default and charge thought
            # tokens against maxOutputTokens, which can starve the visible reply.
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = _sanitize_schema(response_schema)

        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_content}]}],
            "generationConfig": generation_config,
        }
        resp = requests.post(f"{_API_URL}?key={self.api_key}", json=body, timeout=60)
        if not resp.ok:
            if resp.status_code == 429:
                try:
                    payload = resp.json()
                except ValueError:
                    payload = None
                retry_after = _parse_retry_delay(payload)
                suffix = f", retry in {retry_after:.0f}s" if retry_after else ""
                raise RateLimitError(f"Gemini rate limit (429){suffix}", retry_after=retry_after)
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text}")
        data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise RuntimeError(f"Gemini returned no candidates: {reason}")

        cand = candidates[0]
        finish = cand.get("finishReason")
        if finish and finish not in ("STOP", "MAX_TOKENS"):
            raise RuntimeError(f"Gemini finishReason={finish}")

        parts = ((cand.get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError("Gemini returned empty text")
        return text

    def summarize_with_emojis(self, article_text, target_language='en', source_language='es'):
        system_prompt = ai_prompts.get_summarize_with_emojis_prompt(target_language, source_language)
        text = self._generate(
            system_prompt,
            _wrap_article(article_text),
            temperature=_TEMPERATURE_SUMMARIZE,
            max_output_tokens=_MAX_OUTPUT_TOKENS_SUMMARIZE,
        )
        return response_parser.parse_summary_with_emojis(text)

    def evaluate_article(self, article_text, source_language='es'):
        system_prompt = ai_prompts.get_evaluate_article_prompt(source_language)
        text = self._generate(
            system_prompt,
            _wrap_article(article_text),
            temperature=_TEMPERATURE_EVALUATE,
            max_output_tokens=_MAX_OUTPUT_TOKENS_EVALUATE,
            response_schema=ai_prompts.EVALUATION_SCHEMA,
        )
        logger.debug(f"Gemini evaluate response: {text}")
        return response_parser.parse_evaluate_article(text)
