import os
import re
import time
import logging
import threading
import requests
from ai.base_ai_service import BaseAIService, RateLimitError
import ai.ai_prompts as ai_prompts
import response_parser

logger = logging.getLogger(__name__)

_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:generateContent"

# Provider-specific sampling — kept here so other providers stay independent.
# Single combined call: low enough for stable integer scores, high enough
# that the summary doesn't read like a police report.
_TEMPERATURE = 0.4
_MAX_OUTPUT_TOKENS = 1024

# Free-tier RPM caps: flash=10, flash-lite=15. 6.5s spacing keeps us safely under 10 RPM.
_MIN_CALL_INTERVAL_SEC = float(os.getenv("GEMINI_MIN_CALL_INTERVAL_SEC", "6.5"))

# Legacy fallback only: Gemini's responseSchema is a subset of OpenAPI 3.0 — it
# rejects JSON-Schema-only keywords (additionalProperties, minimum, maximum)
# and expects Type values in uppercase form (OBJECT, INTEGER, ...). The primary
# path uses responseJsonSchema, which accepts standard JSON Schema as-is.
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


def _wrap_article(article_text, title=None):
    head = f"<title>\n{title}\n</title>\n" if title else ""
    return f"{head}<article>\n{article_text}\n</article>"


class GeminiRateLimitError(RateLimitError):
    """Raised on HTTP 429. Carries Google's suggested retry delay (seconds) when provided.

    Subclasses the generic ``RateLimitError`` so callers that catch the base class
    (e.g. ``main._with_retry``) still handle Gemini rate limits uniformly.
    """


def _parse_retry_after(resp):
    """Extract a retry delay (seconds) from a 429 response.

    Google returns the delay either as a standard `Retry-After` header or inside the JSON body
    under `error.details[].retryDelay` (e.g. "23s"). Returns None when neither is parseable.
    """
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        details = (resp.json().get("error") or {}).get("details") or []
        for detail in details:
            if "RetryInfo" in detail.get("@type", ""):
                m = re.match(r"([\d.]+)s", detail.get("retryDelay", ""))
                if m:
                    return float(m.group(1))
    except (ValueError, KeyError, AttributeError):
        pass
    return None


class GeminiService(BaseAIService):
    _last_call_at = 0.0
    _stagger_lock = threading.Lock()
    # Flips to True (process-wide) if the API rejects responseJsonSchema,
    # e.g. an older API surface — subsequent calls go straight to the
    # sanitized legacy responseSchema format.
    _use_legacy_schema = False

    def __init__(self, api_key):
        self.api_key = api_key

    def _stagger(self):
        """Block until at least _MIN_CALL_INTERVAL_SEC has elapsed since the previous call."""
        with GeminiService._stagger_lock:
            wait = _MIN_CALL_INTERVAL_SEC - (time.monotonic() - GeminiService._last_call_at)
            if wait > 0:
                logger.debug(f"Staggering Gemini call by {wait:.2f}s")
                time.sleep(wait)
            GeminiService._last_call_at = time.monotonic()

    def _schema_config(self, response_schema):
        if GeminiService._use_legacy_schema:
            return {"responseSchema": _sanitize_schema(response_schema)}
        return {"responseJsonSchema": response_schema}

    def _generate(self, system_prompt, user_content, *, temperature, max_output_tokens, response_schema=None):
        self._stagger()
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            # Gemini 2.5 models enable "thinking" by default and charge thought
            # tokens against maxOutputTokens, which can starve the visible reply.
            "thinkingConfig": {"thinkingBudget": 0},
        }
        if response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config.update(self._schema_config(response_schema))

        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_content}]}],
            "generationConfig": generation_config,
        }
        resp = requests.post(f"{_API_URL}?key={self.api_key}", json=body, timeout=60)
        if resp.status_code == 429:
            retry_after = _parse_retry_after(resp)
            raise GeminiRateLimitError(
                f"429 Too Many Requests for {_MODEL} (retry_after={retry_after})",
                retry_after=retry_after,
            )
        if (
            resp.status_code == 400
            and response_schema is not None
            and not GeminiService._use_legacy_schema
            and "responseJsonSchema" in resp.text
        ):
            logger.warning("Gemini rejected responseJsonSchema — falling back to legacy responseSchema for this process.")
            GeminiService._use_legacy_schema = True
            return self._generate(
                system_prompt, user_content,
                temperature=temperature, max_output_tokens=max_output_tokens,
                response_schema=response_schema,
            )
        if not resp.ok:
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

    def evaluate_and_summarize(self, article_text, title, source_language='es', target_language='en'):
        system_prompt = ai_prompts.get_evaluate_and_summarize_prompt(source_language, target_language)
        text = self._generate(
            system_prompt,
            _wrap_article(article_text, title),
            temperature=_TEMPERATURE,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            response_schema=ai_prompts.EVALUATION_SCHEMA,
        )
        logger.debug(f"Gemini evaluate response: {text}")
        return response_parser.parse_evaluate_and_summarize(text)
