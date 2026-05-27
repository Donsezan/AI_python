import os
import re
import time
import logging
import threading
import requests
from ai.base_ai_service import BaseAIService
import ai.ai_prompts as ai_prompts
import response_parser

logger = logging.getLogger(__name__)

_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{_MODEL}:generateContent"

# Free-tier RPM caps: flash=10, flash-lite=15. 6.5s spacing keeps us safely under 10 RPM.
_MIN_CALL_INTERVAL_SEC = float(os.getenv("GEMINI_MIN_CALL_INTERVAL_SEC", "6.5"))


class GeminiRateLimitError(Exception):
    """Raised on HTTP 429. Carries Google's suggested retry delay (seconds) when provided."""
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


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

    def _generate(self, prompt, json_mode=False):
        self._stagger()
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        if json_mode:
            body["generationConfig"] = {"responseMimeType": "application/json"}
        resp = requests.post(f"{_API_URL}?key={self.api_key}", json=body, timeout=60)
        if resp.status_code == 429:
            retry_after = _parse_retry_after(resp)
            raise GeminiRateLimitError(
                f"429 Too Many Requests for {_MODEL} (retry_after={retry_after})",
                retry_after=retry_after,
            )
        resp.raise_for_status()
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

    def summarize_with_emojis(self, article_text, target_language='en'):
        prompt = ai_prompts.get_summarize_with_emojis_prompt(target_language)
        text = self._generate(f"{prompt}\n\n{article_text}")
        return response_parser.parse_summary_with_emojis(text)

    def evaluate_article(self, article_text):
        prompt = ai_prompts.get_evaluate_article_prompt()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "article_evaluation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "expat_impact": {"type": "integer", "minimum": 1, "maximum": 10, "description": "How relevant or impactful the news is for expatriates (1-10)"},
                        "event_weight": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Significance or uniqueness of the event (1-10)"},
                        "politics": {"type": "integer", "minimum": 0, "maximum": 10, "description": "Non-political/innovation score (0=political, 10=non-political/innovative)"},
                        "timeliness": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Time-sensitivity or urgency (1-10)"},
                        "practical_utility": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Usefulness for reader's daily life (1-10)"}
                    },
                    "required": ["expat_impact", "event_weight", "politics", "timeliness", "practical_utility"],
                    "additionalProperties": False
                }
            }
        }
        full_prompt = f"{prompt} Provide a JSON response with the following schema: {response_format}\n\n{article_text}"
        text = self._generate(full_prompt, json_mode=True)
        logger.debug(f"Gemini evaluate response: {text}")
        return response_parser.parse_evaluate_article(text)
